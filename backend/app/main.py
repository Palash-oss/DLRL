"""
FastAPI Main Application - Production Ready
Real-time sentiment analysis API with real analysis engines and database
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import asyncio
from datetime import datetime
from pathlib import Path
import traceback

from .services.sentiment_engine import analyze_sentiment_lexicon
from .services.image_sentiment import analyze_image_sentiment
from .services.database import get_database
from .services.correction import DQNAgent, CorrectionLayer, apply_action_bias
from .models.text_bigru import BiGRUTextEncoder
from .models.multimodal_autoencoder import MultimodalAutoencoderService

app = FastAPI(
    title="Multimodal Sentiment Analyzer API",
    description="Production-ready sentiment analysis for text + image",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TEXT_EMBED_DIM = 128
IMAGE_FEATURE_DIM = 5
AUTOENCODER_LATENT_DIM = 16
RL_STATE_DIM = 12 + AUTOENCODER_LATENT_DIM
RL_ACTIONS = ["keep", "positive", "neutral", "negative"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_ACTION_MAP = {
    label: RL_ACTIONS.index(label)
    for label in SENTIMENT_LABELS
    if label in RL_ACTIONS
}

correction_layer = CorrectionLayer(RL_STATE_DIM, num_classes=len(RL_ACTIONS)).to(device)
correction_optimizer = torch.optim.Adam(correction_layer.parameters(), lr=5e-4)
rl_agent = DQNAgent(state_dim=RL_STATE_DIM, device=device, checkpoint_dir=str(CHECKPOINT_DIR))

correction_weights_path = CHECKPOINT_DIR / "correction_layer.pt"
if correction_weights_path.exists():
    try:
        correction_layer.load_state_dict(torch.load(correction_weights_path, map_location=device))
        print("✓ Correction layer checkpoint loaded")
    except Exception as exc:
        print(f"Warning: correction layer checkpoint mismatch ({exc}); using fresh weights.")

# Database and auxiliary models
db = None
text_encoder = None
autoencoder_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global db, text_encoder, autoencoder_service
    print("=" * 60)
    print("🚀 Starting Sentiment Analysis API v2.0")
    print("=" * 60)
    
    # Initialize database
    db = get_database("../sentiment_data.db")
    print("✓ Database initialized")
    
    # Pre-load image analyzer (loads ResNet50)
    from .services.image_sentiment import get_image_analyzer
    get_image_analyzer(device)
    print(f"✓ Image analyzer loaded (device: {device})")

    text_encoder = BiGRUTextEncoder(checkpoint_dir=str(CHECKPOINT_DIR), device=device)
    if text_encoder.available:
        print("✓ Text Bi-GRU encoder ready")
    else:
        print("⚠ Text Bi-GRU encoder weights not found; lexicon-only mode.")

    autoencoder_service = MultimodalAutoencoderService(
        checkpoint_dir=str(CHECKPOINT_DIR),
        device=device,
        text_dim=TEXT_EMBED_DIM,
        image_dim=IMAGE_FEATURE_DIM,
        latent_dim=AUTOENCODER_LATENT_DIM,
    )
    if autoencoder_service.available:
        print("✓ Multimodal autoencoder ready")
    else:
        print("⚠ Multimodal autoencoder weights not found; latent fusion disabled.")
    
    print("✓ Text sentiment engine ready")
    print("=" * 60)
    print("API is ready to accept requests!")
    print("=" * 60)


# Request/Response models
class TextRequest(BaseModel):
    text: str


class MultimodalRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict
    explainability: Optional[dict] = None
    metadata: Optional[dict] = None


class FeedbackRequest(BaseModel):
    state: List[float]
    action: int
    correct: bool


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✓ WebSocket client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"✗ WebSocket client disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except:
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)


manager = ConnectionManager()


def _extract_probabilities(source) -> List[float]:
    if not source:
        return [0.0, 0.0, 0.0]
    if isinstance(source, dict) and 'probabilities' in source:
        data = source['probabilities']
    else:
        data = source or {}
    return [
        float(data.get('negative', 0.0)),
        float(data.get('neutral', 0.0)),
        float(data.get('positive', 0.0)),
    ]


def build_state_vector(
    combined_probs: dict,
    text_result: Optional[dict],
    image_result: Optional[dict],
    compound_score: float,
    text_length: int,
    has_image: bool,
    latent_features: Optional[List[float]] = None,
) -> List[float]:
    state = []
    state.extend(_extract_probabilities(combined_probs))
    state.extend(_extract_probabilities(text_result))
    state.extend(_extract_probabilities(image_result))
    state.append(float(compound_score))
    state.append(min(text_length / 500.0, 1.0))
    state.append(1.0 if has_image else 0.0)
    if latent_features:
        clipped = latent_features[:AUTOENCODER_LATENT_DIM]
        if len(clipped) < AUTOENCODER_LATENT_DIM:
            clipped = clipped + [0.0] * (AUTOENCODER_LATENT_DIM - len(clipped))
        state.extend(clipped)
    else:
        state.extend([0.0] * AUTOENCODER_LATENT_DIM)
    return state


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Multimodal Sentiment Analyzer API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "analyze": "/api/analyze",
            "dashboard": "/api/dashboard",
            "websocket": "/api/sentiment-stream"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    stats = db.get_statistics(days=1) if db else {}
    return {
        "status": "healthy",
        "device": str(device),
        "database": "connected" if db else "not initialized",
        "total_predictions": stats.get('total_predictions', 0)
    }


@app.post("/api/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: MultimodalRequest):
    """
    Analyze sentiment from text and/or image.
    Supports text-only, image-only, or multimodal.
    """
    try:
        if not request.text and not request.image_base64:
            raise HTTPException(
                status_code=400,
                detail="At least one modality (text or image) is required"
            )
        
        text_result = None
        image_result = None
        modality = None
        text_embedding = None
        text_model_metadata = None
        image_feature_vector = None
        latent_vector = None
        
        # Analyze text
        if request.text:
            if len(request.text) > 10000:
                raise HTTPException(
                    status_code=400,
                    detail="Text too long. Maximum 10,000 characters."
                )
            text_result = analyze_sentiment_lexicon(request.text)
            if text_encoder and text_encoder.available:
                bigru_output = text_encoder.predict(request.text)
                if bigru_output:
                    text_embedding = bigru_output.get("embedding")
                    bigru_probs = bigru_output.get("probabilities", {})
                    lexicon_probs = dict(text_result['probabilities'])
                    combined_text_probs = {}
                    for label in lexicon_probs:
                        combined_text_probs[label] = (
                            lexicon_probs[label] + bigru_probs.get(label, 0.0)
                        ) / 2.0
                    text_result['probabilities'] = combined_text_probs
                    text_result['sentiment'] = max(combined_text_probs, key=combined_text_probs.get)
                    text_result['confidence'] = combined_text_probs[text_result['sentiment']]
                    text_model_metadata = {
                        'lexicon_probs': lexicon_probs,
                        'bigru_probs': bigru_probs,
                        'combined_probs': combined_text_probs,
                    }
            modality = "text"
        
        # Analyze image
        if request.image_base64:
            try:
                # Decode and validate image
                image_bytes = base64.b64decode(request.image_base64)
                if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
                    raise HTTPException(
                        status_code=400,
                        detail="Image too large. Maximum 10MB."
                    )
                
                image = Image.open(io.BytesIO(image_bytes))
                image_result = analyze_image_sentiment(image, device)
                color_stats = image_result.get('color_analysis', {}) if image_result else {}
                if color_stats:
                    image_feature_vector = [
                        float(color_stats.get('brightness', 0.0) / 255.0),
                        float(color_stats.get('saturation', 0.0) / 255.0),
                        float(color_stats.get('warmth', 0.0)),
                        float(color_stats.get('coolness', 0.0)),
                        float(color_stats.get('color_score', 0.0)),
                    ]
                modality = "image" if not text_result else "multimodal"
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image data: {str(e)}"
                )
        
        # Combine results if multimodal
        if text_result and image_result:
            # Weighted average: 60% text, 40% image
            text_weight = 0.6
            image_weight = 0.4
            
            combined_probs = {
                'negative': text_result['probabilities']['negative'] * text_weight + 
                           image_result['probabilities']['negative'] * image_weight,
                'neutral': text_result['probabilities']['neutral'] * text_weight + 
                          image_result['probabilities']['neutral'] * image_weight,
                'positive': text_result['probabilities']['positive'] * text_weight + 
                           image_result['probabilities']['positive'] * image_weight,
            }
            
            # Determine final sentiment
            sentiment_label = max(combined_probs, key=combined_probs.get)
            confidence = combined_probs[sentiment_label]
            
            # Combine compound scores
            compound_score = (text_result.get('compound_score', 0) * text_weight + 
                            image_result.get('combined_score', 0) * image_weight)
            
            result = {
                'sentiment': sentiment_label,
                'confidence': confidence,
                'probabilities': combined_probs,
                'compound_score': compound_score,
                'text_analysis': text_result,
                'image_analysis': image_result
            }
        else:
            result = text_result or image_result
        
        combined_probs = result['probabilities']
        compound_score = result.get('compound_score', 0.0)
        text_length = len(request.text) if request.text else 0
        has_image = request.image_base64 is not None

        text_vector_for_auto = None
        if text_embedding:
            text_vector_for_auto = list(text_embedding[:TEXT_EMBED_DIM])
            if len(text_vector_for_auto) < TEXT_EMBED_DIM:
                text_vector_for_auto.extend([0.0] * (TEXT_EMBED_DIM - len(text_vector_for_auto)))

        if autoencoder_service and autoencoder_service.available:
            latent_vector = autoencoder_service.get_latent(text_vector_for_auto, image_feature_vector)

        state_vector = build_state_vector(
            combined_probs,
            text_result,
            image_result,
            compound_score,
            text_length,
            has_image,
            latent_vector,
        )
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw_correction_logits = correction_layer(state_tensor.to(device)).squeeze(0).cpu()

        if (
            raw_correction_logits.shape[0] == len(RL_ACTIONS)
            and len(SENTIMENT_ACTION_MAP) == len(SENTIMENT_LABELS)
        ):
            correction_logits = torch.stack(
                [
                    raw_correction_logits[SENTIMENT_ACTION_MAP['negative']],
                    raw_correction_logits[SENTIMENT_ACTION_MAP['neutral']],
                    raw_correction_logits[SENTIMENT_ACTION_MAP['positive']],
                ]
            )
        else:
            correction_logits = raw_correction_logits

        base_tensor = torch.tensor(
            [
                combined_probs['negative'],
                combined_probs['neutral'],
                combined_probs['positive'],
            ],
            dtype=torch.float32,
        )
        corrected_probs = torch.softmax(torch.log(base_tensor + 1e-6) + correction_logits, dim=0)
        action_idx = rl_agent.select_action(state_tensor)
        corrected_probs = apply_action_bias(corrected_probs, action_idx)
        corrected_dict = {
            'negative': float(corrected_probs[0]),
            'neutral': float(corrected_probs[1]),
            'positive': float(corrected_probs[2]),
        }
        result['probabilities'] = corrected_dict
        result['sentiment'] = max(corrected_dict, key=corrected_dict.get)
        result['confidence'] = corrected_dict[result['sentiment']]
        # Add explainability
        explainability = {}
        if text_result and 'attention_weights' in text_result:
            explainability['text_attention'] = {
                'tokens': text_result['tokens'],
                'attention_scores': text_result['attention_weights'],
                'highlighted_tokens': [
                    {'token': tok, 'score': score}
                    for tok, score in zip(text_result['tokens'], text_result['attention_weights'])
                    if score > 0.3
                ]
            }
        
        if image_result and 'color_analysis' in image_result:
            explainability['image_analysis'] = image_result['color_analysis']
        
        # Save to database
        if db:
            try:
                db_data = {
                    'text': request.text,
                    'has_image': request.image_base64 is not None,
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'compound_score': result.get('compound_score', 0.0),
                    'modality': modality,
                    'metadata': {
                        'text_length': text_length,
                        'has_explainability': bool(explainability),
                        'rl_action': RL_ACTIONS[action_idx],
                        'text_model_used': bool(text_model_metadata),
                        'auto_latent_norm': float(np.linalg.norm(latent_vector)) if latent_vector else 0.0,
                    }
                }
                db.save_prediction(db_data)
            except Exception as e:
                print(f"Warning: Failed to save to database: {e}")
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'timestamp': datetime.now().isoformat(),
            'modality': modality
        })
        
        response = SentimentResponse(
            sentiment=result['sentiment'],
            confidence=float(result['confidence']),
            probabilities={k: float(v) for k, v in result['probabilities'].items()},
            explainability=explainability if explainability else None,
            metadata={
                'modality': modality,
                'compound_score': compound_score,
                'rl_state': state_vector,
                'rl_action': action_idx,
                'rl_action_label': RL_ACTIONS[action_idx],
                'text_models': text_model_metadata,
                'autoencoder_latent': latent_vector,
            }
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/upload-post")
async def upload_post(
    text: Optional[str] = None,
    image: Optional[UploadFile] = File(None)
):
    """Upload a social media post (text + image file)."""
    try:
        image_base64 = None
        if image:
            image_bytes = await image.read()
            if len(image_bytes) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail="Image too large. Maximum 10MB."
                )
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        request = MultimodalRequest(text=text, image_base64=image_base64)
        return await analyze_sentiment(request)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing upload: {str(e)}"
        )


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    reward = 1.0 if feedback.correct else -1.0
    rl_agent.push_transition(feedback.state, feedback.action, reward, feedback.state, True)
    loss = rl_agent.optimize()
    if rl_agent.steps_done % 400 == 0:
        rl_agent.update_target()
        rl_agent.save()

    state_tensor = torch.tensor([feedback.state], dtype=torch.float32, device=device)
    logits = correction_layer(state_tensor)
    target = torch.tensor([feedback.action], dtype=torch.long, device=device)
    correction_optimizer.zero_grad()
    correction_loss = reward * F.cross_entropy(logits, target)
    correction_loss.backward()
    correction_optimizer.step()
    torch.save(correction_layer.state_dict(), correction_weights_path)

    if db:
        try:
            db.save_feedback({'state': feedback.state, 'action': feedback.action, 'reward': reward})
        except Exception as exc:
            print(f"Warning: failed to persist feedback: {exc}")

    return {
        "status": "ok",
        "reward": reward,
        "loss": loss,
        "buffer_size": len(rl_agent.buffer),
    }


@app.websocket("/api/sentiment-stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time sentiment streaming."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive data
            data = await websocket.receive_json()
            
            # Process request
            request = MultimodalRequest(**data)
            
            # Analyze (reuse main analysis logic)
            try:
                result = await analyze_sentiment(request)
                
                response = {
                    "sentiment": result.sentiment,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                await manager.send_personal_message(response, websocket)
            
            except Exception as e:
                await manager.send_personal_message(
                    {"error": str(e), "success": False},
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.get("/api/dashboard")
async def get_dashboard_data():
    """Get dashboard statistics and recent predictions."""
    if not db:
        return {
            "total_predictions": 0,
            "sentiment_distribution": {},
            "recent_predictions": [],
            "daily_stats": []
        }
    
    try:
        stats = db.get_statistics(days=7)
        recent = db.get_recent_predictions(limit=20)
        
        return {
            "total_predictions": stats.get('total_predictions', 0),
            "sentiment_distribution": stats.get('sentiment_distribution', {}),
            "modality_distribution": stats.get('modality_distribution', {}),
            "avg_confidence": stats.get('avg_confidence', 0.0),
            "daily_stats": stats.get('daily_stats', []),
            "recent_predictions": recent,
            "feedback_curve": stats.get('feedback_curve', [])
        }
    except Exception as e:
        print(f"Error fetching dashboard data: {e}")
        return {
            "total_predictions": 0,
            "sentiment_distribution": {},
            "recent_predictions": [],
            "error": str(e)
        }


@app.get("/api/stats")
async def get_stats():
    """Get quick statistics."""
    if not db:
        return {"total": 0}
    
    stats = db.get_statistics(days=30)
    return {
        "total_predictions": stats.get('total_predictions', 0),
        "sentiment_distribution": stats.get('sentiment_distribution', {}),
        "avg_confidence": round(stats.get('avg_confidence', 0.0), 3)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

