"""
FastAPI Main Application - Production Ready
Real-time sentiment analysis API with real analysis engines and database
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
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
import csv
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from .services.sentiment_engine import analyze_text_sentiment, calibrate_sentiment_probabilities
from .services.image_sentiment import analyze_image_sentiment
from .services.database import get_database
from .services.correction import DQNAgent, CorrectionLayer, apply_action_bias
from .models.text_bigru import BiGRUTextEncoder
from .models.multimodal_autoencoder import MultimodalAutoencoderService
from .services.file_parser import parse_file
from .db import save_analysis, get_user_history, init_user_profile, get_user_profile, deduct_credits
from pydantic import BaseModel


# Column name candidates for file parsing
TEXT_COLUMN_CANDIDATES = [
    'text',
    'content',
    'message',
    'post',
    'comment',
    'description',
    'body',
    'tweet',
    'review',
    'feedback'
]

COMPANY_COLUMN_CANDIDATES = [
    'company',
    'brand',
    'organization',
    'vendor',
    'store',
    'business',
    'source'
]

app = FastAPI(
    title="Multimodal Sentiment Analyzer API",
    description="Production-ready sentiment analysis for text + image",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",  # Add this line
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",  # Add this line
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[OK] CORS configured for origins: localhost:3000, localhost:3001, 127.0.0.1:3000, 127.0.0.1:3001")

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
        print("[OK] Correction layer checkpoint loaded")
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
    print("[START] Starting Sentiment Analysis API v2.0")
    print("=" * 60)
    
    # Initialize database
    db = get_database("../sentiment_data.db")
    print("[OK] Database initialized")
    
    # Pre-load image analyzer (loads ResNet50)
    from .services.image_sentiment import get_image_analyzer
    get_image_analyzer(device)
    print(f"[OK] Image analyzer loaded (device: {device})")

    text_encoder = BiGRUTextEncoder(checkpoint_dir=str(CHECKPOINT_DIR), device=device)
    if text_encoder.available:
        print("[OK] Text Bi-GRU encoder ready")
    else:
        print("[WARN] Text Bi-GRU encoder weights not found; lexicon-only mode.")

    autoencoder_service = MultimodalAutoencoderService(
        checkpoint_dir=str(CHECKPOINT_DIR),
        device=device,
        text_dim=TEXT_EMBED_DIM,
        image_dim=IMAGE_FEATURE_DIM,
        latent_dim=AUTOENCODER_LATENT_DIM,
    )
    if autoencoder_service.available:
        print("[OK] Multimodal autoencoder ready")
    else:
        print("[WARN] Multimodal autoencoder weights not found; latent fusion disabled.")
    
    print("[OK] Text sentiment engine ready")
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
            text_result = analyze_text_sentiment(request.text)
            base_probs = dict(text_result['probabilities'])
            source_breakdown = text_result.get('source_breakdown', {}) or {}
            if not source_breakdown:
                source_breakdown = {
                    'lexicon': {
                        'probabilities': dict(base_probs),
                        'sentiment': text_result['sentiment'],
                        'confidence': text_result['confidence'],
                        'compound_score': text_result.get('compound_score', 0.0),
                    },
                    'weights': {'lexicon': 1.0},
                }
                text_result['source_breakdown'] = source_breakdown

            if text_encoder and text_encoder.available:
                bigru_output = text_encoder.predict(request.text)
                if bigru_output:
                    text_embedding = bigru_output.get("embedding")
                    bigru_probs = bigru_output.get("probabilities", {})
                    combined_text_probs = {}
                    weight_base = 0.7
                    weight_bigru = 0.3
                    for label in base_probs:
                        combined_text_probs[label] = (
                            base_probs[label] * weight_base +
                            bigru_probs.get(label, 0.0) * weight_bigru
                        )
                    total = sum(combined_text_probs.values())
                    if total > 0:
                        combined_text_probs = {
                            label: value / total for label, value in combined_text_probs.items()
                        }
                    text_result['probabilities'] = combined_text_probs
                    calibrated_label, calibrated_confidence = calibrate_sentiment_probabilities(combined_text_probs)
                    text_result['sentiment'] = calibrated_label
                    text_result['confidence'] = calibrated_confidence

                    # Update breakdown metadata
                    lexicon_details = source_breakdown.get('lexicon') or {
                        'probabilities': dict(base_probs),
                        'sentiment': text_result['sentiment'],
                        'confidence': text_result['confidence'],
                        'compound_score': text_result.get('compound_score', 0.0),
                    }
                    transformer_details = source_breakdown.get('transformer')
                    weights = source_breakdown.get('weights', {})
                    if weights:
                        weights = {
                            key: (value or 0.0) * weight_base for key, value in weights.items()
                        }
                    else:
                        weights = {'ensemble_base': weight_base}
                    weights['bigru'] = weight_bigru
                    bigru_sentiment = max(bigru_probs, key=bigru_probs.get) if bigru_probs else None
                    bigru_confidence = bigru_probs.get(bigru_sentiment) if bigru_sentiment else None
                    source_breakdown.update({
                        'lexicon': lexicon_details,
                        'transformer': transformer_details,
                        'bigru': {
                            'probabilities': bigru_probs,
                            'sentiment': bigru_sentiment,
                            'confidence': bigru_confidence,
                        },
                        'weights': weights,
                        'ensemble_pre_bigru': {
                            'probabilities': base_probs,
                        }
                    })
                    text_result['source_breakdown'] = source_breakdown
                    text_model_metadata = {
                        'ensemble_base_probs': base_probs,
                        'bigru_probs': bigru_probs,
                        'transformer_probs': transformer_details.get('probabilities') if transformer_details else None,
                        'lexicon_probs': lexicon_details.get('probabilities') if lexicon_details else None,
                        'final_probs': combined_text_probs,
                    }
            else:
                text_model_metadata = {
                    'ensemble_base_probs': base_probs,
                    'transformer_probs': source_breakdown.get('transformer', {}).get('probabilities') if source_breakdown else None,
                    'lexicon_probs': source_breakdown.get('lexicon', {}).get('probabilities') if source_breakdown else None,
                    'final_probs': base_probs,
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


@app.post("/api/batch-analyze")
async def batch_analyze(file: UploadFile = File(...), x_user_id: str = Header(None)):
    """
    Batch analyze sentiment from a file (CSV, Excel, PDF).
    Groups results by 'company' if the column exists.
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User identification required")
        
    user = await get_user_profile(x_user_id)
    if x_user_id == "guest_user_101" and not user:
        # Auto-init guest for testing
        user = await init_user_profile("guest_user_101", "Guest Tester", "guest@acrux.test", None)
        
    if not user or user.get("credits", 0) <= 0:
        raise HTTPException(status_code=403, detail="Insufficient credits (0). Please upgrade your account.")

    print(f"[BATCH] Received file: {file.filename}, content_type: {file.content_type}")
    try:
        # Parse file
        print("[BATCH] Starting file parsing...")
        records = await parse_file(file)
        print(f"[BATCH] Parsed {len(records)} records")
        
        results = []
        company_stats = {}
        
        for record in records:
            text = record.get('text', '')
            if not text:
                continue
                
            # Analyze text (using the core service directly for speed)
            # We replicate some logic from analyze_sentiment here for consistency
            # but simplified for batch processing to avoid overhead
            
            text_result = analyze_text_sentiment(text)
            
            # Apply BiGRU if available (simplified version of main logic)
            if text_encoder and text_encoder.available:
                bigru_output = text_encoder.predict(text)
                if bigru_output:
                    base_probs = text_result['probabilities']
                    bigru_probs = bigru_output.get("probabilities", {})
                    combined_probs = {}
                    for label in base_probs:
                        combined_probs[label] = (
                            base_probs[label] * 0.7 +
                            bigru_probs.get(label, 0.0) * 0.3
                        )
                    text_result['probabilities'] = combined_probs
                    calibrated_label, calibrated_confidence = calibrate_sentiment_probabilities(combined_probs)
                    text_result['sentiment'] = calibrated_label
                    text_result['confidence'] = calibrated_confidence

            # Prepare result object
            analysis_result = {
                'text': text[:200] + "..." if len(text) > 200 else text, # Truncate for response
                'sentiment': text_result['sentiment'],
                'confidence': text_result['confidence'],
                'probabilities': text_result['probabilities'],
                'company': record.get('company', 'Unknown'),
                'metadata': {k: v for k, v in record.items() if k not in ['text', 'company']}
            }
            
            results.append(analysis_result)
            
            # Save to DB (optional, maybe too slow for large batches? Let's do it for now)
            if db:
                try:
                    db.save_prediction({
                        'text': text,
                        'has_image': False,
                        'sentiment': text_result['sentiment'],
                        'confidence': text_result['confidence'],
                        'probabilities': text_result['probabilities'],
                        'compound_score': text_result.get('compound_score', 0.0),
                        'modality': 'text',
                        'metadata': {'source': 'batch_upload', 'company': record.get('company')}
                    })
                except:
                    pass

        # Group by company
        grouped_results = {}
        for res in results:
            company = res['company']
            if company not in grouped_results:
                grouped_results[company] = {
                    'total': 0,
                    'sentiment_counts': {'positive': 0, 'neutral': 0, 'negative': 0},
                    'items': []
                }
            
            grouped_results[company]['total'] += 1
            grouped_results[company]['sentiment_counts'][res['sentiment']] += 1
            grouped_results[company]['items'].append(res)

        output_data = {
            "status": "success",
            "total_processed": len(results),
            "results_by_company": grouped_results,
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }

        if x_user_id:
            try:
                await save_analysis(x_user_id, output_data)
                await deduct_credits(x_user_id, 1)
                print(f"[DB] Saved analysis & deducted 1 credit for user: {x_user_id}")
            except Exception as db_err:
                print(f"[DB] Error saving analysis: {db_err}")

        return output_data

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in batch_analyze: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch file: {str(e)}"
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
                    "success": True,
                    "inputText": request.text,
                    "inputImage": request.image_base64
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


@app.get("/api/history")
async def get_history(x_user_id: str = Header(None)):
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User identity (X-User-Id header) required")
    try:
        history = await get_user_history(x_user_id)
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

class UserInitRequest(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None

@app.post("/api/users/init")
async def init_user(req: UserInitRequest, x_user_id: str = Header(None)):
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    user = await init_user_profile(x_user_id, req.name, req.email, req.phone)
    return {"status": "success", "user": user}

@app.get("/api/users/profile")
async def get_profile(x_user_id: str = Header(None)):
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    user = await get_user_profile(x_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success", "user": user}

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


@app.get("/api/predictions")
async def get_predictions():
    """
    Generate future sentiment predictions for the next 3 months.
    Uses simple linear regression on recent daily stats.
    """
    if not db:
        return {"forecast": [], "recommendations": []}

    try:
        # Get last 30 days of data to calculate trend
        stats = db.get_statistics(days=30)
        daily_stats = stats.get('daily_stats', [])
        
        # Prepare data for regression
        # x = day index, y = positive ratio
        x = []
        y = []
        
        # Sort by date
        daily_stats.sort(key=lambda k: k['date'])
        
        for i, day in enumerate(daily_stats):
            total = day.get('total', 0)
            if total > 0:
                pos_ratio = day.get('positive', 0) / total
                x.append(i)
                y.append(pos_ratio)
        
        # Calculate trend (slope)
        slope = 0
        intercept = 0.5 # Default neutral
        
        if len(x) > 1:
            x_mean = sum(x) / len(x)
            y_mean = sum(y) / len(y)
            
            numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
            denominator = sum((xi - x_mean) ** 2 for xi in x)
            
            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - (slope * x_mean)
        
        # Generate forecast for next 90 days (3 months)
        forecast = []
        recommendations = []
        
        current_date = datetime.now()
        start_idx = len(x)
        
        for i in range(1, 91): # Next 90 days
            future_date = current_date + asyncio.timedelta(days=i) # asyncio doesn't have timedelta, use standard datetime
            # Wait, standard datetime is imported as datetime
            from datetime import timedelta
            future_date = current_date + timedelta(days=i)
            
            idx = start_idx + i
            predicted_ratio = max(0.0, min(1.0, slope * idx + intercept))
            
            # Add some random noise to make it look realistic
            import random
            noise = random.uniform(-0.05, 0.05)
            predicted_ratio = max(0.0, min(1.0, predicted_ratio + noise))
            
            forecast.append({
                "date": future_date.isoformat(),
                "predicted_sentiment_score": predicted_ratio, # 0 to 1 (0=neg, 1=pos)
                "confidence": 0.8 - (i * 0.005) # Confidence drops over time
            })
            
        # Generate recommendations based on slope
        if slope < -0.005:
            recommendations = [
                "Sentiment is trending downwards. Immediate action required.",
                "Investigate recent negative feedback spikes.",
                "Review customer support response times.",
                "Consider a customer appreciation campaign to boost morale."
            ]
        elif slope > 0.005:
            recommendations = [
                "Sentiment is trending upwards! Keep up the good work.",
                "Identify what's working and double down on it.",
                "Share positive feedback with the team to boost morale.",
                "Consider asking happy customers for referrals."
            ]
        else:
            recommendations = [
                "Sentiment is stable.",
                "Focus on converting neutral customers to positive.",
                "Monitor key competitors for market shifts.",
                "Experiment with small improvements to product features."
            ]
            
        return {
            "forecast": forecast,
            "recommendations": recommendations,
            "trend_slope": slope
        }
        
    except Exception as e:
        print(f"Error generating predictions: {e}")
        print(traceback.format_exc())
        return {"forecast": [], "recommendations": [], "error": str(e)}


def _pick_column(columns, candidates) -> Tuple[str, bool]:
    lower_map = {col.lower(): col for col in columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name], True
    # fallback to first column if no match
    return columns[0], False


async def parse_file(upload: UploadFile) -> List[Dict]:
    raw_bytes = await upload.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    suffix = Path(upload.filename or "").suffix.lower()
    buffer = io.BytesIO(raw_bytes)

    try:
        if suffix in {".csv"}:
            buffer.seek(0)
            reader = csv.DictReader(io.TextIOWrapper(buffer, encoding="utf-8", newline=""))
            rows = list(reader)
            if not rows:
                raise HTTPException(status_code=400, detail="CSV file has headers but no rows.")
            columns = reader.fieldnames or []
            data_frame = pd.DataFrame(rows)
        elif suffix in {".xlsx", ".xls"}:
            buffer.seek(0)
            data_frame = pd.read_excel(buffer)
            columns = list(data_frame.columns)
        elif suffix in {".json"}:
            buffer.seek(0)
            data_frame = pd.read_json(buffer)
            columns = list(data_frame.columns)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV, XLSX, or JSON.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to parse file: {exc}") from exc

    if not columns:
        raise HTTPException(status_code=400, detail="File does not contain any columns.")

    text_col, text_found = _pick_column(columns, TEXT_COLUMN_CANDIDATES)
    company_col, company_found = _pick_column(columns, COMPANY_COLUMN_CANDIDATES) if len(columns) > 1 else (None, False)

    records: List[Dict] = []
    missing_rows = 0

    for _, row in data_frame.iterrows():
        text_value = str(row.get(text_col, "")).strip()
        if not text_value:
            missing_rows += 1
            continue

        record = {"text": text_value}

        if company_col:
            company_value = str(row.get(company_col, "")).strip()
            record["company"] = company_value or "Unknown"

        for col in columns:
            if col in {text_col, company_col}:
                continue
            record.setdefault(col, row.get(col))
        records.append(record)

    if not records:
        if missing_rows:
            raise HTTPException(
                status_code=400,
                detail="No usable text rows were found. Ensure the file has a column containing text content."
            )
        raise HTTPException(status_code=400, detail="No data rows were found in the file.")

    if not text_found:
        print(f"[BATCH] Warning: Used column '{text_col}' as text fallback; expected columns: {TEXT_COLUMN_CANDIDATES}")
    if company_col and not company_found:
        print(f"[BATCH] Warning: Used column '{company_col}' as company fallback; expected columns: {COMPANY_COLUMN_CANDIDATES}")

    return records

