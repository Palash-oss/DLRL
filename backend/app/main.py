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
import numpy as np
from PIL import Image
import io
import base64
import asyncio
from datetime import datetime
import traceback

from .services.sentiment_engine import analyze_sentiment_lexicon
from .services.image_sentiment import analyze_image_sentiment
from .services.database import get_database

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

# Database
db = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global db
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
        
        # Analyze text
        if request.text:
            if len(request.text) > 10000:
                raise HTTPException(
                    status_code=400,
                    detail="Text too long. Maximum 10,000 characters."
                )
            text_result = analyze_sentiment_lexicon(request.text)
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
                        'text_length': len(request.text) if request.text else 0,
                        'has_explainability': bool(explainability)
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
                'compound_score': result.get('compound_score', 0.0)
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
            "recent_predictions": recent
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

