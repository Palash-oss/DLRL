"""
Model loading and inference utilities
"""

import torch
import json
import os
import sys
from typing import Tuple, Optional

# Add parent directory to path to import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from models.multimodal_model import MultimodalSentimentModel


def load_model(
    model_path: str,
    vocab_path: str,
    device: torch.device,
    embedding_dim: int = 100,
    text_hidden_dim: int = 256,
    image_output_dim: int = 2048,
    latent_dim: int = 512,
    num_classes: int = 3
) -> Tuple[MultimodalSentimentModel, dict]:
    """
    Load trained model and vocabulary.
    
    Returns:
        model, vocab
    """
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    vocab_size = vocab['vocab_size']
    
    # Create model
    model = MultimodalSentimentModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        text_hidden_dim=text_hidden_dim,
        image_output_dim=image_output_dim,
        latent_dim=latent_dim,
        num_classes=num_classes
    )
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"⚠ Model checkpoint not found at {model_path}")
        print("  Using randomly initialized model (for demo purposes)")
        print("  For production, train the model using: python training/train.py")
    
    model.to(device)
    model.eval()
    
    return model, vocab


def predict_sentiment(
    model: MultimodalSentimentModel,
    text_ids: Optional[torch.Tensor] = None,
    text_lengths: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    device: torch.device = None
) -> dict:
    """
    Predict sentiment from model inputs.
    
    Returns:
        Dictionary with predictions, probabilities, and features
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        result = model.predict_sentiment(
            text_ids=text_ids,
            text_lengths=text_lengths,
            images=images
        )
    
    return result

