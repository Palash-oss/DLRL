"""
Explainability services: Attention visualization and Grad-CAM
"""

import torch
import numpy as np
import os
import sys
from typing import Dict, List, Optional
from PIL import Image
import io
import base64
import cv2

# Add parent directory to path to import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from models.image_encoder import GradCAM


def get_text_attention(text: str, attention_weights: np.ndarray) -> Dict:
    """
    Get word-level attention weights for text.
    
    Returns:
        Dictionary with tokens and their attention scores
    """
    tokens = text.split()
    
    # Align attention weights with tokens
    # Attention weights might be shorter due to padding
    num_tokens = len(tokens)
    num_weights = len(attention_weights)
    
    if num_weights > num_tokens:
        attention_weights = attention_weights[:num_tokens]
    elif num_weights < num_tokens:
        # Pad with zeros
        attention_weights = np.pad(
            attention_weights,
            (0, num_tokens - num_weights),
            mode='constant'
        )
    
    # Normalize attention weights
    attention_weights = attention_weights / (attention_weights.sum() + 1e-8)
    
    return {
        'tokens': tokens,
        'attention_scores': attention_weights.tolist(),
        'highlighted_tokens': [
            {
                'token': token,
                'score': float(score)
            }
            for token, score in zip(tokens, attention_weights)
        ]
    }


def get_image_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_idx: int,
    target_layer: Optional[torch.nn.Module] = None
) -> Optional[str]:
    """
    Generate Grad-CAM heatmap for image.
    
    Returns:
        Base64-encoded heatmap image
    """
    try:
        # Get target layer (last conv layer of ResNet)
        if target_layer is None:
            target_layer = model.image_encoder.backbone[-1]
        
        # Create Grad-CAM
        gradcam = GradCAM(model.image_encoder, target_layer)
        
        # Generate CAM
        cam = gradcam.generate_cam(image_tensor, class_idx)
        
        # Convert to numpy
        cam_np = cam.cpu().numpy()
        
        # Resize to original image size
        original_image = image_tensor[0].cpu().permute(1, 2, 0).numpy()
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        
        # Resize CAM
        cam_resized = cv2.resize(cam_np, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            cv2.COLORMAP_JET
        )
        heatmap = np.float32(heatmap) / 255
        
        # Overlay on original image
        overlayed = heatmap * 0.4 + original_image * 0.6
        overlayed = np.uint8(255 * overlayed)
        
        # Convert to base64
        pil_image = Image.fromarray(overlayed)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_str
    
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None


def visualize_attention(text: str, attention_weights: np.ndarray) -> str:
    """
    Create HTML visualization of attention weights.
    """
    tokens = text.split()
    attention_weights = attention_weights[:len(tokens)]
    attention_weights = attention_weights / (attention_weights.sum() + 1e-8)
    
    html = "<div style='font-family: monospace;'>"
    for token, weight in zip(tokens, attention_weights):
        # Color intensity based on attention weight
        intensity = int(weight * 255)
        color = f"rgb({intensity}, {255 - intensity}, 0)"
        html += f"<span style='background-color: {color}; padding: 2px;'>{token}</span> "
    html += "</div>"
    
    return html

