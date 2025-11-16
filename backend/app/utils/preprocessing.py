"""
Preprocessing utilities for text and images
"""

import torch
import numpy as np
from PIL import Image
import io
import base64
import re
from typing import Tuple, Optional, Dict
from torchvision import transforms


def preprocess_text(text: str, vocab: Dict, max_seq_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess text and convert to token IDs.
    
    Returns:
        text_ids: Token IDs tensor [max_seq_len]
        text_length: Actual sequence length
    """
    word_to_idx = vocab['word_to_idx']
    
    # Preprocess
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Convert to indices
    ids = [word_to_idx.get(token, word_to_idx.get('<UNK>', 1)) for token in tokens]
    
    # Truncate or pad
    text_length = min(len(ids), max_seq_len)
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
    else:
        ids = ids + [word_to_idx.get('<PAD>', 0)] * (max_seq_len - len(ids))
    
    return torch.tensor(ids, dtype=torch.long), torch.tensor(text_length, dtype=torch.long)


def preprocess_image(image_base64: str, image_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Preprocess image from base64 string.
    
    Returns:
        image_tensor: Preprocessed image tensor [3, H, W]
    """
    # Decode base64
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    return image_tensor


def preprocess_image_file(image_file) -> torch.Tensor:
    """
    Preprocess image from file upload.
    """
    image = Image.open(image_file).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)

