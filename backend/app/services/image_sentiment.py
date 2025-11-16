"""
Image Sentiment Analysis
Analyzes images using color psychology, brightness, and ResNet50 features
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Tuple
from torchvision import models, transforms


class ImageSentimentAnalyzer:
    def __init__(self, device='cpu'):
        self.device = device
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = self.resnet.to(device)
        self.resnet.eval()
        
        # Remove final classification layer to get features
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_colors(self, image: Image.Image) -> Dict:
        """Analyze color distribution and psychology."""
        # Convert to RGB numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Calculate average color
        avg_color = np.mean(img_array, axis=(0, 1))
        r, g, b = avg_color
        
        # Calculate brightness
        brightness = np.mean(avg_color)
        
        # Calculate saturation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        # Color psychology scoring
        # Warm colors (red, orange, yellow) - energetic but can be negative if too intense
        warmth = (r - b) / 255.0
        
        # Cool colors (blue, green) - calming, positive
        coolness = (b - r) / 255.0
        
        # Brightness indicates positivity
        brightness_score = (brightness / 255.0 - 0.5) * 2  # -1 to 1
        
        # High saturation with warm colors can indicate intensity (positive or negative)
        # Low saturation often indicates neutrality or sadness
        saturation_factor = saturation / 255.0
        
        # Calculate color-based sentiment score
        color_score = 0.0
        
        # Bright + saturated + warm/cool balance
        if brightness > 150 and saturation > 80:
            # Bright and saturated - likely positive
            color_score += 0.5 + (brightness / 255.0) * 0.3
        elif brightness < 80:
            # Dark images - slightly negative
            color_score -= 0.3
        
        # Blue and green (calming) contribute positively
        if b > 120 or g > 120:
            color_score += 0.2
        
        # Very red can indicate anger or passion
        if r > 180 and saturation > 100:
            color_score -= 0.1  # Slightly negative (intense)
        
        return {
            'brightness': float(brightness),
            'saturation': float(saturation),
            'warmth': float(warmth),
            'coolness': float(coolness),
            'color_score': float(np.clip(color_score, -1, 1))
        }
    
    def extract_deep_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract deep features using ResNet50."""
        with torch.no_grad():
            features = self.feature_extractor(image_tensor.unsqueeze(0).to(self.device))
            features = features.squeeze()
        return features
    
    def analyze_sentiment(self, image: Image.Image) -> Dict:
        """
        Analyze sentiment from image.
        Returns sentiment label, confidence, and probabilities.
        """
        # Color analysis
        color_analysis = self.analyze_colors(image)
        
        # Transform image for deep learning
        image_tensor = self.transform(image)
        
        # Extract deep features
        features = self.extract_deep_features(image_tensor)
        
        # Feature-based heuristics
        # High-level features (last layers) often correlate with scene understanding
        feature_magnitude = torch.norm(features).item()
        feature_variance = torch.var(features).item()
        
        # Combine color psychology with feature analysis
        color_score = color_analysis['color_score']
        brightness_factor = (color_analysis['brightness'] / 255.0 - 0.5) * 0.5
        
        # Feature magnitude and variance can indicate complexity and richness
        # (rich, complex images often positive; simple, monotone often neutral/negative)
        complexity_score = np.log1p(feature_variance) * 0.1
        
        # Combined sentiment score
        combined_score = color_score + brightness_factor + complexity_score
        combined_score = np.clip(combined_score, -1.5, 1.5)
        
        # Convert to probabilities
        if combined_score <= -0.3:
            sentiment_label = 'negative'
            neg_prob = 0.5 + min(abs(combined_score) * 0.25, 0.45)
            pos_prob = max(0.05, 0.25 - abs(combined_score) * 0.15)
            neu_prob = 1.0 - neg_prob - pos_prob
        elif combined_score >= 0.3:
            sentiment_label = 'positive'
            pos_prob = 0.5 + min(combined_score * 0.25, 0.45)
            neg_prob = max(0.05, 0.25 - combined_score * 0.15)
            neu_prob = 1.0 - pos_prob - neg_prob
        else:
            sentiment_label = 'neutral'
            neu_prob = 0.45 + (0.3 - abs(combined_score)) * 0.4
            remaining = 1.0 - neu_prob
            if combined_score > 0:
                pos_prob = remaining * 0.65
                neg_prob = remaining * 0.35
            else:
                neg_prob = remaining * 0.65
                pos_prob = remaining * 0.35
        
        confidence = max(neg_prob, neu_prob, pos_prob)
        
        return {
            'sentiment': sentiment_label,
            'confidence': float(confidence),
            'probabilities': {
                'negative': float(neg_prob),
                'neutral': float(neu_prob),
                'positive': float(pos_prob)
            },
            'color_analysis': color_analysis,
            'combined_score': float(combined_score)
        }


# Global analyzer instance
_image_analyzer = None

def get_image_analyzer(device='cpu'):
    """Get or create image analyzer singleton."""
    global _image_analyzer
    if _image_analyzer is None:
        _image_analyzer = ImageSentimentAnalyzer(device)
    return _image_analyzer


def analyze_image_sentiment(image: Image.Image, device='cpu') -> Dict:
    """Analyze image sentiment."""
    analyzer = get_image_analyzer(device)
    return analyzer.analyze_sentiment(image)
