"""
Image Sentiment Analysis
Analyzes images using color psychology, brightness, and ResNet50 features
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Optional
from torchvision import models, transforms

try:
    from transformers import pipeline
except Exception as exc:
    pipeline = None
    _pipeline_import_error = exc
else:
    _pipeline_import_error = None


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
        self._clip_pipeline = None
        self._clip_available = False
        self._clip_candidates = [
            "positive emotional tone",
            "neutral emotional tone",
            "negative emotional tone",
        ]
        self._clip_label_map = {
            "positive emotional tone": "positive",
            "neutral emotional tone": "neutral",
            "negative emotional tone": "negative",
        }
        self._fer_pipeline = None
        self._fer_available = False
        self._fer_model_id = "nateraw/vit-base-patch16-224-in21k-finetuned-fer2013"
        if pipeline is None:
            if _pipeline_import_error:
                print(f"[WARN] Transformer import unavailable for image sentiment: {_pipeline_import_error}")
        else:
            try:
                self._clip_pipeline = pipeline(
                    "zero-shot-image-classification",
                    model="openai/clip-vit-base-patch32",
                )
                self._clip_available = True
                print("[OK] Loaded CLIP image sentiment model")
            except Exception as exc:
                print(f"[WARN] Failed to load CLIP image sentiment model ({exc}).")
            try:
                self._fer_pipeline = pipeline(
                    "image-classification",
                    model=self._fer_model_id,
                    top_k=7,
                )
                self._fer_available = True
                print(f"[OK] Loaded facial emotion model: {self._fer_model_id}")
            except Exception as exc:
                print(f"[WARN] Failed to load facial emotion model ({exc}).")
    
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

    def _analyze_with_clip(self, image: Image.Image) -> Optional[Dict]:
        """Run zero-shot classification to estimate sentiment."""
        if not self._clip_available or self._clip_pipeline is None:
            return None
        try:
            outputs = self._clip_pipeline(
                image,
                candidate_labels=self._clip_candidates,
                hypothesis_template="This image conveys {}.",
            )
        except Exception as exc:
            print(f"[WARN] CLIP image sentiment inference failed: {exc}")
            return None

        clip_probs = {label: 0.0 for label in ['positive', 'neutral', 'negative']}
        for item in outputs:
            mapped = self._clip_label_map.get(item['label'])
            if mapped:
                clip_probs[mapped] = float(item['score'])

        total = sum(clip_probs.values())
        if total <= 0:
            return None

        sentiment_label = max(clip_probs, key=clip_probs.get)
        confidence = clip_probs[sentiment_label]

        return {
            'probabilities': clip_probs,
            'sentiment': sentiment_label,
            'confidence': float(confidence),
            'raw_scores': outputs,
        }

    def _analyze_with_fer(self, image: Image.Image) -> Optional[Dict]:
        """Use a facial emotion recognition model to infer sentiment."""
        if not self._fer_available or self._fer_pipeline is None:
            return None
        try:
            outputs = self._fer_pipeline(image)
        except Exception as exc:
            print(f"[WARN] Facial emotion inference failed: {exc}")
            return None

        sentiment_map = {
            'angry': 'negative',
            'disgust': 'negative',
            'fear': 'negative',
            'sad': 'negative',
            'happy': 'positive',
            'surprise': 'positive',
            'neutral': 'neutral',
        }

        sentiment_probs = {label: 0.0 for label in ['negative', 'neutral', 'positive']}
        for item in outputs:
            label = item['label'].lower()
            score = float(item['score'])
            mapped = sentiment_map.get(label)
            if mapped:
                sentiment_probs[mapped] += score

        total = sum(sentiment_probs.values())
        if total <= 0:
            return None

        sentiment_probs = {label: value / total for label, value in sentiment_probs.items()}
        sentiment_label = max(sentiment_probs, key=sentiment_probs.get)
        confidence = sentiment_probs[sentiment_label]

        return {
            'probabilities': sentiment_probs,
            'sentiment': sentiment_label,
            'confidence': float(confidence),
            'raw_scores': outputs,
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
        
        # Convert to probabilities using heuristic model
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
        heuristic_result = {
            'sentiment': sentiment_label,
            'confidence': float(confidence),
            'probabilities': {
                'negative': float(neg_prob),
                'neutral': float(neu_prob),
                'positive': float(pos_prob),
            },
            'color_analysis': color_analysis,
            'combined_score': float(combined_score),
        }

        clip_result = self._analyze_with_clip(image)
        fer_result = self._analyze_with_fer(image)

        sources = [('heuristic', heuristic_result, 0.2)]

        if clip_result:
            sources.append(('clip', clip_result, 0.3))

        if fer_result:
            sources.append(('facial_emotion', fer_result, 0.5))

        if len(sources) == 1:
            return heuristic_result

        total_weight = sum(weight for _, _, weight in sources)
        combined_probs = {label: 0.0 for label in ['negative', 'neutral', 'positive']}
        combined_score = 0.0
        normalized_weights = {}

        for name, source, weight in sources:
            norm_weight = weight / total_weight if total_weight else 0.0
            normalized_weights[name] = norm_weight
            for label in combined_probs:
                combined_probs[label] += source['probabilities'].get(label, 0.0) * norm_weight
            score_delta = (
                source['probabilities'].get('positive', 0.0) -
                source['probabilities'].get('negative', 0.0)
            )
            combined_score += score_delta * norm_weight

        final_sentiment = max(combined_probs, key=combined_probs.get)
        final_confidence = combined_probs[final_sentiment]

        return {
            'sentiment': final_sentiment,
            'confidence': float(final_confidence),
            'probabilities': {label: float(value) for label, value in combined_probs.items()},
            'color_analysis': color_analysis,
            'combined_score': float(combined_score),
            'source_breakdown': {
                'clip': clip_result,
                'facial_emotion': fer_result,
                'heuristic': heuristic_result,
                'weights': normalized_weights,
            }
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
