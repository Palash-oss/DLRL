"""
Real Sentiment Analysis Engine
Uses lexicon-based approach with intensity boosters and negation handling
"""

import re
from typing import Dict, List, Tuple
import numpy as np

# Sentiment lexicon (word -> score)
SENTIMENT_LEXICON = {
    # Very Positive (2.0 to 3.0)
    'amazing': 2.5, 'awesome': 2.5, 'excellent': 2.5, 'fantastic': 2.5, 'wonderful': 2.5,
    'outstanding': 2.8, 'spectacular': 2.8, 'brilliant': 2.7, 'extraordinary': 2.8,
    'love': 2.3, 'adore': 2.5, 'perfect': 2.6, 'best': 2.4, 'beautiful': 2.3,
    'great': 2.0, 'superb': 2.6, 'magnificent': 2.7, 'phenomenal': 2.8,
    
    # Positive (0.5 to 2.0)
    'good': 1.5, 'nice': 1.3, 'happy': 1.8, 'joy': 1.9, 'pleased': 1.6,
    'delighted': 2.0, 'satisfied': 1.5, 'glad': 1.7, 'enjoy': 1.7, 'fun': 1.6,
    'like': 1.3, 'better': 1.2, 'pretty': 1.4, 'cool': 1.5, 'fine': 1.2,
    'positive': 1.5, 'fortunate': 1.7, 'correct': 1.2, 'superior': 1.8,
    'smile': 1.6, 'laugh': 1.7, 'hope': 1.4, 'optimistic': 1.8,
    
    # Slightly Positive (0.0 to 0.5)
    'okay': 0.3, 'ok': 0.3, 'alright': 0.4, 'decent': 0.5, 'fair': 0.4,
    'acceptable': 0.4, 'adequate': 0.3, 'moderate': 0.2,
    
    # Very Negative (-3.0 to -2.0)
    'terrible': -2.5, 'horrible': -2.6, 'awful': -2.5, 'disgusting': -2.7,
    'hate': -2.3, 'worst': -2.6, 'pathetic': -2.4, 'miserable': -2.5,
    'disaster': -2.6, 'nightmare': -2.7, 'dreadful': -2.5, 'abysmal': -2.8,
    'atrocious': -2.7, 'appalling': -2.6, 'horrific': -2.8,
    
    # Negative (-2.0 to -0.5)
    'bad': -1.5, 'poor': -1.4, 'sad': -1.6, 'unhappy': -1.7, 'angry': -1.8,
    'disappointed': -1.6, 'upset': -1.7, 'annoyed': -1.5, 'frustrated': -1.7,
    'depressed': -2.0, 'miserable': -2.1, 'worried': -1.4, 'concerned': -1.2,
    'wrong': -1.3, 'fail': -1.6, 'failed': -1.7, 'failure': -1.8,
    'lost': -1.4, 'boring': -1.3, 'dull': -1.2, 'weak': -1.3,
    'negative': -1.5, 'unfortunate': -1.6, 'inferior': -1.7,
    
    # Slightly Negative (-0.5 to 0.0)
    'meh': -0.2, 'whatever': -0.3, 'not good': -0.5, 'mediocre': -0.4,
    'lacking': -0.4, 'insufficient': -0.5, 'minor': -0.3,
}

# Intensity boosters
BOOSTERS = {
    'very': 0.4, 'really': 0.3, 'extremely': 0.5, 'absolutely': 0.5,
    'incredibly': 0.5, 'amazingly': 0.4, 'totally': 0.4, 'completely': 0.4,
    'so': 0.3, 'too': 0.3, 'quite': 0.2, 'highly': 0.3, 'utterly': 0.5,
    'exceptionally': 0.5, 'remarkably': 0.4, 'particularly': 0.3,
}

# Negations
NEGATIONS = {
    'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere',
    'hardly', 'barely', 'scarcely', 'rarely', "don't", "doesn't", "didn't",
    "won't", "wouldn't", "can't", "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't"
}

# Punctuation emphasis
EXCLAMATION_BOOST = 0.3
QUESTION_DAMPENER = -0.2


def preprocess_text_for_sentiment(text: str) -> List[str]:
    """Tokenize and preprocess text."""
    text = text.lower()
    # Keep punctuation for emphasis detection
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    tokens = re.findall(r"[\w']+|[.,!?;]", text)
    return tokens


def analyze_sentiment_lexicon(text: str) -> Dict:
    """
    Analyze sentiment using lexicon-based approach.
    Returns sentiment, confidence, and probabilities.
    """
    if not text or len(text.strip()) == 0:
        return {
            'sentiment': 'neutral',
            'confidence': 0.33,
            'compound_score': 0.0,
            'probabilities': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
        }
    
    tokens = preprocess_text_for_sentiment(text)
    
    sentiments = []
    attention_weights = []
    
    for i, token in enumerate(tokens):
        if token in ['.', ',', ';']:
            attention_weights.append(0.0)
            continue
            
        base_score = SENTIMENT_LEXICON.get(token, 0.0)
        
        # Check for negation in previous 3 words
        negation_found = False
        negation_index = -1
        for j in range(max(0, i-3), i):
            if tokens[j] in NEGATIONS:
                negation_found = True
                negation_index = j
                break
        
        if base_score != 0.0:
            # Word has sentiment
            if negation_found:
                base_score *= -0.7  # Flip and dampen
            
            # Check for booster in previous word
            if i > 0 and tokens[i-1] in BOOSTERS:
                boost = BOOSTERS[tokens[i-1]]
                if base_score > 0:
                    base_score += boost
                else:
                    base_score -= boost
            
            sentiments.append(base_score)
            attention_weights.append(abs(base_score) * 1.5)  # High attention on sentiment word
            
            # Also give attention to the negation word if found
            if negation_found and negation_index >= 0:
                attention_weights[negation_index] = abs(base_score) * 1.2  # High attention on negation
        elif token in NEGATIONS:
            # Negation word without immediate sentiment - give it attention
            attention_weights.append(0.8)
        elif token in BOOSTERS:
            # Booster word - give it attention
            attention_weights.append(0.6)
        else:
            attention_weights.append(0.1)  # Low attention for neutral words
    
    if not sentiments:
        # No sentiment words found - analyze emotionally neutral
        return {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'compound_score': 0.0,
            'probabilities': {'negative': 0.25, 'neutral': 0.50, 'positive': 0.25},
            'attention_weights': [0.1] * len(tokens),  # Low uniform attention
            'tokens': tokens
        }
    
    # Calculate compound score
    compound_score = np.sum(sentiments)
    
    # Normalize by text length (longer texts get dampened)
    compound_score = compound_score / np.sqrt(len(tokens) + 1)
    
    # Apply punctuation emphasis
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    if exclamation_count > 0:
        compound_score += EXCLAMATION_BOOST * min(exclamation_count, 3)
    if question_count > 0:
        compound_score += QUESTION_DAMPENER * min(question_count, 2)
    
    # Clamp compound score
    compound_score = np.clip(compound_score, -3.0, 3.0)
    
    # Convert to probabilities using softmax-like transformation
    # Thresholds: < -0.5: negative, > 0.5: positive, else: neutral
    if compound_score <= -0.5:
        sentiment_label = 'negative'
        # More negative = higher negative probability
        neg_prob = 0.5 + min(abs(compound_score) * 0.15, 0.45)
        pos_prob = max(0.05, 0.3 - abs(compound_score) * 0.1)
        neu_prob = 1.0 - neg_prob - pos_prob
    elif compound_score >= 0.5:
        sentiment_label = 'positive'
        pos_prob = 0.5 + min(compound_score * 0.15, 0.45)
        neg_prob = max(0.05, 0.3 - compound_score * 0.1)
        neu_prob = 1.0 - pos_prob - neg_prob
    else:
        sentiment_label = 'neutral'
        neu_prob = 0.5 + (0.5 - abs(compound_score)) * 0.3
        remaining = 1.0 - neu_prob
        if compound_score > 0:
            pos_prob = remaining * 0.6
            neg_prob = remaining * 0.4
        else:
            neg_prob = remaining * 0.6
            pos_prob = remaining * 0.4
    
    confidence = max(neg_prob, neu_prob, pos_prob)
    
    # Normalize attention weights
    if max(attention_weights) > 0:
        attention_weights = [w / max(attention_weights) for w in attention_weights]
    
    return {
        'sentiment': sentiment_label,
        'confidence': float(confidence),
        'compound_score': float(compound_score),
        'probabilities': {
            'negative': float(neg_prob),
            'neutral': float(neu_prob),
            'positive': float(pos_prob)
        },
        'attention_weights': attention_weights,
        'tokens': tokens
    }


def test_sentiment_analyzer():
    """Test the sentiment analyzer."""
    test_cases = [
        "I love this amazing product! It's absolutely fantastic!",
        "This is terrible. I hate it so much.",
        "It's okay, nothing special.",
        "Not bad, but could be better.",
        "I'm feeling sad and disappointed today.",
        "What an incredible experience! Best day ever!",
        "This is the worst thing I've ever seen.",
        "Pretty good, I'm satisfied.",
    ]
    
    for text in test_cases:
        result = analyze_sentiment_lexicon(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
        print(f"Scores: {result['probabilities']}")


if __name__ == "__main__":
    test_sentiment_analyzer()
