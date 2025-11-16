import React, { useState } from 'react';
import axios from 'axios';
import { RiSearchLine } from 'react-icons/ri';
import './SentimentAnalyzer.css';
import WordHighlight from './WordHighlight';
import ImageHeatmap from './ImageHeatmap';
import SentimentGauge from './SentimentGauge';

function SentimentAnalyzer() {
  const [text, setText] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const convertImageToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = (error) => reject(error);
    });
  };

  const handleAnalyze = async () => {
    if (!text && !image) {
      setError('Please provide at least text or image');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let imageBase64 = null;
      if (image) {
        imageBase64 = await convertImageToBase64(image);
      }

      const response = await axios.post('/api/analyze', {
        text: text || null,
        image_base64: imageBase64
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentClass = (sentiment) => {
    return `sentiment-${sentiment}`;
  };

  return (
    <div className="sentiment-analyzer">
      <div className="card">
        <h2>Analyzer</h2>

        <div className="input-group">
          <label htmlFor="text-input">Text</label>
          <textarea
            id="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze..."
          />
        </div>

        <div className="input-group">
          <label htmlFor="image-input">Image</label>
          <input
            id="image-input"
            type="file"
            accept="image/*"
            onChange={handleImageChange}
          />
          {imagePreview && (
            <div className="image-preview">
              <img src={imagePreview} alt="Preview" />
            </div>
          )}
        </div>

        <button
          className="button"
          onClick={handleAnalyze}
          disabled={loading || (!text && !image)}
        >
          <RiSearchLine size={16} style={{ marginRight: '8px' }} />
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="sentiment-result">
            <div className="result-header">
              <span className={`sentiment-badge ${getSentimentClass(result.sentiment)}`}>
                {result.sentiment}
              </span>
              <div className="confidence-info">
                <div className="confidence-label">Confidence</div>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  >
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            <SentimentGauge probabilities={result.probabilities} />

            <div className="probability-list">
              <h3>Probabilities</h3>
              {Object.entries(result.probabilities).map(([sentiment, prob]) => (
                <div key={sentiment} className="probability-item">
                  <span>{sentiment}</span>
                  <span>{(prob * 100).toFixed(2)}%</span>
                </div>
              ))}
            </div>

            {result.explainability && (
              <div className="explainability-section">
                <h3>Explainability</h3>
                
                {result.explainability.text_attention && (
                  <div className="explainability-item">
                    <h4>Text Attention</h4>
                    <WordHighlight
                      text={text}
                      attention={result.explainability.text_attention}
                    />
                  </div>
                )}

                {result.explainability.image_gradcam && (
                  <div className="explainability-item">
                    <h4>Image Heatmap (Grad-CAM)</h4>
                    <ImageHeatmap
                      originalImage={imagePreview}
                      heatmap={result.explainability.image_gradcam}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default SentimentAnalyzer;

