import React from 'react';
import './SentimentGauge.css';

function SentimentGauge({ probabilities }) {
  if (!probabilities) return null;

  const sentimentColors = {
    positive: '#28a745',
    neutral: '#ffc107',
    negative: '#dc3545'
  };

  const maxSentiment = Object.entries(probabilities).reduce(
    (a, b) => (probabilities[a[0]] > probabilities[b[0]] ? a : b)
  );

  // Calculate angle for gauge (0 to 180 degrees)
  const positiveProb = probabilities.positive || 0;
  const negativeProb = probabilities.negative || 0;
  const angle = 90 + (positiveProb - negativeProb) * 90; // -90 to +90, centered at 0

  return (
    <div className="sentiment-gauge">
      <h3>Sentiment Gauge</h3>
      <div className="gauge-container">
        <svg viewBox="0 0 200 120" className="gauge-svg">
          {/* Gauge arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#e0e0e0"
            strokeWidth="20"
            strokeLinecap="round"
          />
          {/* Positive section */}
          <path
            d="M 20 100 A 80 80 0 0 1 100 20"
            fill="none"
            stroke={sentimentColors.positive}
            strokeWidth="20"
            strokeLinecap="round"
            opacity={positiveProb}
          />
          {/* Neutral section */}
          <path
            d="M 100 20 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={sentimentColors.negative}
            strokeWidth="20"
            strokeLinecap="round"
            opacity={negativeProb}
          />
          {/* Needle */}
          <line
            x1="100"
            y1="100"
            x2={100 + 70 * Math.cos((angle - 90) * (Math.PI / 180))}
            y2={100 - 70 * Math.sin((angle - 90) * (Math.PI / 180))}
            stroke="#333"
            strokeWidth="3"
            strokeLinecap="round"
          />
          {/* Center dot */}
          <circle cx="100" cy="100" r="5" fill="#333" />
        </svg>
        <div className="gauge-labels">
          <span className="gauge-label negative">Negative</span>
          <span className="gauge-label neutral">Neutral</span>
          <span className="gauge-label positive">Positive</span>
        </div>
      </div>
      <div className="gauge-value">
        <span className="value-label">Current Sentiment:</span>
        <span
          className="value-text"
          style={{ color: sentimentColors[maxSentiment[0]] }}
        >
          {maxSentiment[0].toUpperCase()}
        </span>
      </div>
    </div>
  );
}

export default SentimentGauge;

