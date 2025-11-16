import React from 'react';
import './WordHighlight.css';

function WordHighlight({ text, attention }) {
  if (!attention || !attention.highlighted_tokens) {
    return <div className="word-highlight">No attention data available</div>;
  }

  const tokens = attention.tokens || text.split(' ');
  const scores = attention.attention_scores || [];

  // Normalize scores for visualization
  const maxScore = Math.max(...scores, 0.01);
  const normalizedScores = scores.map(s => s / maxScore);

  return (
    <div className="word-highlight">
      <div className="word-container">
        {tokens.map((token, idx) => {
          const score = normalizedScores[idx] || 0;
          // Color intensity based on attention score
          const backgroundColor = `rgba(102, 126, 234, ${score * 0.6})`;
          const textColor = score > 0.5 ? 'white' : '#333';

          return (
            <span
              key={idx}
              className="word-token"
              style={{
                backgroundColor,
                color: textColor,
                fontWeight: score > 0.7 ? 'bold' : 'normal'
              }}
              title={`Attention: ${(score * 100).toFixed(1)}%`}
            >
              {token}
            </span>
          );
        })}
      </div>
      <div className="legend">
        <span className="legend-item">
          <span className="legend-color low"></span>
          Low Attention
        </span>
        <span className="legend-item">
          <span className="legend-color high"></span>
          High Attention
        </span>
      </div>
    </div>
  );
}

export default WordHighlight;

