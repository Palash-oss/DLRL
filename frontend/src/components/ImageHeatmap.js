import React from 'react';
import './ImageHeatmap.css';

function ImageHeatmap({ originalImage, heatmap }) {
  if (!heatmap) {
    return <div className="image-heatmap">No heatmap data available</div>;
  }

  return (
    <div className="image-heatmap">
      <div className="heatmap-container">
        <div className="heatmap-image">
          <img
            src={`data:image/png;base64,${heatmap}`}
            alt="Grad-CAM Heatmap"
          />
        </div>
        {originalImage && (
          <div className="original-image">
            <img src={originalImage} alt="Original" />
            <div className="image-label">Original</div>
          </div>
        )}
      </div>
      <div className="heatmap-info">
        <p>
          <strong>Grad-CAM Visualization:</strong> Red regions indicate areas
          that strongly influence the sentiment prediction.
        </p>
      </div>
    </div>
  );
}

export default ImageHeatmap;

