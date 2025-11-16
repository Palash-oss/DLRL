import React, { useState, useEffect, useRef } from 'react';
import { RiSendPlaneLine } from 'react-icons/ri';
import './LiveFeed.css';

function LiveFeed() {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [inputImage, setInputImage] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const connectWebSocket = () => {
    // Connect directly to backend WebSocket
    const wsUrl = 'ws://localhost:8000/api/sentiment-stream';
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      setConnected(true);
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Attach the input that was sent
      if (ws.pendingInput) {
        data.inputText = ws.pendingInput.text;
        data.hasImage = ws.pendingInput.hasImage;
        ws.pendingInput = null;
      }
      
      setMessages((prev) => [data, ...prev].slice(0, 50)); // Keep last 50 messages
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };
    
    ws.onclose = () => {
      setConnected(false);
      console.log('WebSocket disconnected');
      // Reconnect after 3 seconds
      setTimeout(connectWebSocket, 3000);
    };
    
    wsRef.current = ws;
  };

  const handleSend = () => {
    if (!connected || !wsRef.current) return;

    const message = {
      text: inputText || null,
      image_base64: inputImage || null
    };

    if (inputText || inputImage) {
      // Store input for display
      const inputForDisplay = {
        text: inputText,
        hasImage: !!inputImage
      };
      
      wsRef.current.send(JSON.stringify(message));
      
      // Store to attach to next message
      wsRef.current.pendingInput = inputForDisplay;
      
      setInputText('');
      setInputImage(null);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        setInputImage(base64);
      };
      reader.readAsDataURL(file);
    }
  };

  const getSentimentClass = (sentiment) => {
    return `sentiment-${sentiment}`;
  };

  return (
    <div className="live-feed">
      <div className="card">
        <div className="feed-header">
          <h2>Live</h2>
          <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
            <span className="status-dot"></span>
            {connected ? 'Connected' : 'Disconnected'}
          </div>
        </div>

        <div className="feed-input">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to analyze in real-time..."
            rows={3}
          />
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            style={{ marginTop: 'var(--space-md)' }}
          />
          <button
            className="button"
            onClick={handleSend}
            disabled={!connected || (!inputText && !inputImage)}
            style={{ marginTop: 'var(--space-lg)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 'var(--space-sm)' }}
          >
            <RiSendPlaneLine size={16} />
            <span>Send</span>
          </button>
        </div>

        <div className="feed-messages">
          {messages.length === 0 ? (
            <div className="empty-feed">No messages yet</div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className="feed-message">
                <div className="message-header">
                  <span className={`sentiment-badge ${getSentimentClass(msg.sentiment)}`}>
                    {msg.sentiment.toUpperCase()}
                  </span>
                  <span className="message-time">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                </div>
                {msg.inputText && (
                  <div className="message-input">
                    "{msg.inputText}"
                    {msg.hasImage && <span style={{ marginLeft: '8px', color: 'var(--text-tertiary)', fontSize: 'var(--text-xs)' }}>+ Image</span>}
                  </div>
                )}
                <div className="message-confidence">
                  {(msg.confidence * 100).toFixed(0)}% confident
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default LiveFeed;

