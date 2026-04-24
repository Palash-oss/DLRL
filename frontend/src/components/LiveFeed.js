import React, { useState, useEffect, useRef } from 'react';
import { RiSendPlaneLine } from 'react-icons/ri';
import './LiveFeed.css';

function LiveFeed() {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [inputImage, setInputImage] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const isConnectingRef = useRef(false);
  const isMountedRef = useRef(true);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;

  useEffect(() => {
    isMountedRef.current = true;
    connectWebSocket();
    
    return () => {
      isMountedRef.current = false;
      
      // Clear reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      
      // Close WebSocket cleanly
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        wsRef.current.onmessage = null;
        wsRef.current.onopen = null;
        
        if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
          wsRef.current.close();
        }
        wsRef.current = null;
      }
      
      isConnectingRef.current = false;
    };
  }, []);

  const connectWebSocket = () => {
    // Don't connect if unmounted or already connecting/connected
    if (!isMountedRef.current || 
        isConnectingRef.current || 
        (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING))) {
      return;
    }

    // Stop if max reconnect attempts reached
    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      setConnected(false);
      return;
    }

    isConnectingRef.current = true;
    const wsUrl = 'ws://localhost:8000/api/sentiment-stream';
    
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        if (!isMountedRef.current) {
          ws.close();
          return;
        }
        
        setConnected(true);
        isConnectingRef.current = false;
        reconnectAttemptsRef.current = 0;
        console.log('WebSocket connected successfully');
      };
      
      ws.onmessage = (event) => {
        if (!isMountedRef.current) return;
        
        try {
          const data = JSON.parse(event.data);
          
          // Only add if we got sentiment analysis back
          if (data.sentiment && data.confidence && data.probabilities) {
            data.id = `${Date.now()}-${Math.random()}`;
            setMessages((prev) => {
              // Check if we already have this exact message (prevent duplicates)
              const isDuplicate = prev.some(msg => 
                msg.timestamp === data.timestamp && 
                msg.sentiment === data.sentiment &&
                msg.inputText === data.inputText
              );
              
              if (isDuplicate) return prev;
              return [data, ...prev].slice(0, 50);
            });
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        isConnectingRef.current = false;
        
        if (isMountedRef.current) {
          setConnected(false);
        }
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        isConnectingRef.current = false;
        
        if (!isMountedRef.current) return;
        
        setConnected(false);
        
        // Clear old reference
        if (wsRef.current === ws) {
          wsRef.current = null;
        }
        
        // Only reconnect if component is still mounted and connection was established before
        if (isMountedRef.current && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000); // Exponential backoff
          
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (isMountedRef.current) {
              connectWebSocket();
            }
          }, delay);
        }
      };
      
    } catch (err) {
      console.error('Error creating WebSocket:', err);
      isConnectingRef.current = false;
      setConnected(false);
      
      // Retry with backoff
      if (isMountedRef.current && reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current += 1;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000);
        
        reconnectTimeoutRef.current = setTimeout(() => {
          if (isMountedRef.current) {
            connectWebSocket();
          }
        }, delay);
      }
    }
  };

  const handleSend = () => {
    if (!connected || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected');
      return;
    }

    if (!inputText && !inputImage) {
      return;
    }

    const message = {
      text: inputText || null,
      image_base64: inputImage || null
    };

    try {
      wsRef.current.send(JSON.stringify(message));
      
      // Clear inputs
      setInputText('');
      setInputImage(null);
      
      // Clear file input
      const fileInput = document.getElementById('live-feed-image');
      if (fileInput) {
        fileInput.value = '';
      }
    } catch (err) {
      console.error('Error sending message:', err);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Check file size (max 5MB)
      if (file.size > 5 * 1024 * 1024) {
        alert('Image too large. Maximum size is 5MB.');
        e.target.value = '';
        return;
      }
      
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        setInputImage(base64);
      };
      reader.onerror = () => {
        alert('Error reading image file');
        e.target.value = '';
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
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Enter text to analyze in real-time..."
            rows={3}
          />
          <div style={{ marginTop: 'var(--space-md)', display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              id="live-feed-image"
              style={{ display: 'none' }}
            />
            <label htmlFor="live-feed-image" className="button" style={{ cursor: 'pointer', display: 'inline-block' }}>
              {inputImage ? '✓ Image Selected' : 'Choose Image'}
            </label>
            {inputImage && (
              <button
                className="button"
                onClick={() => setInputImage(null)}
                style={{ background: '#dc3545' }}
              >
                Remove Image
              </button>
            )}
          </div>
          <button
            className="button"
            onClick={handleSend}
            disabled={!connected || (!inputText && !inputImage)}
            style={{ 
              marginTop: 'var(--space-lg)', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              gap: 'var(--space-sm)',
              width: '100%'
            }}
          >
            <RiSendPlaneLine size={16} />
            <span>{connected ? 'Send' : 'Connecting...'}</span>
          </button>
        </div>

        <div className="feed-messages">
          {messages.length === 0 ? (
            <div className="empty-feed">No posts yet. Share your first thought!</div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id || msg.timestamp} className="feed-message">
                {/* Post Header */}
                <div className="post-header">
                  <div className="post-info">
                    <div className="post-avatar">📱</div>
                    <div>
                      <div className="post-author">Your Post</div>
                      <div className="post-time">{new Date(msg.timestamp).toLocaleString()}</div>
                    </div>
                  </div>
                </div>

                {/* Post Content */}
                <div className="post-content">
                  {msg.inputText && (
                    <div className="post-text">{msg.inputText}</div>
                  )}
                  
                  {msg.inputImage && (
                    <div className="post-image">
                      <img 
                        src={`data:image/jpeg;base64,${msg.inputImage}`} 
                        alt="Posted content"
                      />
                    </div>
                  )}
                </div>

                {/* Sentiment Analysis Result */}
                <div className="post-sentiment">
                  <div className="sentiment-result-header">
                    <span className="sentiment-label">Sentiment Analysis</span>
                    <span className={`sentiment-badge ${getSentimentClass(msg.sentiment)}`}>
                      {msg.sentiment.toUpperCase()}
                    </span>
                  </div>
                  <div className="sentiment-confidence">
                    <div className="confidence-bar-container">
                      <div 
                        className="confidence-bar-fill" 
                        style={{ width: `${msg.confidence * 100}%` }}
                      >
                        <span className="confidence-text">{(msg.confidence * 100).toFixed(0)}% Confident</span>
                      </div>
                    </div>
                  </div>
                  {msg.probabilities && (
                    <div className="sentiment-breakdown">
                      <div className="prob-item">
                        <span className="prob-label">😊 Positive</span>
                        <span className="prob-value">{(msg.probabilities.positive * 100).toFixed(0)}%</span>
                      </div>
                      <div className="prob-item">
                        <span className="prob-label">😐 Neutral</span>
                        <span className="prob-value">{(msg.probabilities.neutral * 100).toFixed(0)}%</span>
                      </div>
                      <div className="prob-item">
                        <span className="prob-label">😔 Negative</span>
                        <span className="prob-value">{(msg.probabilities.negative * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  )}
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

