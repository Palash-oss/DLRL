import React, { useState, useEffect } from 'react';
import apiClient from '../apiClient';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import './Dashboard.css';

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [trendData, setTrendData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    // Simulate real-time updates
    const interval = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await apiClient.get('/api/dashboard');
      const data = response.data;
      setStats(data);
      
      // Use real daily stats from backend
      if (data.daily_stats && data.daily_stats.length > 0) {
        const trend = data.daily_stats.reverse().map(day => ({
          time: new Date(day.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          positive: day.positive || 0,
          neutral: day.neutral || 0,
          negative: day.negative || 0,
          total: day.total || 0
        }));
        setTrendData(trend);
      } else {
        // No data yet - show empty trend
        setTrendData([]);
      }
      setLoading(false);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="dashboard">
        <div className="card">
          <div className="loading">Loading dashboard...</div>
        </div>
      </div>
    );
  }

  const distributionData = stats
    ? [
        { name: 'Positive', value: stats.sentiment_distribution.positive, color: '#28a745' },
        { name: 'Neutral', value: stats.sentiment_distribution.neutral, color: '#ffc107' },
        { name: 'Negative', value: stats.sentiment_distribution.negative, color: '#dc3545' }
      ]
    : [];

  const learningCurveData = stats?.feedback_curve
    ? stats.feedback_curve.map((entry) => ({
        episode: entry.episode || 0,
        total_reward: entry.total_reward || 0,
      }))
    : [];

  return (
    <div className="dashboard">
      <div className="card">
        <h2>Overview</h2>
        
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{stats?.total_predictions || 0}</div>
            <div className="stat-label">Total</div>
          </div>
          <div className="stat-card positive">
            <div className="stat-value">{stats?.sentiment_distribution?.positive || 0}</div>
            <div className="stat-label">Positive</div>
          </div>
          <div className="stat-card neutral">
            <div className="stat-value">{stats?.sentiment_distribution?.neutral || 0}</div>
            <div className="stat-label">Neutral</div>
          </div>
          <div className="stat-card negative">
            <div className="stat-value">{stats?.sentiment_distribution?.negative || 0}</div>
            <div className="stat-label">Negative</div>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Distribution</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={distributionData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
            <XAxis dataKey="name" stroke="#ffffff" style={{ fontSize: '13px' }} />
            <YAxis stroke="#ffffff" style={{ fontSize: '13px' }} />
            <Tooltip 
              contentStyle={{ 
                background: '#0a0a0a', 
                border: '1px solid #3b82f6',
                borderRadius: '8px',
                color: '#ffffff',
                boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)'
              }}
              itemStyle={{ color: '#ffffff' }}
              labelStyle={{ color: '#ffffff' }}
            />
            <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>Trends</h2>
        {trendData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
              <XAxis dataKey="time" stroke="#ffffff" style={{ fontSize: '13px' }} />
              <YAxis stroke="#ffffff" style={{ fontSize: '13px' }} />
              <Tooltip 
                contentStyle={{ 
                  background: '#0a0a0a', 
                  border: '1px solid #3b82f6',
                  borderRadius: '8px',
                  color: '#ffffff',
                  boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)'
                }}
                itemStyle={{ color: '#ffffff' }}
                labelStyle={{ color: '#ffffff' }}
              />
              <Legend 
                wrapperStyle={{ fontSize: '13px', color: '#a0a0a0' }}
              />
              <Line type="monotone" dataKey="positive" stroke="#10b981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="neutral" stroke="#f59e0b" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="negative" stroke="#ef4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div style={{textAlign: 'center', padding: 'var(--space-3xl)', color: 'var(--text-tertiary)', fontSize: 'var(--text-sm)'}}>
            No trend data yet
          </div>
        )}
      </div>

      {learningCurveData.length > 0 && (
        <div className="card">
          <h2>Learning Curve</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={learningCurveData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
              <XAxis
                dataKey="episode"
                stroke="#ffffff"
                style={{ fontSize: '13px' }}
                tickFormatter={(value) => `#${value}`}
              />
              <YAxis stroke="#ffffff" style={{ fontSize: '13px' }} />
              <Tooltip
                contentStyle={{
                  background: '#0a0a0a',
                  border: '1px solid #ec4899',
                  borderRadius: '8px',
                  color: '#ffffff',
                }}
                itemStyle={{ color: '#ffffff' }}
                labelStyle={{ color: '#ffffff' }}
              />
              <Line type="monotone" dataKey="total_reward" stroke="#ec4899" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {stats?.recent_predictions && stats.recent_predictions.length > 0 && (
        <div className="card">
          <h2>Recent</h2>
          <div className="recent-predictions">
            {stats.recent_predictions.slice(0, 10).map((pred) => (
              <div key={pred.id} className="prediction-item">
                <div className="prediction-header">
                  <span className={`sentiment-badge sentiment-${pred.sentiment}`}>
                    {pred.sentiment}
                  </span>
                  <span className="prediction-time">
                    {new Date(pred.timestamp).toLocaleString()}
                  </span>
                </div>
                {pred.text && (
                  <div className="prediction-text">
                    {pred.text.substring(0, 100)}{pred.text.length > 100 ? '...' : ''}
                  </div>
                )}
                <div className="prediction-confidence">
                  Confidence: {(pred.confidence * 100).toFixed(1)}% | 
                  Modality: {pred.modality || 'text'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;

