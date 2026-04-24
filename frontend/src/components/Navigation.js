import React, { useState, useEffect } from 'react';
import './Navigation.css';
import { auth } from '../firebase';
import apiClient from '../apiClient';

function Navigation({ activeTab, setActiveTab }) {
  const [credits, setCredits] = useState('...');

  useEffect(() => {
    const fetchProfile = async () => {
      const user = auth.currentUser;
      if (user) {
        try {
          const res = await apiClient.get('/api/users/profile', {
            headers: { 'X-User-Id': user.uid }
          });
          if (res.data && res.data.user) {
            setCredits(res.data.user.credits);
          }
        } catch (e) {
          console.error("Failed to fetch credits");
        }
      }
    };
    fetchProfile();
    
    // Poll every 10 seconds just in case they used credits
    const interval = setInterval(fetchProfile, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <nav className="navigation">
      <div className="nav-container" style={{ display: 'flex', width: '100%', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="nav-brand">
          <h1>ACRUX</h1>
        </div>
        <div className="nav-tabs">
          <button
            className={`nav-tab ${activeTab === 'analyzer' ? 'active' : ''}`}
            onClick={() => setActiveTab('analyzer')}
          >
            <div className="nav-text-container">
              <span className="nav-text default">ANALYZER</span>
              <span className="nav-text hover">ANALYZER</span>
            </div>
          </button>
          <button
            className={`nav-tab ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <div className="nav-text-container">
              <span className="nav-text default">DASHBOARD</span>
              <span className="nav-text hover">DASHBOARD</span>
            </div>
          </button>
          <button
            className={`nav-tab ${activeTab === 'predictions' ? 'active' : ''}`}
            onClick={() => setActiveTab('predictions')}
          >
            <div className="nav-text-container">
              <span className="nav-text default">PREDICTIONS</span>
              <span className="nav-text hover">PREDICTIONS</span>
            </div>
          </button>
        </div>

        <div className="nav-user-info" style={{ display: 'flex', alignItems: 'center', gap: '16px', fontFamily: 'SF Mono, Fira Code, monospace', fontSize: '0.8rem' }}>
          <div style={{ color: 'var(--text-secondary)' }}>
            USER: <span style={{ color: '#fff' }}>{auth.currentUser?.email || auth.currentUser?.phoneNumber || 'GUEST'}</span>
          </div>
          <div style={{ background: 'rgba(249, 115, 22, 0.1)', border: '1px solid var(--accent-orange)', padding: '4px 12px', borderRadius: '4px', color: 'var(--accent-orange)' }}>
            CREDITS: <strong>{credits}</strong>
          </div>
          <button 
            onClick={() => auth.signOut()} 
            style={{ background: 'transparent', border: '1px solid #ef4444', color: '#ef4444', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '0.7rem' }}
          >
            DISCONNECT
          </button>
        </div>
      </div>
    </nav>
  );
}

export default Navigation;

