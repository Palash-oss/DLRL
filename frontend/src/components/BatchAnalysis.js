import React, { useState, useRef, useEffect } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import { RiFileUploadLine, RiFileListLine, RiHistoryLine } from 'react-icons/ri';
import apiClient from '../apiClient';
import { auth } from '../firebase';
import './Dashboard.css';

function BatchAnalysis({ setActiveTab, setBatchData }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeModule, setActiveModule] = useState('DATA IMPORT');
  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    if (activeModule === 'BATCH HISTORY') {
      fetchHistory();
    }
  }, [activeModule]);

  const fetchHistory = async () => {
    setLoadingHistory(true);
    try {
      const user = auth.currentUser;
      const uid = user ? user.uid : 'guest_user_101';
      const response = await apiClient.get('/api/history', {
        headers: { 'X-User-Id': uid }
      });
      setHistory(response.data.history || []);
    } catch (err) {
      console.error('Failed to fetch history:', err);
    } finally {
      setLoadingHistory(false);
    }
  };

  useGSAP(() => {
    if (containerRef.current) {
      gsap.from('.cyber-card', {
        y: 30,
        opacity: 0,
        duration: 0.6,
        ease: 'power2.out',
      });
    }
  }, [activeModule]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a file to analyze');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const user = auth.currentUser;
      const headers = { 'X-User-Id': user ? user.uid : 'guest_user_101' };
      
      const response = await apiClient.post('/api/batch-analyze', formData, { headers });
      setBatchData(response.data);
      setActiveTab('dashboard');
    } catch (err) {
      console.error('Batch analysis error:', err);
      let errorMessage = 'An error occurred during batch analysis.';
      if (err.response) {
        errorMessage = err.response.data.detail || `Server Error (${err.response.status})`;
      } else {
        errorMessage = err.message;
      }
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const loadPastAnalysis = (data) => {
    setBatchData(data);
    setActiveTab('dashboard');
  };

  return (
    <div className="amber-dashboard" ref={containerRef}>
      {/* LEFT SIDEBAR: MODULES */}
      <aside className="cyber-sidebar">
        <div className="sidebar-header">MODULES</div>
        <ul className="module-list">
          {['DATA IMPORT', 'BATCH HISTORY', 'EXPORT', 'SETTINGS'].map((mod) => (
            <li 
              key={mod} 
              className={`module-item ${activeModule === mod ? 'active' : ''}`}
              onClick={() => setActiveModule(mod)}
            >
              <span className="mod-icon">{mod === 'BATCH HISTORY' ? '⟲' : '◈'}</span> {mod}
            </li>
          ))}
        </ul>
      </aside>

      {/* CENTER: CONTENT */}
      <main className="cyber-main" style={{ gridColumn: '2 / -1' }}>
        {activeModule === 'DATA IMPORT' ? (
          <>
            <div className="cyber-header-row mb-4">
              <h2 className="cyber-heading"><span className="orange-dot">●</span> DATA IMPORT // BATCH ANALYSIS</h2>
              <span className="cyber-filter">STATUS: AWAITING INPUT</span>
            </div>

            <div className="cyber-card" style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>
              <div className="cam-label text-orange mb-3">▤ UPLOAD FEEDBACK DATA</div>
              <p style={{ marginBottom: '32px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                Upload your feedback data (CSV, Excel, PDF) to generate AI-powered insights and sentiment analysis.
              </p>

              <div className="input-group">
                <label htmlFor="file-upload" className="file-upload-label" style={{ width: '100%' }}>
                  <div className={`upload-area ${file ? 'has-file' : ''}`} style={{
                    borderColor: file ? '#f97316' : 'rgba(255, 255, 255, 0.1)',
                    background: file ? 'rgba(249, 115, 22, 0.05)' : 'rgba(10, 10, 10, 0.8)',
                    padding: '40px',
                    borderRadius: '4px',
                    borderStyle: 'dashed',
                    borderWidth: '1px',
                    transition: 'all 0.3s ease',
                    cursor: 'pointer',
                    textAlign: 'center'
                  }}>
                    <RiFileUploadLine size={32} color={file ? '#f97316' : '#94a3b8'} style={{ marginBottom: '16px' }} />

                    {file ? (
                      <div>
                        <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#f8fafc', marginBottom: '4px', fontFamily: 'SF Mono, Fira Code, monospace' }}>
                          {file.name}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                          {(file.size / 1024).toFixed(1)} KB
                        </div>
                      </div>
                    ) : (
                      <div>
                        <span style={{ display: 'block', fontWeight: 500, marginBottom: '8px', fontSize: '0.9rem', color: '#e2e8f0', fontFamily: 'SF Mono, Fira Code, monospace' }}>
                          Drop your file here or click to browse
                        </span>
                      </div>
                    )}
                  </div>
                </label>
                <input
                  id="file-upload"
                  type="file"
                  accept=".csv,.xlsx,.xls,.pdf"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
              </div>

              {error && <div className="auth-error mt-4">{error}</div>}

              <div style={{ marginTop: '32px', display: 'flex', justifyContent: 'flex-end' }}>
                <button className="cyber-btn primary" onClick={handleAnalyze} disabled={loading || !file}>
                  {loading ? 'PROCESSING...' : 'RUN ANALYSIS'}
                </button>
              </div>
            </div>
          </>
        ) : activeModule === 'BATCH HISTORY' ? (
          <>
            <div className="cyber-header-row mb-4">
              <h2 className="cyber-heading"><span className="orange-dot">●</span> BATCH HISTORY // PERSISTENT CACHE</h2>
              <span className="cyber-filter">USER: {auth.currentUser?.email}</span>
            </div>

            <div className="history-grid" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {loadingHistory ? (
                <div className="cyber-card" style={{ textAlign: 'center', padding: '40px' }}>ACCESSING_MONGO_DB...</div>
              ) : history.length > 0 ? (
                history.map((item, idx) => (
                  <div key={idx} className="cyber-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div className="text-orange" style={{ fontSize: '0.9rem', fontWeight: 600 }}>{item.filename || 'UNNAMED_BATCH'}</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>{new Date(item.timestamp).toLocaleString()} // {item.total_processed} RECORDS</div>
                    </div>
                    <button className="cyber-btn-outline" onClick={() => loadPastAnalysis(item)}>RESTORE_SESSION</button>
                  </div>
                ))
              ) : (
                <div className="cyber-card" style={{ textAlign: 'center', padding: '40px', color: 'var(--text-secondary)' }}>NO_HISTORY_FOUND</div>
              )}
            </div>
          </>
        ) : (
          <div className="cyber-card" style={{ textAlign: 'center', padding: '40px' }}>MODULE_UNDER_CONSTRUCTION</div>
        )}
      </main>

      {/* BOTTOM STATUS BAR */}
      <footer className="cyber-footer">
        <div className="sys-load">
          DATABASE STATUS <div className="load-bar"><div className="load-fill" style={{width: '100%', background: '#22c55e'}}></div></div> CONNECTED
        </div>
        <div className="sys-status">
          <span className="orange-dot">●</span> DB_CLUSTER_MONGO // READY
        </div>
      </footer>
    </div>
  );
}

export default BatchAnalysis;

