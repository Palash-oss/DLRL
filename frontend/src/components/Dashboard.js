import React, { useState, useRef, useMemo } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, PieChart, Pie, Cell } from 'recharts';
import './Dashboard.css';

gsap.registerPlugin(ScrollTrigger);

const COLORS = {
  positive: '#f97316', // Orange
  neutral: '#94a3b8', // Gray
  negative: '#ef4444'  // Red
};

function Dashboard({ batchData }) {
  const [activeModule, setActiveModule] = useState('LIVE FEED');
  const dashboardRef = useRef(null);

  useGSAP(() => {
    gsap.from('.cyber-card, .signal-item', {
      y: 30,
      opacity: 0,
      duration: 0.6,
      stagger: 0.1,
      ease: 'power2.out',
    });
  }, [batchData]);

  // Process Batch Data
  const stats = useMemo(() => {
    if (!batchData || !batchData.results_by_company) return null;

    let totalPositive = 0;
    let totalNeutral = 0;
    let totalNegative = 0;
    let total = batchData.total_processed || 0;
    let allItems = [];

    Object.values(batchData.results_by_company).forEach(companyData => {
      totalPositive += companyData.sentiment_counts.positive || 0;
      totalNeutral += companyData.sentiment_counts.neutral || 0;
      totalNegative += companyData.sentiment_counts.negative || 0;
      allItems = [...allItems, ...companyData.items];
    });

    const pieData = [
      { name: 'Positive', value: totalPositive, color: COLORS.positive },
      { name: 'Neutral', value: totalNeutral, color: COLORS.neutral },
      { name: 'Negative', value: totalNegative, color: COLORS.negative }
    ];

    // Mock trend data based on sentiment distribution for the visual
    const trendData = [
      { time: 'BATCH START', loss: 0.8 },
      { time: 'T-1', loss: 0.6 },
      { time: 'T-2', loss: totalNegative / total },
      { time: 'NOW', loss: (totalNegative / total) * 0.8 },
    ];

    return {
      total,
      totalPositive,
      totalNeutral,
      totalNegative,
      pieData,
      trendData,
      items: allItems.slice(0, 10) // Take first 10 for the feed
    };
  }, [batchData]);

  if (!stats) {
    return (
      <div className="amber-dashboard">
        <main className="cyber-main" style={{ gridColumn: '1 / -1', justifyContent: 'center', alignItems: 'center' }}>
          <div className="cyber-card" style={{ textAlign: 'center', maxWidth: '500px' }}>
            <h2 className="cyber-heading text-red mb-3">SYSTEM OFFLINE</h2>
            <p style={{ color: 'var(--text-secondary)' }}>No batch data loaded. Please return to the Analyzer and import a CSV dataset to initialize the neural matrix.</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="amber-dashboard" ref={dashboardRef}>
      {/* LEFT SIDEBAR: MODULES */}
      <aside className="cyber-sidebar">
        <div className="sidebar-header">MODULES</div>
        <ul className="module-list">
          {['BATCH STATS', 'HISTORY', 'MODELS', 'SETTINGS'].map((mod) => (
            <li 
              key={mod} 
              className={`module-item ${activeModule === mod ? 'active' : ''}`}
              onClick={() => setActiveModule(mod)}
            >
              <span className="mod-icon">◈</span> {mod}
            </li>
          ))}
        </ul>
      </aside>

      {/* CENTER: BATCH FEED & CHARTS */}
      <main className="cyber-main">
        <div className="cyber-header-row">
          <h2 className="cyber-heading"><span className="orange-dot">●</span> OVERALL BATCH METRICS</h2>
          <span className="cyber-filter">RECORDS: {stats.total}</span>
        </div>

        <div className="stats-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
            <div className="cyber-card">
              <div className="cam-label text-orange">SENTIMENT DISTRIBUTION</div>
              <div style={{ height: '180px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={stats.pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {stats.pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#0a0a0a', border: '1px solid #333' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="cyber-card">
              <div className="cam-label text-orange">KEY INDICATORS</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '16px' }}>
                 <div>
                   <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px' }}>POSITIVE RATIO</div>
                   <div style={{ fontSize: '1.5rem', fontWeight: 600, color: COLORS.positive }}>{((stats.totalPositive / stats.total) * 100).toFixed(1)}%</div>
                 </div>
                 <div>
                   <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px' }}>NEGATIVE RATIO</div>
                   <div style={{ fontSize: '1.5rem', fontWeight: 600, color: COLORS.negative }}>{((stats.totalNegative / stats.total) * 100).toFixed(1)}%</div>
                 </div>
              </div>
            </div>
        </div>

        <div className="cyber-header-row mb-3">
          <h3 className="cyber-subheading">👁 RECENT BATCH SIGNALS</h3>
        </div>

        <div className="signal-feed">
          {stats.items.map((item, idx) => (
            <div key={idx} className={`signal-item ${item.sentiment}`}>
              <div className="signal-meta">
                <span className="user-handle">{item.company || 'BATCH_RECORD'}</span>
              </div>
              <div className="signal-score">
                SENTIMENT: <span className="score-val" style={{ color: COLORS[item.sentiment] }}>{(item.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="signal-text" style={{ marginTop: '12px' }}>
                {item.text}
              </div>
            </div>
          ))}
        </div>
      </main>

      {/* RIGHT: NEURAL HEALTH */}
      <aside className="cyber-right">
        <div className="cyber-header-row">
          <h2 className="cyber-heading"><span className="orange-dot">●</span> NEURAL HEALTH</h2>
        </div>

        <div className="cyber-card model-convergence mb-4">
          <div className="cyber-header-row mb-3">
            <span className="cam-label">ERROR RATE TREND</span>
            <span className="cam-label">LOSS: {stats.trendData[3].loss.toFixed(3)}</span>
          </div>
          <div className="chart-container" style={{ height: '120px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.trendData}>
                <XAxis dataKey="time" hide />
                <Bar dataKey="loss" fill="rgba(249, 115, 22, 0.2)" />
                <LineChart data={stats.trendData} style={{position: 'absolute', top: 0, left: 0}}>
                   <Line type="monotone" dataKey="loss" stroke="#f97316" strokeWidth={2} dot={{ fill: '#f97316', r: 3 }} />
                </LineChart>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="x-axis-labels">
            <span>START</span>
            <span>T-2</span>
            <span>NOW</span>
          </div>
        </div>

        <div className="anomalies-section">
          <div className="cam-label text-red mb-2">⚠️ ACTIVE ANOMALIES</div>
          {stats.totalNegative > stats.totalPositive ? (
             <div className="anomaly-card">
              <div className="anom-info">
                <div className="anom-title">Negative Sentiment Spike</div>
                <div className="anom-cluster">CLUSTER: BATCH_DATA</div>
              </div>
              <button className="cyber-btn-outline">INVESTIGATE</button>
            </div>
          ) : (
            <div style={{ fontSize: '0.8rem', color: '#94a3b8', border: '1px solid #333', padding: '12px', textAlign: 'center' }}>
              NO MAJOR ANOMALIES DETECTED
            </div>
          )}
        </div>
      </aside>

      {/* BOTTOM STATUS BAR */}
      <footer className="cyber-footer">
        <div className="sys-load">
          INGESTION LOAD <div className="load-bar"><div className="load-fill" style={{ width: '100%' }}></div></div> COMPLETED
        </div>
        <div className="sys-status">
          <span className="orange-dot">●</span> Tokyo_Edge_04 // ONLINE
        </div>
      </footer>
    </div>
  );
}

export default Dashboard;

