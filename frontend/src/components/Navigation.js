import React from 'react';
import './Navigation.css';
import { RiBarChartLine, RiDashboardLine, RiRadioButtonLine } from 'react-icons/ri';

function Navigation({ activeTab, setActiveTab }) {
  return (
    <nav className="navigation">
      <div className="nav-container">
        <div className="nav-brand">
          <h1>Sentiment</h1>
        </div>
        <div className="nav-tabs">
          <button
            className={`nav-tab ${activeTab === 'analyzer' ? 'active' : ''}`}
            onClick={() => setActiveTab('analyzer')}
          >
            <RiBarChartLine size={16} />
            <span>Analyzer</span>
          </button>
          <button
            className={`nav-tab ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <RiDashboardLine size={16} />
            <span>Dashboard</span>
          </button>
          <button
            className={`nav-tab ${activeTab === 'live' ? 'active' : ''}`}
            onClick={() => setActiveTab('live')}
          >
            <RiRadioButtonLine size={16} />
            <span>Live</span>
          </button>
        </div>
      </div>
    </nav>
  );
}

export default Navigation;

