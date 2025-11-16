import React, { useState } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import SentimentAnalyzer from './components/SentimentAnalyzer';
import LiveFeed from './components/LiveFeed';
import Navigation from './components/Navigation';

function App() {
  const [activeTab, setActiveTab] = useState('analyzer');

  return (
    <div className="App">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="main-content">
        {activeTab === 'analyzer' && <SentimentAnalyzer />}
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'live' && <LiveFeed />}
      </div>
    </div>
  );
}

export default App;

