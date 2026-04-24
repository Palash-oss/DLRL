import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import Dashboard from './components/Dashboard';
import BatchAnalysis from './components/BatchAnalysis';
import FuturePredictions from './components/FuturePredictions';
import Navigation from './components/Navigation';
import Background3D from './components/Background3D';
import LandingPage from './components/LandingPage';
import AuthPage from './components/AuthPage';
import { auth } from './firebase';
import { onAuthStateChanged } from 'firebase/auth';
import LocomotiveScroll from 'locomotive-scroll';

function MainApp() {
  const [activeTab, setActiveTab] = useState('analyzer');
  const [batchData, setBatchData] = useState(null);
  const [user, setUser] = useState(null);
  
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });

    // Locomotive Scroll v5 initialization with improved smooth settings
    const locoScroll = new LocomotiveScroll({
      lenisOptions: {
        lerp: 0.1,
        smoothWheel: true,
        wheelMultiplier: 1.2,
      }
    });

    return () => {
      unsubscribe();
      if (locoScroll) locoScroll.destroy();
    };
  }, []);

  if (user === undefined) return null; // Wait for auth check

  return (
    <div className="App">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="main-content">
        {activeTab === 'analyzer' && <BatchAnalysis setActiveTab={setActiveTab} setBatchData={setBatchData} />}
        {activeTab === 'dashboard' && <Dashboard batchData={batchData} />}
        {activeTab === 'predictions' && <FuturePredictions batchData={batchData} />}
      </div>
    </div>
  );
}

function ProtectedRoute({ children }) {
  // Bypassing auth check for now as requested
  return children;
}

function App() {
  return (
    <Router>
      <Background3D />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<AuthPage />} />
        <Route path="/app" element={
          <ProtectedRoute>
            <MainApp />
          </ProtectedRoute>
        } />
      </Routes>
    </Router>
  );
}

export default App;
