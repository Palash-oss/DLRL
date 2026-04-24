import React, { useState } from 'react';
import { auth } from '../firebase';
import { 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider
} from 'firebase/auth';
import { useNavigate } from 'react-router-dom';
import apiClient from '../apiClient';
import './AuthPage.css';

function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [step, setStep] = useState(1); // 1 = Creds, 2 = OTP
  const [name, setName] = useState('');
  const [contact, setContact] = useState(''); // Email or Phone
  const [password, setPassword] = useState('');
  const [otp, setOtp] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleAuthStep1 = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // Basic validation
      const isEmail = contact.includes('@');
      const loginEmail = isEmail ? contact : `${contact}@phone.acrux.net`;

      if (isLogin) {
        // Step 1 of Login -> Verify creds, move to OTP
        try {
          await signInWithEmailAndPassword(auth, loginEmail, password);
        } catch (firebaseErr) {
          if (firebaseErr.code === 'auth/invalid-api-key' && password === '123456') {
             console.log("SIMULATED LOGIN - FIREBASE NOT CONFIGURED YET");
             // Fake UID for local mongo
             auth.currentUser = { uid: "simulated_user_123", email: loginEmail };
          } else {
             throw firebaseErr;
          }
        }
        setStep(2); // Move to OTP
      } else {
        // Sign Up -> Direct
        if (!name) throw new Error("Name is required");
        
        let uid = "simulated_user_123";
        try {
           const userCred = await createUserWithEmailAndPassword(auth, loginEmail, password);
           uid = userCred.user.uid;
        } catch (firebaseErr) {
           if (firebaseErr.code === 'auth/invalid-api-key') {
              console.log("SIMULATED SIGNUP - FIREBASE NOT CONFIGURED YET");
              auth.currentUser = { uid: uid, email: loginEmail };
           } else {
              throw firebaseErr;
           }
        }
        
        // Initialize user with 200 credits in MongoDB
        await apiClient.post('/api/users/init', {
          name,
          email: isEmail ? contact : null,
          phone: !isEmail ? contact : null
        }, {
          headers: { 'X-User-Id': uid }
        });
        
        navigate('/app');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyOTP = (e) => {
    e.preventDefault();
    setLoading(true);
    // Simulate OTP verification
    setTimeout(() => {
      if (otp === '123456') {
        navigate('/app');
      } else {
        setError('Invalid OTP Code. Hint: Use 123456');
        setLoading(false);
      }
    }, 1000);
  };

  const handleGoogleSignIn = async () => {
    const provider = new GoogleAuthProvider();
    try {
      const result = await signInWithPopup(auth, provider);
      
      // Attempt to init (if they already exist, backend handles it safely)
      await apiClient.post('/api/users/init', {
        name: result.user.displayName || 'Google User',
        email: result.user.email,
      }, {
        headers: { 'X-User-Id': result.user.uid }
      });

      navigate('/app');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card cyber-card">
        <div className="auth-header">
          <h2 className="cyber-heading">{isLogin ? 'SYSTEM_ACCESS' : 'CREATE_IDENTITY'}</h2>
          <p className="auth-subtitle">SECURE_UPLINK_V2.0</p>
        </div>

        {error && <div className="auth-error">{error}</div>}

        {step === 1 ? (
          <form onSubmit={handleAuthStep1} className="auth-form">
            {!isLogin && (
              <div className="input-group">
                <label>FULL_NAME</label>
                <input 
                  type="text" 
                  value={name} 
                  onChange={(e) => setName(e.target.value)} 
                  placeholder="Cipher Protocol"
                  required={!isLogin}
                />
              </div>
            )}
            <div className="input-group">
              <label>CONTACT (EMAIL OR NUMBER)</label>
              <input 
                type="text" 
                value={contact} 
                onChange={(e) => setContact(e.target.value)} 
                placeholder="user@acrux.network or +1234567890"
                required 
              />
            </div>
            <div className="input-group">
              <label>ACCESS_CODE (PASSWORD)</label>
              <input 
                type="password" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)} 
                placeholder="••••••••"
                required 
              />
            </div>

            <button type="submit" className="cyber-btn primary" disabled={loading}>
              {loading ? 'PROCESSING...' : (isLogin ? 'REQUEST OTP' : 'ESTABLISH IDENTITY (+200 CREDITS)')}
            </button>
          </form>
        ) : (
          <form onSubmit={handleVerifyOTP} className="auth-form">
            <div className="input-group">
              <label>ONE-TIME PASSWORD (OTP)</label>
              <input 
                type="text" 
                value={otp} 
                onChange={(e) => setOtp(e.target.value)} 
                placeholder="Sent to your contact..."
                required 
              />
              <p style={{fontSize: '0.65rem', color: 'var(--accent-orange)', marginTop: '4px'}}>
                System Note: Enter 123456 to bypass simulation.
              </p>
            </div>
            <button type="submit" className="cyber-btn primary" disabled={loading}>
              {loading ? 'VERIFYING...' : 'CONFIRM UPLINK'}
            </button>
          </form>
        )}

        {step === 1 && (
          <>
            <div className="auth-divider">
              <span>OR</span>
            </div>

            <button onClick={handleGoogleSignIn} className="cyber-btn-outline google-btn">
              CONTINUE_WITH_GO_ID
            </button>

            <div className="auth-footer">
              <p onClick={() => {setIsLogin(!isLogin); setError('');}}>
                {isLogin ? 'NEED_NEW_IDENTITY?' : 'ALREADY_HAVE_IDENTITY?'} 
                <span className="text-orange"> {isLogin ? 'SIGN_UP' : 'LOGIN'}</span>
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default AuthPage;
