import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// TODO: Replace with your actual Firebase project configuration
const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID
};

let app;
let auth;

try {
  if (!process.env.REACT_APP_FIREBASE_API_KEY || process.env.REACT_APP_FIREBASE_API_KEY === 'YOUR_API_KEY') {
    throw new Error("Firebase API Key is missing or placeholder. Running in Simulation Mode.");
  }
  app = initializeApp(firebaseConfig);
  auth = getAuth(app);
} catch (error) {
  console.warn("ACRUX_AUTH_LOAD_FAILED: Running in localized simulation mode.", error.message);
  // Mock Auth object to prevent startup crashes
  auth = {
    currentUser: null,
    onAuthStateChanged: (callback) => {
      // Simulate no user logged in by default
      callback(null);
      return () => {};
    },
    signInWithEmailAndPassword: () => Promise.reject({ code: 'auth/invalid-api-key' }),
    createUserWithEmailAndPassword: () => Promise.reject({ code: 'auth/invalid-api-key' }),
    signOut: () => { window.location.href = '/login'; return Promise.resolve(); }
  };
}

export { auth };
