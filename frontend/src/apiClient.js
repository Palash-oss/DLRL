import axios from 'axios';

const apiClient = axios.create({
  // Use 8010 by default to match the FastAPI server we run locally
  baseURL: process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:8010',
  headers: {
    'Content-Type': 'application/json'
  }
});

export default apiClient;
