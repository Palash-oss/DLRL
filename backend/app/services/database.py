"""
Database manager for sentiment analysis predictions
Uses SQLite for persistence
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class Database:
    def __init__(self, db_path: str = "sentiment_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                text TEXT,
                has_image BOOLEAN DEFAULT 0,
                sentiment VARCHAR(20),
                confidence REAL,
                neg_prob REAL,
                neu_prob REAL,
                pos_prob REAL,
                compound_score REAL,
                modality VARCHAR(20),
                metadata TEXT
            )
        ''')
        
        # Analytics table for dashboard
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE DEFAULT CURRENT_DATE,
                total_predictions INTEGER DEFAULT 0,
                positive_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0,
                negative_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                state TEXT,
                action INTEGER,
                reward REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, prediction_data: Dict) -> int:
        """Save a prediction to database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (text, has_image, sentiment, confidence, neg_prob, neu_prob, pos_prob, 
             compound_score, modality, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_data.get('text', '')[:1000],  # Limit text length
            prediction_data.get('has_image', False),
            prediction_data.get('sentiment'),
            prediction_data.get('confidence'),
            prediction_data.get('probabilities', {}).get('negative'),
            prediction_data.get('probabilities', {}).get('neutral'),
            prediction_data.get('probabilities', {}).get('positive'),
            prediction_data.get('compound_score', 0.0),
            prediction_data.get('modality', 'text'),
            json.dumps(prediction_data.get('metadata', {}))
        ))
        
        prediction_id = cursor.lastrowid
        
        # Update analytics
        self._update_analytics(cursor, prediction_data)
        
        conn.commit()
        conn.close()
        
        return prediction_id

    def save_feedback(self, feedback_data: Dict) -> int:
        """Store user feedback for RL loop."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            '''
            INSERT INTO feedback (state, action, reward)
            VALUES (?, ?, ?)
            ''',
            (
                json.dumps(feedback_data.get('state', [])),
                feedback_data.get('action', 0),
                feedback_data.get('reward', 0.0),
            ),
        )

        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return feedback_id
    
    def _update_analytics(self, cursor, prediction_data: Dict):
        """Update daily analytics."""
        today = datetime.now().date()
        
        # Check if today's analytics exist
        cursor.execute('SELECT id FROM analytics WHERE date = ?', (today,))
        result = cursor.fetchone()
        
        sentiment = prediction_data.get('sentiment', 'neutral')
        confidence = prediction_data.get('confidence', 0.0)
        
        if result:
            # Update existing
            cursor.execute(f'''
                UPDATE analytics 
                SET total_predictions = total_predictions + 1,
                    {sentiment}_count = {sentiment}_count + 1,
                    avg_confidence = (avg_confidence * total_predictions + ?) / (total_predictions + 1)
                WHERE date = ?
            ''', (confidence, today))
        else:
            # Create new
            pos_count = 1 if sentiment == 'positive' else 0
            neu_count = 1 if sentiment == 'neutral' else 0
            neg_count = 1 if sentiment == 'negative' else 0
            
            cursor.execute('''
                INSERT INTO analytics 
                (date, total_predictions, positive_count, neutral_count, negative_count, avg_confidence)
                VALUES (?, 1, ?, ?, ?, ?)
            ''', (today, pos_count, neu_count, neg_count, confidence))
    
    def get_recent_predictions(self, limit: int = 50) -> List[Dict]:
        """Get recent predictions."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, text, has_image, sentiment, confidence, 
                   neg_prob, neu_prob, pos_prob, modality
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        conn.close()
        return results
    
    def get_statistics(self, days: int = 7) -> Dict:
        """Get statistics for dashboard."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total = cursor.fetchone()[0]
        
        # Sentiment distribution
        cursor.execute('''
            SELECT sentiment, COUNT(*) as count
            FROM predictions
            GROUP BY sentiment
        ''')
        sentiment_dist = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Recent analytics (last N days)
        cursor.execute('''
            SELECT date, total_predictions, positive_count, neutral_count, 
                   negative_count, avg_confidence
            FROM analytics
            ORDER BY date DESC
            LIMIT ?
        ''', (days,))
        
        daily_stats = []
        for row in cursor.fetchall():
            daily_stats.append({
                'date': row[0],
                'total': row[1],
                'positive': row[2],
                'neutral': row[3],
                'negative': row[4],
                'avg_confidence': row[5]
            })
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM predictions WHERE confidence IS NOT NULL')
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # Modality distribution
        cursor.execute('''
            SELECT modality, COUNT(*) as count
            FROM predictions
            GROUP BY modality
        ''')
        modality_dist = {row[0]: row[1] for row in cursor.fetchall()}

        feedback_limit = max(days * 10, 50)
        cursor.execute('''
            SELECT timestamp, reward
            FROM feedback
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (feedback_limit,))
        feedback_curve = []
        cumulative_reward = 0.0
        for episode, row in enumerate(cursor.fetchall(), start=1):
            reward = row[1] or 0.0
            cumulative_reward += reward
            feedback_curve.append({
                'episode': episode,
                'reward': reward,
                'total_reward': cumulative_reward,
                'timestamp': row[0],
            })
        
        conn.close()
        
        return {
            'total_predictions': total,
            'sentiment_distribution': sentiment_dist,
            'daily_stats': daily_stats,
            'avg_confidence': avg_confidence,
            'modality_distribution': modality_dist,
            'feedback_curve': feedback_curve
        }
    
    def clear_old_data(self, days: int = 30):
        """Clear data older than N days."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM predictions
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        cursor.execute('''
            DELETE FROM analytics
            WHERE date < date('now', '-' || ? || ' days')
        ''', (days,))
        
        conn.commit()
        conn.close()


# Global database instance
_db = None

def get_database(db_path: str = "sentiment_data.db") -> Database:
    """Get or create database singleton."""
    global _db
    if _db is None:
        _db = Database(db_path)
    return _db
