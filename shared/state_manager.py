import asyncio
from typing import Optional
from datetime import datetime

class StateManager:
    """
    Shared state between MCP server and Flask app.
    Thread-safe singleton for sharing data.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._lock = asyncio.Lock()
        
        # Current state
        self.current_track = None
        self.crowd_sentiment = None
        self.track_suggestions = []
        self.last_analysis_time = None
        
    async def update_current_track(self, track_data: dict):
        """Update current track info"""
        async with self._lock:
            self.current_track = track_data
    
    async def update_crowd_sentiment(self, sentiment_data: dict):
        """Update crowd sentiment from vision analysis"""
        async with self._lock:
            self.crowd_sentiment = sentiment_data
            self.last_analysis_time = datetime.now()
    
    async def update_suggestions(self, suggestions: list):
        """Update track suggestions from AI"""
        async with self._lock:
            self.track_suggestions = suggestions
    
    async def get_state(self) -> dict:
        """Get complete current state"""
        async with self._lock:
            return {
                'current_track': self.current_track,
                'crowd_sentiment': self.crowd_sentiment,
                'suggestions': self.track_suggestions,
                'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None
            }

# Global instance
state = StateManager()