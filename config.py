import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Spotify
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8080"
    SPOTIFY_TEST_PLAYLIST_ID = "7y7qS22Lk07wrwEdYbGdA"
    PLAYLIST_MAX_SIZE = 10
    
    # # Anthropic
    # ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # # Flask
    # FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    # FLASK_HOST = "0.0.0.0"
    # FLASK_PORT = 5000
    
    # # App settings
    # CROWD_ANALYSIS_INTERVAL = 30  # seconds
    # MAX_TRACK_SUGGESTIONS = 5
    
    # # Paths
    # CAMERA_INDEX = 0  # Webcam