# authenticate_spotify.py
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from config import Config

def authenticate():
    """
    Run this ONCE to authenticate with Spotify.
    It will save the token to .spotify_cache for future use.
    """
    print("=" * 60)
    print("SPOTIFY AUTHENTICATION")
    print("=" * 60)
    print("\nThis will open a browser window for you to log in to Spotify.")
    print("After logging in, you'll be redirected. Copy the full URL and paste it here.")
    print()
    
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=Config.SPOTIFY_CLIENT_ID,
        client_secret=Config.SPOTIFY_CLIENT_SECRET,
        redirect_uri=Config.SPOTIFY_REDIRECT_URI,
        scope="user-library-read user-read-playback-state user-modify-playback-state",
        cache_path=".spotify_cache",
        open_browser=True  # Will open browser
    ))
    
    # Test authentication
    try:
        user = sp.current_user()
        print(f"\n✅ Successfully authenticated as: {user['display_name']}")
        print(f"✅ Token saved to .spotify_cache")
        print(f"\nYou can now start the MCP server!")
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")

if __name__ == "__main__":
    authenticate()