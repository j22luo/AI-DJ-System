import spotipy
from spotipy.oauth2 import SpotifyOAuth
from config import Config
from pathlib import Path

def authenticate():
    """
    Run this ONCE to authenticate with Spotify.
    """
    project_root = Path(__file__).parent
    cache_path = project_root / ".spotify_cache"
    
    print("=" * 60)
    print("SPOTIFY AUTHENTICATION")
    print("=" * 60)
    print(f"\nCache will be saved to: {cache_path}")
    print("\nThis will open a browser window for you to log in to Spotify.")
    print()
    
    # Ensure cache directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ✅ Add ALL required scopes
    scopes = [
        "user-library-read",           # Read saved tracks
        "user-read-playback-state",    # See what's playing
        "user-modify-playback-state",  # Control playback
        "user-read-currently-playing", # Current track
        "playlist-read-private",       # Read playlists
        "playlist-read-collaborative", # Collaborative playlists
        # NO EXTRA SCOPE NEEDED - audio_features is available without special scope
    ]
    
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=Config.SPOTIFY_CLIENT_ID,
        client_secret=Config.SPOTIFY_CLIENT_SECRET,
        redirect_uri=Config.SPOTIFY_REDIRECT_URI,
        scope=" ".join(scopes),  # Join scopes with space
        cache_path=str(cache_path),
        open_browser=True
    ))
    
    # Test authentication
    try:
        user = sp.current_user()
        print(f"\n✅ Successfully authenticated as: {user['display_name']}")
        print(f"✅ Token saved to: {cache_path}")
        
        # Test audio features
        print(f"\n🧪 Testing audio features access...")
        playback = sp.current_playback()
        if playback and playback.get('item'):
            track_id = playback['item']['id']
            features = sp.audio_features([track_id])
            # if features and features[0]:
            #     print(f"✅ Audio features access confirmed!")
            #     print(f"   Energy: {features[0]['energy']}")
            # else:
            #     print(f"⚠️  Audio features returned None")
        else:
            print(f"⚠️  No track playing - can't test audio features")
            print(f"   (This is OK, just start playing something in Spotify)")
        
        print(f"\n✅ You can now start the MCP server!")
        
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    authenticate()