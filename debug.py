# debug_audio_features.py
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from config import Config
from pathlib import Path
import json

# Initialize the Authentication Manager
auth_manager = SpotifyClientCredentials(
    client_id=Config.SPOTIFY_CLIENT_ID,
    client_secret=Config.SPOTIFY_CLIENT_SECRET
)

def debug_audio_features():
    """Comprehensive debug of audio features issue"""
    
    print("=" * 60)
    print("AUDIO FEATURES DEBUG")
    print("=" * 60)
    
    # Setup
    project_root = Path(__file__).parent
    cache_path = project_root / ".spotify_cache"
    
    print(f"\n1️⃣  Cache file check:")
    print(f"   Path: {cache_path}")
    print(f"   Exists: {cache_path.exists()}")
    
    if cache_path.exists():
        with open(cache_path) as f:
            token_data = json.load(f)
        print(f"   Scopes: {token_data.get('scope', 'N/A')}")
        
        import time
        expires_at = token_data.get('expires_at', 0)
        if expires_at < time.time():
            print(f"   ⚠️  TOKEN EXPIRED! Need to re-authenticate")
        else:
            remaining = int((expires_at - time.time()) / 60)
            print(f"   ✅ Token valid for {remaining} more minutes")
    
    # Initialize client
    print(f"\n2️⃣  Initializing Spotify client...")
    
    scopes = [
        "user-library-read",
        "user-read-playback-state",
        "user-modify-playback-state",
        "user-read-currently-playing",
        "playlist-read-private",
        "playlist-read-collaborative"
    ]
    
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=Config.SPOTIFY_CLIENT_ID,
        client_secret=Config.SPOTIFY_CLIENT_SECRET,
        redirect_uri=Config.SPOTIFY_REDIRECT_URI,
        scope=" ".join(scopes),
        cache_path=str(cache_path),
        open_browser=False
    ))
    # sp = spotipy.Spotify(auth_manager=auth_manager)
    
    try:
        user = sp.current_user()
        print(f"   ✅ Logged in as: {user['display_name']}")
        print(f"   User ID: {user['id']}")
        print(f"   Country: {user.get('country', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Auth failed: {e}")
        return
    
    # Get current playback
    print(f"\n3️⃣  Getting current playback...")
    
    try:
        playback = sp.current_playback()
        
        if not playback:
            print(f"   ❌ No playback data returned")
            print(f"   Make sure Spotify is open and playing a song!")
            return
        
        if not playback.get('item'):
            print(f"   ❌ No track in playback")
            print(f"   Playback data: {playback}")
            return
        
        track = playback['item']
        track_id = track['id']
        track_uri = track['uri']
        track_name = track['name']
        track_artist = track['artists'][0]['name']
        
        print(f"   ✅ Currently playing: {track_name} by {track_artist}")
        print(f"   Track ID: {track_id}")
        print(f"   Track URI: {track_uri}")
        print(f"   Track type: {type(track_id)}")
        print(f"   ID length: {len(track_id)}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Try audio_features with different approaches
    print(f"\n4️⃣  Testing audio_features()...")
    
    # Test 1: With list (correct way)
    print(f"\n   Test 1: sp.audio_features(['{track_id}'])")
    try:
        features = sp.audio_features([track_id])
        print(f"   Response type: {type(features)}")
        print(f"   Response: {features}")
        
        if features:
            print(f"   Response length: {len(features)}")
            if features[0]:
                print(f"   ✅ SUCCESS!")
                print(f"   Energy: {features[0].get('energy')}")
                print(f"   Tempo: {features[0].get('tempo')}")
            else:
                print(f"   ⚠️  First element is None")
        else:
            print(f"   ⚠️  Response is None or empty")
            
    except Exception as e:
        print(f"   ❌ FAILED!")
        print(f"   Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Get detailed error info
        if hasattr(e, '__dict__'):
            print(f"   Error details: {e.__dict__}")
        
        import traceback
        print(f"\n   Full traceback:")
        traceback.print_exc()
    
    # Test 2: Using track() endpoint instead
    print(f"\n   Test 2: sp.track('{track_id}') [alternative]")
    try:
        track_info = sp.track(track_id)
        print(f"   ✅ Track endpoint works!")
        print(f"   Track name: {track_info['name']}")
        print(f"   Popularity: {track_info.get('popularity')}")
    except Exception as e:
        print(f"   ❌ Track endpoint also failed: {e}")
    
    # Test 3: Check API permissions
    print(f"\n5️⃣  Checking API access...")
    
    try:
        # Try different endpoints to see what works
        print(f"   Testing user_playlists...")
        playlists = sp.current_user_playlists(limit=1)
        print(f"   ✅ Playlists work")
        
        print(f"   Testing playback...")
        playback = sp.current_playback()
        print(f"   ✅ Playback works")
        
        print(f"   Testing saved_tracks...")
        saved = sp.current_user_saved_tracks(limit=1)
        print(f"   ✅ Saved tracks work")
        
    except Exception as e:
        print(f"   ⚠️  Some endpoints failing: {e}")
    
    # Test 4: Direct API call (bypass spotipy)
    print(f"\n6️⃣  Testing direct API call...")
    
    try:
        import requests
        
        # Get token
        token = sp.auth_manager.get_access_token(as_dict=False)
        
        print(f"   Token (first 20 chars): {token[:20]}...")
        
        # Direct API call
        url = f"https://api.spotify.com/v1/audio-features/{track_id}"
        headers = {"Authorization": f"Bearer {token}"}
        
        print(f"   URL: {url}")
        print(f"   Headers: Authorization: Bearer {token[:20]}...")
        
        response = requests.get(url, headers=headers)
        
        print(f"   Status code: {response.status_code}")
        print(f"   Response headers: {dict(response.headers)}")
        print(f"   Response body: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Direct API call SUCCESS!")
            print(f"   Energy: {data.get('energy')}")
        else:
            print(f"   ❌ Direct API call FAILED!")
            
    except Exception as e:
        print(f"   ❌ Direct API error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    debug_audio_features()