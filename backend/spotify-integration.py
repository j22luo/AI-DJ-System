import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    redirect_uri="http://localhost:8888/callback",
    scope="user-library-read user-read-playback-state user-modify-playback-state"
))

# Get user's playlists
playlists = sp.current_user_playlists()

# Get tracks from a playlist
tracks = sp.playlist_tracks("playlist_id")

# Get current playing track
current = sp.current_playback()

# Queue next track
sp.add_to_queue("spotify:track:TRACK_ID")

# Get audio features (energy, danceability, valence)
features = sp.audio_features("track_id")