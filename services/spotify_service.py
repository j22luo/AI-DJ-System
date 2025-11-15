import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Optional, List, Dict
import asyncio
from config import Config

class SpotifyService():
    """API Wrapper for interacting with Spotify"""
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=Config.SPOTIFY_CLIENT_ID,
            client_secret=Config.SPOTIFY_CLIENT_SECRET,
            redirect_uri=Config.SPOTIFY_REDIRECT_URI,
            scope="user-library-read user-read-playback-state user-modify-playback-state",
            cache_path=".spotify_cache",
            open_browser=True  # Will open browser
        ))
        self._queue_tasks = {}

    # Claude Tools

    async def get_current_track_audio_features(self) -> Optional[Dict]:
        loop = asyncio.get_event_loop()
        playback = await loop.run_in_executor(None, self.sp.current_playback)

        if not playback or not playback.get('item'):
            return None
        
        track = playback['item']
        track_id = playback['item']['id']

        features = await loop.run_in_executor(None, self.sp.audio_features, track_id)

        f = features[0]
        return {
            'energy': f['energy'],
            'danceability': f['danceability'],
            'valence': f['valence'],
            'tempo': f['tempo'],
            'loudness': f['loudness'],
            'acousticness': f['acousticness'],
            'instrumentalness': f['instrumentalness'],
            'speechiness': f['speechiness'],
            'duration_ms': track['duration_ms'],
            'progress_ms': playback.get('progress_ms', 0),
        }

    async def get_test_playlist(self) -> Optional[Dict]:
        """Get tracks from fixed test playlist"""
        try:
            loop = asyncio.get_event_loop()
            
            playlist = await loop.run_in_executor(
                None, 
                self.sp.playlist, 
                Config.SPOTIFY_TEST_PLAYLIST_ID
            )
            
            tracks_data = await loop.run_in_executor(
                None,
                self.sp.playlist_tracks,
                Config.SPOTIFY_TEST_PLAYLIST_ID
            )
            
            tracks = []
            for item in tracks_data['items']:
                if item['track']:  
                    track = item['track']
                    tracks.append({
                        'id': track['id'],
                        'uri': track['uri'],
                        'title': track['name'],
                        'artist': ', '.join([a['name'] for a in track['artists']]),
                        'album': track['album']['name'],
                        'duration_ms': track['duration_ms'],
                        'album_art': track['album']['images'][0]['url'] if track['album']['images'] else None,
                        'popularity': track.get('popularity', 0)
                    })
            
            return {
                'id': playlist['id'],
                'name': playlist['name'],
                'total_tracks': len(tracks),
                'tracks': tracks
            }
            
        except Exception as e:
            print(f"Error fetching test playlist: {e}")
            return None

    async def queue_track(self, uri: str, offset: float = 0.0) -> bool:
        """
        Queue a track to play after specified offset time
        """

        try:
            if offset <= 0:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.sp.add_to_queue, uri)
                return True
            else:
                # Schedule for later
                task = asyncio.create_task(self._queue_after_delay(uri, offset))
                self._queue_tasks[uri] = task
                return True
                
        except Exception as e:
            print(f"Error queueing track: {e}")
            return False

    async def get_multiple_audio_features(self, track_ids: List[str]) -> Dict[str, Dict]:
        """
        Get audio features for multiple tracks efficiently
        """
        try:
            # Spotify API allows batch fetching up to 100 tracks
            loop = asyncio.get_event_loop()
            features_list = await loop.run_in_executor(
                None, 
                self.sp.audio_features, 
                track_ids
            )
            
            result = {}
            for track_id, features in zip(track_ids, features_list):
                if features:
                    result[track_id] = {
                        'energy': features['energy'],
                        'danceability': features['danceability'],
                        'valence': features['valence'],
                        'tempo': features['tempo'],
                        'loudness': features['loudness'],
                        'acousticness': features['acousticness'],
                        'instrumentalness': features['instrumentalness'],
                        'speechiness': features['speechiness']
                    }
            
            return result
            
        except Exception as e:
            print(f"Error getting multiple audio features: {e}")
            return {}
    
    # Other methods

    async def get_current_track(self) -> Optional[Dict]:
        """Get currently playing track"""
        loop = asyncio.get_event_loop()
        playback = await loop.run_in_executor(None, self.sp.current_playback)
        
        if not playback or not playback.get('item'):
            return None
        
        track = playback['item']
        return {
            'id': track['id'],
            'title': track['name'],
            'artist': ', '.join([a['name'] for a in track['artists']]),
            'album': track['album']['name'],
            'duration_ms': track['duration_ms'],
            'progress_ms': playback.get('progress_ms', 0),
            'uri': track['uri'],
            'album_art': track['album']['images'][0]['url'] if track['album']['images'] else None
        }
    
    async def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """Get audio features for a track"""
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(None, self.sp.audio_features, track_id)
        
        if not features or not features[0]:
            return None
        
        f = features[0]
        return {
            'energy': f['energy'],
            'danceability': f['danceability'],
            'valence': f['valence'],
            'tempo': f['tempo'],
            'loudness': f['loudness'],
            'acousticness': f['acousticness'],
            'instrumentalness': f['instrumentalness'],
            'speechiness': f['speechiness']
        }
    
    async def _queue_after_delay(self, uri: str, delay: float):
        """
        Internal method to queue track after a delay
        """
        try:
            await asyncio.sleep(delay)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.sp.add_to_queue, uri)
            
            # Clean up task reference
            if uri in self._queue_tasks:
                del self._queue_tasks[uri]
                
        except asyncio.CancelledError:
            print(f"Queue task cancelled for: {uri}")
        except Exception as e:
            print(f"Error in delayed queue: {e}")
    
    async def cancel_scheduled_queue(self, uri: str) -> bool:
        """
        Cancel a scheduled queue operation
        """
        if uri in self._queue_tasks:
            task = self._queue_tasks[uri]
            task.cancel()
            del self._queue_tasks[uri]
            print(f"Cancelled scheduled queue for: {uri}")
            return True
        return False
    
    async def get_scheduled_tracks(self) -> List[str]:
        """
        Get list of track URIs that are scheduled to be queued
        """
        return list(self._queue_tasks.keys())
    
