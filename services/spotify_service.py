import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from typing import Optional, List, Dict
import asyncio
from config import Config
from pathlib import Path

class SpotifyService():
    """API Wrapper for interacting with Spotify"""
    def __init__(self):
        # Get absolute path to project root
        project_root = Path(__file__).parent.parent
        cache_path = project_root / ".spotify_cache"
        
        # Check if token cache exists
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Spotify token cache not found at: {cache_path}\n"
                "Please run 'python authenticate_spotify.py' first to authenticate."
            )
        
        # User-authenticated client for playback and user data
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=Config.SPOTIFY_CLIENT_ID,
            client_secret=Config.SPOTIFY_CLIENT_SECRET,
            redirect_uri=Config.SPOTIFY_REDIRECT_URI,
            scope="user-library-read user-read-playback-state user-modify-playback-state user-read-currently-playing playlist-read-private playlist-read-collaborative",
            cache_path=str(cache_path),  # Use absolute path
            open_browser=False
        ))
        
        # Validate token works
        try:
            self.sp.current_user()
            print(f"✅ Authenticated with Spotify (cache: {cache_path})")
        except Exception as e:
            raise ConnectionError(
                f"Spotify authentication failed: {e}\n"
                f"Cache location: {cache_path}\n"
                "Your token may have expired. Run 'python authenticate_spotify.py' again."
            )
        self._play_tasks = {}

    # Claude Tools

    async def get_current_track_audio_features(self) -> Optional[Dict]:
        loop = asyncio.get_event_loop()
        playback = await loop.run_in_executor(None, self.sp.current_playback)

        if not playback or not playback.get('item'):
            return None
        
        track = playback['item']
        track_id = track['id']


        try:
            analysis = await loop.run_in_executor(None, self.sp.audio_analysis, track_id)

            # Extract key DJ-relevant metadata for party/mixing context
            track_data = analysis['track']
            sections = analysis.get('sections', [])
            segments = analysis.get('segments', [])
            
            # Calculate average segment characteristics for overall vibe
            avg_loudness = sum(s.get('loudness_max', 0) for s in segments[:20]) / min(len(segments), 20) if segments else 0
            avg_timbre = [sum(s.get('timbre', [0]*12)[i] for s in segments[:20]) / min(len(segments), 20) for i in range(12)] if segments else [0]*12
            
            # Get section types for track structure understanding
            section_types = [s.get('loudness', 0) for s in sections] if sections else []
            
            return {
                'tempo': track_data.get('tempo'),
                'key': track_data.get('key'),  # 0-11 (C, C#, D, etc.)
                'mode': track_data.get('mode'),  # 0=minor, 1=major
                'time_signature': track_data.get('time_signature'),
                'loudness': track_data.get('loudness'),
                'duration_ms': int(track_data.get('duration') * 1000) if track_data.get('duration') else track['duration_ms'],
                'progress_ms': playback.get('progress_ms', 0),
                
                # Party/DJ relevant characteristics
                'avg_segment_loudness': avg_loudness,  # Energy level throughout song
                'timbre_profile': avg_timbre[:4],  # First 4 timbre coefficients (brightness, fullness, etc.)
                'num_sections': len(sections),  # Track complexity/structure
                'section_loudness_variation': max(section_types) - min(section_types) if section_types else 0,  # Dynamic range
                
                # Basic track info for context
                'track_name': track['name'],
                'artist': ', '.join([a['name'] for a in track['artists']]),
                'popularity': track.get('popularity', 0),
                'track_id': track_id
            }
            
        except Exception as e:
            print(f"⚠️  Audio analysis failed: {e}")
            # Fallback to basic track info
            return {
                'error': 'audio_analysis_failed',
                'duration_ms': track['duration_ms'],
                'progress_ms': playback.get('progress_ms', 0),
                'track_name': track['name'],
                'artist': ', '.join([a['name'] for a in track['artists']]),
                'popularity': track.get('popularity', 0),
                'track_id': track_id
            }
        
    async def get_track_audio_features(self, track_id: str) -> Optional[Dict]:
        loop = asyncio.get_event_loop()
        try:
            analysis = await loop.run_in_executor(None, self.sp.audio_analysis, track_id)

            # Extract key DJ-relevant metadata for party/mixing context
            track_data = analysis['track']
            sections = analysis.get('sections', [])
            segments = analysis.get('segments', [])
            
            # Calculate average segment characteristics for overall vibe
            avg_loudness = sum(s.get('loudness_max', 0) for s in segments[:20]) / min(len(segments), 20) if segments else 0
            avg_timbre = [sum(s.get('timbre', [0]*12)[i] for s in segments[:20]) / min(len(segments), 20) for i in range(12)] if segments else [0]*12
            
            # Get section types for track structure understanding
            section_types = [s.get('loudness', 0) for s in sections] if sections else []
            
            return {
                'tempo': track_data.get('tempo'),
                'key': track_data.get('key'),  # 0-11 (C, C#, D, etc.)
                'mode': track_data.get('mode'),  # 0=minor, 1=major
                'time_signature': track_data.get('time_signature'),
                'loudness': track_data.get('loudness'),
                'duration_ms': int(track_data.get('duration') * 1000) if track_data.get('duration') else track['duration_ms'],
                'progress_ms': playback.get('progress_ms', 0),
                
                # Party/DJ relevant characteristics
                'avg_segment_loudness': avg_loudness,  # Energy level throughout song
                'timbre_profile': avg_timbre[:4],  # First 4 timbre coefficients (brightness, fullness, etc.)
                'num_sections': len(sections),  # Track complexity/structure
                'section_loudness_variation': max(section_types) - min(section_types) if section_types else 0,  # Dynamic range
                
                # Basic track info for context
                'track_name': track['name'],
                'artist': ', '.join([a['name'] for a in track['artists']]),
                'popularity': track.get('popularity', 0),
                'track_id': track_id
            }
            
        except Exception as e:
            print(f"⚠️  Audio analysis failed: {e}")
            # Fallback to basic track info
            return {
                'error': 'audio_analysis_failed',
                'duration_ms': track['duration_ms'],
                'progress_ms': playback.get('progress_ms', 0),
                'track_name': track['name'],
                'artist': ', '.join([a['name'] for a in track['artists']]),
                'popularity': track.get('popularity', 0),
                'track_id': track_id
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

    async def get_playlist_with_audio_features(self, playlist_id: Optional[str] = None) -> List[Dict]:
        """
        Get playlist tracks with audio features for DJ decision making.

        Args:
            playlist_id: Spotify playlist ID (uses test playlist if None)

        Returns:
            List of track dicts with audio features
        """
        try:
            if playlist_id is None:
                playlist_id = Config.SPOTIFY_TEST_PLAYLIST_ID

            loop = asyncio.get_event_loop()

            # Get playlist tracks
            tracks_data = await loop.run_in_executor(
                None,
                self.sp.playlist_tracks,
                playlist_id
            )

            track_ids = []
            track_info = {}

            for item in tracks_data['items']:
                if item['track']:
                    track = item['track']
                    track_id = track['id']
                    track_ids.append(track_id)
                    track_info[track_id] = {
                        'id': track_id,
                        'name': track['name'],
                        'artist': ', '.join([a['name'] for a in track['artists']]),
                        'uri': track['uri'],
                        'duration_ms': track['duration_ms'],
                        'popularity': track.get('popularity', 0)
                    }

            # Get audio features for all tracks (batch request)
            if track_ids:
                audio_features = await loop.run_in_executor(
                    None,
                    self.sp.audio_features,
                    track_ids
                )

                # Get audio analysis for tracks (sample a few for performance)
                # We'll get full analysis for first 10 tracks only
                sample_ids = track_ids[:10]

                enriched_tracks = []
                for i, track_id in enumerate(track_ids):
                    track_data = track_info[track_id].copy()

                    # Add audio features
                    if audio_features and i < len(audio_features) and audio_features[i]:
                        af = audio_features[i]
                        track_data['tempo'] = af.get('tempo', 120)
                        track_data['key'] = af.get('key', 0)
                        track_data['mode'] = af.get('mode', 1)
                        track_data['loudness'] = af.get('loudness', -10)
                        track_data['energy'] = af.get('energy', 0.5)
                        track_data['danceability'] = af.get('danceability', 0.5)
                        track_data['valence'] = af.get('valence', 0.5)

                    # Add simplified analysis data for first 10 tracks
                    if track_id in sample_ids:
                        try:
                            analysis = await loop.run_in_executor(
                                None,
                                self.sp.audio_analysis,
                                track_id
                            )

                            # Extract avg segment loudness
                            segments = analysis.get('segments', [])
                            if segments:
                                avg_loudness = sum(s.get('loudness_max', -20) for s in segments) / len(segments)
                                track_data['avg_segment_loudness'] = avg_loudness
                        except:
                            pass  # Skip if analysis fails

                    enriched_tracks.append(track_data)

                return enriched_tracks

            return []

        except Exception as e:
            print(f"Error fetching playlist with audio features: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def play_track_after_delay(self, uri: str, delay: float = 0.0) -> bool:
        """
        Plays a specific track after a given offset in seconds.
        Schedules the playback as a background task.
        """
        try:
            # Create a background task to handle the delay and playback
            task = asyncio.create_task(self._play_after_delay(uri, delay))
            self._play_tasks[uri] = task
            return True
            
        except Exception as e:
            print(f"Error scheduling track playback: {e}")
            return False

    async def _play_after_delay(self, uri: str, delay: float):
        """
        Internal method to wait for the delay and then start playback.
        """
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            
            loop = asyncio.get_event_loop()
            playback = await loop.run_in_executor(None, self.sp.current_playback)
            device_id = playback['device']['id']
            await loop.run_in_executor(
                None, 
                lambda: self.sp.start_playback(uris=[uri], device_id=device_id)
            )
            
            print(f"Started playback of: {uri}")
            
        except asyncio.CancelledError:
            print(f"Playback task cancelled for: {uri}")
        except Exception as e:
            print(f"Error in delayed playback: {e}")
        finally:
            # Once the task is done (played or cancelled), remove it from the dict
            if uri in self._play_tasks:
                del self._play_tasks[uri]

    # async def get_multiple_audio_features(self, track_ids: List[str]) -> Dict[str, Dict]:
    #     """
    #     Get audio features for multiple tracks efficiently
    #     """
    #     try:
    #         # Spotify API allows batch fetching up to 100 tracks
    #         loop = asyncio.get_event_loop()
    #         features_list = await loop.run_in_executor(
    #             None, 
    #             self.sp_public.audio_features, 
    #             track_ids
    #         )
    #         
    #         result = {}
    #         for track_id, features in zip(track_ids, features_list):
    #             if features:
    #                 result[track_id] = {
    #                     'energy': features['energy'],
    #                     'danceability': features['danceability'],
    #                     'valence': features['valence'],
    #                     'tempo': features['tempo'],
    #                     'loudness': features['loudness'],
    #                     'acousticness': features['acousticness'],
    #                     'instrumentalness': features['instrumentalness'],
    #                     'speechiness': features['speechiness']
    #                 }
    #         
    #         return result
    #         
    #     except Exception as e:
    #         print(f"Error getting multiple audio features: {e}")
    #         return {}
    async def cancel_scheduled_play(self, uri: str) -> bool:
        """
        Cancel a scheduled play operation
        """
        if uri in self._play_tasks:
            task = self._play_tasks[uri]
            task.cancel()
            del self._play_tasks[uri]
            # The 'finally' block in _play_after_delay will handle deletion
            print(f"Cancelled scheduled play for: {uri}")
            return True
        return False
    
    async def get_scheduled_play_tracks(self) -> List[str]:
        """
        Get list of track URIs that are scheduled to be played
        """
        return list(self._play_tasks.keys())
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
    
