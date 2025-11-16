from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncio
import json
from config import Config
from services.spotify_service import SpotifyService
from services.audio_analysis import MicDBFFrequencyMonitor
import base64
from services.camera_capture import CameraRecorder
from services.vision_analysis import get_vision_analyzer
from services.advanced_audio_analysis import get_audio_analyzer
from services.party_state_analyzer import get_party_state_analyzer
from services.dj_decision_engine import get_dj_decision_engine
from services.feedback_tracker import get_feedback_tracker
# from shared.state_manager import state

app = Server("houseparty-dj-mcp")

# Initialize services
spotify_service = SpotifyService()

# ---- Microphone monitor setup ----
mic_monitor = MicDBFFrequencyMonitor(
    device_index=None,      # set to an explicit device index once calibrated
    sample_rate=22050,
    block_duration=0.01,    # 10ms blocks (~100 Hz)
    history_seconds=5.0,    # keep last 5 seconds
)
mic_monitor.start()

# ---- Camera Recorder setup ----
camera_recorder = CameraRecorder()
camera_recorder.detect_camera()
camera_recorder.create_mock_camera()

# ---- Vision Analyzer setup ----
vision_analyzer = get_vision_analyzer()

# ---- Advanced Audio Analyzer setup ----
audio_analyzer = get_audio_analyzer(sample_rate=mic_monitor.sample_rate)

# ---- Party State Analyzer setup ----
party_state_analyzer = get_party_state_analyzer()

# ---- DJ Decision Engine setup ----
dj_engine = get_dj_decision_engine()

# ---- Feedback Tracker setup ----
feedback_tracker = get_feedback_tracker()

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_party_sound",
            description=(
                "Get a graph snapshot of the last 5 seconds of microphone audio. "
                "Returns dBFS loudness and FFT-based spectral centroid, plus a "
                "base64-encoded PNG image of the graph."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="take_picture",
            description=(
                "Take a picture from the webcam"
                "Returns a base64-encoded PNG image"
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_current_track_audio_features",
            description="Get detailed audio analysis of the currently playing track - tempo, key, mode, time_signature, loudness, duration_ms, progress_ms, avg_segment_loudness, timbre_profile, num_sections, section_loudness_variation, track_name, artist, popularity, track_id",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="get_track_audio_features",
            description="Get detailed audio analysis of some track using its id - tempo, key, mode, time_signature, loudness, duration_ms, progress_ms, avg_segment_loudness, timbre_profile, num_sections, section_loudness_variation, track_name, artist, popularity, track_id",
            inputSchema={"type": "object", "properties": {
                "track_id": {"type": "string", "description": "Spotify track id"}
            }, "required": ["track_id"]}
        ),
        Tool(
            name="get_test_playlist",
            description="Gives data about a pre-specified playlist - id, name, total_tracks (number of tracks) and a list of tracks with ids ",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        # Tool(
        #     name="get_multiple_audio_features",
        #     description="Given a list of track_ids, get audio features for each track - energy, danceability, valence, tempo, loudness, acousticness, instrumentalness, speechiness",
        #     inputSchema={
        #         "type": "object", 
        #         "properties": {
        #             "track_ids": {
        #                 "type": "array",
        #                 "items": {"type": "string"},
        #                 "description": "List of track IDs to get audio features from",
        #                 "minItems": 0,
        #                 "maxItems": Config.PLAYLIST_MAX_SIZE
        #             },
        #         }, 
        #         "required": ["track_ids"]}
        # ),
        Tool(
            name="play_track_after_delay",
            description="Play a specific track after a given delay in seconds",
            inputSchema={
                "type": "object",
                "properties": {
                    "track_uri": {"type": "string", "description": "Spotify track URI"},
                    "delay": {"type": "number", "description": "Seconds until track starts playing"}
                },
                "required": ["track_uri"]
            }
        ),
        # Tool(
        #     name="cancel_scheduled_play",
        #     description="Given URI of scheduled song, stop it from playing",
        #     inputSchema={
        #         "type": "object",
        #         "properties": {
        #             "uri": {"type": "string", "description": "Spotify track URI"}
        #         },
        #         "required": ["uri"]
        #     }
        # ),
        Tool(
            name="get_scheduled_play_tracks",
            description="Get the list of URIs scheduled tracks that will be played",
            inputSchema={"type": "object","properties": {},"required": []}
        ),
        Tool(
            name="analyze_party_visuals",
            description=(
                "Analyze party visuals from camera for crowd metrics: "
                "estimated people count, crowd density, motion level, activity zones, "
                "overall energy, brightness, color vibrancy, crowd clustering, dance floor occupancy. "
                "Returns comprehensive party visual metrics."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="analyze_party_audio_advanced",
            description=(
                "Advanced audio analysis beyond basic dBFS: "
                "speech/music/cheering probability, crowd noise level, crowd excitement, "
                "frequency band energy (bass/mid/high), beat detection, estimated tempo, "
                "volume trends, and overall audio energy metric."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="get_party_state",
            description=(
                "Get unified party state by correlating ALL sensor data (vision + audio + music). "
                "Returns: overall energy, crowd engagement, party momentum (heating_up/peak/cooling_down), "
                "music-crowd alignment, energy trends, and AI-generated DJ suggestions. "
                "This is the main tool for understanding the current party state and getting recommendations."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="should_switch_track",
            description=(
                "AI decision on whether to switch the current track. "
                "Analyzes party state, track progress, music-crowd alignment, and engagement. "
                "Returns: should_switch (bool), confidence, reason, urgency (immediate/soon/normal), "
                "and suggested_timing (seconds until switch)."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="recommend_next_track",
            description=(
                "Get AI recommendation for the best next track to play. "
                "Analyzes party state and scores available tracks based on energy matching, "
                "tempo compatibility, popularity, and smooth transitions. "
                "Returns: track_id, track_name, artist, confidence, reasoning, target_energy, target_tempo."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "playlist_id": {
                        "type": "string",
                        "description": "Spotify playlist ID to select tracks from (optional, uses test playlist if not provided)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="generate_playlist_plan",
            description=(
                "Generate a multi-track playlist plan with energy arc strategy. "
                "Creates a sequence of tracks designed to build_up, maintain, cool_down, or wave pattern. "
                "Returns: list of tracks, energy_arc values, arc_strategy, estimated_duration_minutes. "
                "Perfect for planning ahead and maintaining party flow."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "num_tracks": {
                        "type": "number",
                        "description": "Number of tracks to plan (default 5)"
                    },
                    "playlist_id": {
                        "type": "string",
                        "description": "Spotify playlist ID to select tracks from (optional, uses test playlist if not provided)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="start_track_feedback",
            description=(
                "Start tracking feedback for a track that's beginning to play. "
                "Records starting party state and track info for later outcome analysis. "
                "Call this when a new track starts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "track_id": {"type": "string", "description": "Spotify track ID"},
                    "track_name": {"type": "string", "description": "Track name"},
                    "artist": {"type": "string", "description": "Artist name"},
                    "was_recommended": {"type": "boolean", "description": "Was this an AI recommendation?"},
                    "recommendation_confidence": {"type": "number", "description": "Confidence of recommendation (0-1)"}
                },
                "required": ["track_id", "track_name", "artist"]
            }
        ),
        Tool(
            name="end_track_feedback",
            description=(
                "End tracking for current track and record its outcome. "
                "Calculates success score based on energy/engagement changes, completion, and crowd reaction. "
                "Call this when a track ends or is skipped."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "completed": {"type": "boolean", "description": "Did track play to completion?"},
                    "skipped": {"type": "boolean", "description": "Was track skipped early?"},
                    "notes": {"type": "string", "description": "Additional notes about outcome"}
                },
                "required": []
            }
        ),
        Tool(
            name="get_session_summary",
            description=(
                "Get summary of current DJ session. "
                "Returns: total tracks played, avg success score, completion/skip counts, "
                "energy changes, and list of tracks with outcomes."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="get_best_performing_tracks",
            description=(
                "Get historically best performing tracks based on past feedback. "
                "Returns top tracks ranked by success score across multiple plays. "
                "Useful for selecting crowd favorites."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "top_n": {"type": "number", "description": "Number of top tracks to return (default 10)"}
                },
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        match name:
            case "get_current_track_audio_features":
                result = await spotify_service.get_current_track_audio_features()
                return [TextContent(type="text", text=json.dumps(result))]
            case "get_track_audio_features":
                result = await spotify_service.get_track_audio_features(arguments.get("track_id"))
                return [TextContent(type="text", text=json.dumps(result))]
            case "get_test_playlist":
                success = await spotify_service.get_test_playlist()
                return [TextContent(type="text", text=json.dumps({"success": success}))]
            case "play_track_after_delay":
                success = await spotify_service.play_track_after_delay(arguments.get("track_uri"), arguments.get("delay"))
                return [TextContent(type="text", text=json.dumps({"success": success}))]
            # case "cancel_scheduled_play":
            #     success = await spotify_service.cancel_scheduled_play(arguments.get("uri"))
            #     return [TextContent(type="text", text=json.dumps({"success": success}))]
            case "get_scheduled_play_tracks":
                result = await spotify_service.get_scheduled_play_tracks()
                return [TextContent(type="text", text=json.dumps({"result" : result}))]
            case "get_party_sound":
                snapshot = mic_monitor.get_graph_snapshot()
                image_bytes = mic_monitor.get_graph_image()

                if snapshot is None or image_bytes is None:
                    payload = {"error": "No audio data yet"}
                else:
                    image_b64 = base64.b64encode(image_bytes).decode("ascii")
                    payload = {
                        "graph": snapshot,  # structured numeric data
                        "image": {
                            "mime_type": "image/png",
                            "base64": image_b64,
                        },
                    }

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(payload, indent=2),
                    )
                ]
            case "take_picture":
                try:
                    # Capture with compression (640x480 max, 70% JPEG quality for context window optimization)
                    image_bytes = camera_recorder._capture_into_raw_bytes(
                        max_width=640,
                        max_height=480,
                        jpeg_quality=70
                    )
                    image_b64 = base64.b64encode(image_bytes).decode("ascii")
                    payload = {
                        "success": True,
                        "mime_type": "image/jpeg",  # Fixed: was incorrectly "image/png"
                        "base64": image_b64,
                        "size_kb": len(image_bytes) / 1024
                    }
                    return [TextContent(type="text", text=json.dumps(payload))]
                except Exception as e:
                    print(f"Error capturing or encoding image: {e}")
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

            case "analyze_party_visuals":
                try:
                    # Capture image
                    image_bytes = camera_recorder._capture_into_raw_bytes(
                        max_width=640,
                        max_height=480,
                        jpeg_quality=70
                    )

                    # Analyze with vision analyzer
                    analysis_result = vision_analyzer.analyze_image(image_bytes)

                    # Convert to dict
                    result_dict = vision_analyzer.to_dict(analysis_result)

                    return [TextContent(type="text", text=json.dumps(result_dict, indent=2))]
                except Exception as e:
                    print(f"Error in vision analysis: {e}")
                    import traceback
                    traceback.print_exc()
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

            case "analyze_party_audio_advanced":
                try:
                    # Get raw audio samples from mic monitor
                    audio_samples = mic_monitor.get_raw_audio_samples()

                    if len(audio_samples) == 0:
                        return [TextContent(type="text", text=json.dumps({"error": "No audio data available yet"}))]

                    # Get current basic metrics
                    recent_data = mic_monitor.get_recent_data()
                    if not recent_data:
                        return [TextContent(type="text", text=json.dumps({"error": "No audio metrics available yet"}))]

                    latest = recent_data[-1]
                    current_dbfs = latest[1]
                    spectral_centroid = latest[2]

                    # Perform advanced analysis
                    analysis_result = audio_analyzer.analyze_audio(
                        audio_samples,
                        current_dbfs,
                        spectral_centroid
                    )

                    # Convert to dict
                    result_dict = audio_analyzer.to_dict(analysis_result)

                    return [TextContent(type="text", text=json.dumps(result_dict, indent=2))]
                except Exception as e:
                    print(f"Error in advanced audio analysis: {e}")
                    import traceback
                    traceback.print_exc()
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

            case "get_party_state":
                try:
                    # Gather all sensor data
                    vision_data = None
                    audio_data = None
                    music_data = None

                    # Get vision analysis
                    try:
                        image_bytes = camera_recorder._capture_into_raw_bytes(
                            max_width=640,
                            max_height=480,
                            jpeg_quality=70
                        )
                        vision_result = vision_analyzer.analyze_image(image_bytes)
                        vision_data = vision_analyzer.to_dict(vision_result)
                    except Exception as e:
                        print(f"Vision analysis failed: {e}")

                    # Get audio analysis
                    try:
                        audio_samples = mic_monitor.get_raw_audio_samples()
                        if len(audio_samples) > 0:
                            recent_data = mic_monitor.get_recent_data()
                            if recent_data:
                                latest = recent_data[-1]
                                audio_result = audio_analyzer.analyze_audio(
                                    audio_samples,
                                    latest[1],
                                    latest[2]
                                )
                                audio_data = audio_analyzer.to_dict(audio_result)
                    except Exception as e:
                        print(f"Audio analysis failed: {e}")

                    # Get music analysis
                    try:
                        music_data = await spotify_service.get_current_track_audio_features()
                    except Exception as e:
                        print(f"Music analysis failed: {e}")

                    # Correlate all data
                    party_state = party_state_analyzer.analyze_party_state(
                        vision_data=vision_data,
                        audio_data=audio_data,
                        music_data=music_data
                    )

                    # Convert to dict
                    result_dict = party_state_analyzer.to_dict(party_state)

                    return [TextContent(type="text", text=json.dumps(result_dict, indent=2))]

                except Exception as e:
                    print(f"Error in party state analysis: {e}")
                    import traceback
                    traceback.print_exc()
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

            case "should_switch_track":
                try:
                    # Get party state
                    vision_data = None
                    audio_data = None
                    music_data = None

                    # Gather sensor data (same as get_party_state)
                    try:
                        image_bytes = camera_recorder._capture_into_raw_bytes(max_width=640, max_height=480, jpeg_quality=70)
                        vision_result = vision_analyzer.analyze_image(image_bytes)
                        vision_data = vision_analyzer.to_dict(vision_result)
                    except Exception as e:
                        print(f"Vision analysis failed: {e}")

                    try:
                        audio_samples = mic_monitor.get_raw_audio_samples()
                        if len(audio_samples) > 0:
                            recent_data = mic_monitor.get_recent_data()
                            if recent_data:
                                latest = recent_data[-1]
                                audio_result = audio_analyzer.analyze_audio(audio_samples, latest[1], latest[2])
                                audio_data = audio_analyzer.to_dict(audio_result)
                    except Exception as e:
                        print(f"Audio analysis failed: {e}")

                    try:
                        music_data = await spotify_service.get_current_track_audio_features()
                    except Exception as e:
                        print(f"Music analysis failed: {e}")

                    # Correlate data
                    party_state = party_state_analyzer.analyze_party_state(vision_data, audio_data, music_data)
                    party_state_dict = party_state_analyzer.to_dict(party_state)

                    # Get current track info
                    track_info = music_data if music_data else {}

                    # Calculate time since track start
                    progress_ms = track_info.get("progress_ms", 0)
                    time_since_start = progress_ms / 1000.0

                    # Make decision
                    decision = dj_engine.should_switch_track(
                        party_state=party_state_dict,
                        track_info=track_info,
                        time_since_track_start=time_since_start
                    )

                    result_dict = dj_engine.to_dict_decision(decision)

                    return [TextContent(type="text", text=json.dumps(result_dict, indent=2))]

                except Exception as e:
                    print(f"Error in should_switch_track: {e}")
                    import traceback
                    traceback.print_exc()
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

            case "recommend_next_track":
                try:
                    # Get party state
                    vision_data = None
                    audio_data = None
                    music_data = None

                    try:
                        image_bytes = camera_recorder._capture_into_raw_bytes(max_width=640, max_height=480, jpeg_quality=70)
                        vision_result = vision_analyzer.analyze_image(image_bytes)
                        vision_data = vision_analyzer.to_dict(vision_result)
                    except: pass

                    try:
                        audio_samples = mic_monitor.get_raw_audio_samples()
                        if len(audio_samples) > 0:
                            recent_data = mic_monitor.get_recent_data()
                            if recent_data:
                                latest = recent_data[-1]
                                audio_result = audio_analyzer.analyze_audio(audio_samples, latest[1], latest[2])
                                audio_data = audio_analyzer.to_dict(audio_result)
                    except: pass

                    try:
                        music_data = await spotify_service.get_current_track_audio_features()
                    except: pass

                    party_state = party_state_analyzer.analyze_party_state(vision_data, audio_data, music_data)
                    party_state_dict = party_state_analyzer.to_dict(party_state)

                    # Get available tracks from playlist
                    playlist_id = arguments.get("playlist_id")
                    available_tracks = await spotify_service.get_playlist_with_audio_features(playlist_id)

                    if not available_tracks:
                        return [TextContent(type="text", text=json.dumps({"error": "No tracks available in playlist"}))]

                    # Get current track for smooth transitions
                    current_track = music_data if music_data else None

                    # Get recommendation
                    recommendation = dj_engine.recommend_next_track(
                        party_state=party_state_dict,
                        available_tracks=available_tracks,
                        current_track=current_track
                    )

                    result_dict = dj_engine.to_dict_recommendation(recommendation)

                    return [TextContent(type="text", text=json.dumps(result_dict, indent=2))]

                except Exception as e:
                    print(f"Error in recommend_next_track: {e}")
                    import traceback
                    traceback.print_exc()
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

            case "generate_playlist_plan":
                try:
                    # Get party state
                    vision_data = None
                    audio_data = None
                    music_data = None

                    try:
                        image_bytes = camera_recorder._capture_into_raw_bytes(max_width=640, max_height=480, jpeg_quality=70)
                        vision_result = vision_analyzer.analyze_image(image_bytes)
                        vision_data = vision_analyzer.to_dict(vision_result)
                    except: pass

                    try:
                        audio_samples = mic_monitor.get_raw_audio_samples()
                        if len(audio_samples) > 0:
                            recent_data = mic_monitor.get_recent_data()
                            if recent_data:
                                latest = recent_data[-1]
                                audio_result = audio_analyzer.analyze_audio(audio_samples, latest[1], latest[2])
                                audio_data = audio_analyzer.to_dict(audio_result)
                    except: pass

                    try:
                        music_data = await spotify_service.get_current_track_audio_features()
                    except: pass

                    party_state = party_state_analyzer.analyze_party_state(vision_data, audio_data, music_data)
                    party_state_dict = party_state_analyzer.to_dict(party_state)

                    # Get available tracks
                    playlist_id = arguments.get("playlist_id")
                    available_tracks = await spotify_service.get_playlist_with_audio_features(playlist_id)

                    if not available_tracks:
                        return [TextContent(type="text", text=json.dumps({"error": "No tracks available in playlist"}))]

                    # Get parameters
                    num_tracks = arguments.get("num_tracks", 5)

                    # Generate playlist plan
                    plan = dj_engine.generate_playlist_plan(
                        party_state=party_state_dict,
                        available_tracks=available_tracks,
                        num_tracks=int(num_tracks)
                    )

                    result_dict = dj_engine.to_dict_playlist_plan(plan)

                    return [TextContent(type="text", text=json.dumps(result_dict, indent=2))]

                except Exception as e:
                    print(f"Error in generate_playlist_plan: {e}")
                    import traceback
                    traceback.print_exc()
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

            case "start_track_feedback":
                try:
                    # Get current party state
                    vision_data = None
                    audio_data = None
                    music_data = None

                    try:
                        image_bytes = camera_recorder._capture_into_raw_bytes(max_width=640, max_height=480, jpeg_quality=70)
                        vision_result = vision_analyzer.analyze_image(image_bytes)
                        vision_data = vision_analyzer.to_dict(vision_result)
                    except: pass

                    try:
                        audio_samples = mic_monitor.get_raw_audio_samples()
                        if len(audio_samples) > 0:
                            recent_data = mic_monitor.get_recent_data()
                            if recent_data:
                                latest = recent_data[-1]
                                audio_result = audio_analyzer.analyze_audio(audio_samples, latest[1], latest[2])
                                audio_data = audio_analyzer.to_dict(audio_result)
                    except: pass

                    try:
                        music_data = await spotify_service.get_current_track_audio_features()
                    except: pass

                    party_state = party_state_analyzer.analyze_party_state(vision_data, audio_data, music_data)
                    party_state_dict = party_state_analyzer.to_dict(party_state)

                    # Start tracking
                    feedback_tracker.start_track(
                        track_id=arguments["track_id"],
                        track_name=arguments["track_name"],
                        artist=arguments["artist"],
                        party_state=party_state_dict,
                        was_recommended=arguments.get("was_recommended", False),
                        recommendation_confidence=arguments.get("recommendation_confidence")
                    )

                    return [TextContent(type="text", text=json.dumps({"success": True, "message": "Track feedback started"}))]

                except Exception as e:
                    print(f"Error in start_track_feedback: {e}")
                    import traceback
                    traceback.print_exc()
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

            case "end_track_feedback":
                try:
                    # Get current party state
                    vision_data = None
                    audio_data = None
                    music_data = None

                    try:
                        image_bytes = camera_recorder._capture_into_raw_bytes(max_width=640, max_height=480, jpeg_quality=70)
                        vision_result = vision_analyzer.analyze_image(image_bytes)
                        vision_data = vision_analyzer.to_dict(vision_result)
                    except: pass

                    try:
                        audio_samples = mic_monitor.get_raw_audio_samples()
                        if len(audio_samples) > 0:
                            recent_data = mic_monitor.get_recent_data()
                            if recent_data:
                                latest = recent_data[-1]
                                audio_result = audio_analyzer.analyze_audio(audio_samples, latest[1], latest[2])
                                audio_data = audio_analyzer.to_dict(audio_result)
                    except: pass

                    try:
                        music_data = await spotify_service.get_current_track_audio_features()
                    except: pass

                    party_state = party_state_analyzer.analyze_party_state(vision_data, audio_data, music_data)
                    party_state_dict = party_state_analyzer.to_dict(party_state)

                    # End tracking
                    feedback_tracker.end_track(
                        party_state=party_state_dict,
                        completed=arguments.get("completed", True),
                        skipped=arguments.get("skipped", False),
                        notes=arguments.get("notes", "")
                    )

                    return [TextContent(type="text", text=json.dumps({"success": True, "message": "Track feedback ended and recorded"}))]

                except Exception as e:
                    print(f"Error in end_track_feedback: {e}")
                    import traceback
                    traceback.print_exc()
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

            case "get_session_summary":
                try:
                    summary = feedback_tracker.get_session_summary()
                    return [TextContent(type="text", text=json.dumps(summary, indent=2))]
                except Exception as e:
                    print(f"Error in get_session_summary: {e}")
                    import traceback
                    traceback.print_exc()
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

            case "get_best_performing_tracks":
                try:
                    top_n = arguments.get("top_n", 10)
                    best_tracks = feedback_tracker.get_best_performing_tracks(top_n=int(top_n))
                    return [TextContent(type="text", text=json.dumps({"best_tracks": best_tracks}, indent=2))]
                except Exception as e:
                    print(f"Error in get_best_performing_tracks: {e}")
                    import traceback
                    traceback.print_exc()
                    return [TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]


        # if name == "get_party_context":
        #     context = await state.get_state()
        #     return [TextContent(type="text", text=json.dumps(context, indent=2))]
        
        # elif name == "suggest_next_tracks":
        #     # AI logic for suggesting tracks goes here
        #     # This would use Spotify API + current context
        #     suggestions = await generate_suggestions(arguments.get("num_suggestions", 3))
        #     await state.update_suggestions(suggestions)
        #     return [TextContent(type="text", text=json.dumps(suggestions, indent=2))]
        
        # elif name == "queue_track":
        #     success = await spotify_service.queue_track(arguments["track_uri"])
        #     return [TextContent(type="text", text=json.dumps({"success": success}))]
        
        # else:
        #     return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


if __name__ == "__main__":
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    
    asyncio.run(main())
