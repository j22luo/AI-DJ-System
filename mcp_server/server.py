from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncio
import json
from config import Config
from services.spotify_service import SpotifyService
from services.audio_analysis import MicDBFFrequencyMonitor
import base64
from services.camera_capture import CameraRecorder
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
                    image_bytes = camera_recorder._capture_into_raw_bytes()
                    image_b64 = base64.b64encode(image_bytes).decode("ascii")
                    payload = {
                        "success": True, # structured numeric data
                        "mime_type": "image/png",
                        "base64": image_b64,
                    }
                    return [TextContent(type="text", text=json.dumps(payload))]
                except Exception as e:
                    print(f"Error capturing or encoding image: {e}")
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
