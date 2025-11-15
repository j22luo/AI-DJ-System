from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncio
import json

from services.spotify_service import SpotifyService
# from services.crowd_vision_service import CrowdVisionService
# from shared.state_manager import state

app = Server("houseparty-dj-mcp")

# Initialize services
spotify_service = SpotifyService()
# crowd_service = CrowdVisionService()

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_party_context",
            description="Get current track, crowd sentiment, and party state",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="suggest_next_tracks",
            description="Suggest next tracks based on current context",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_suggestions": {
                        "type": "integer",
                        "description": "Number of tracks to suggest (default: 3)"
                    }
                }
            }
        ),
        Tool(
            name="queue_track",
            description="Add a track to Spotify queue",
            inputSchema={
                "type": "object",
                "properties": {
                    "track_uri": {"type": "string", "description": "Spotify track URI"}
                },
                "required": ["track_uri"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "get_party_context":
            context = await state.get_state()
            return [TextContent(type="text", text=json.dumps(context, indent=2))]
        
        elif name == "suggest_next_tracks":
            # AI logic for suggesting tracks goes here
            # This would use Spotify API + current context
            suggestions = await generate_suggestions(arguments.get("num_suggestions", 3))
            await state.update_suggestions(suggestions)
            return [TextContent(type="text", text=json.dumps(suggestions, indent=2))]
        
        elif name == "queue_track":
            success = await spotify_service.queue_track(arguments["track_uri"])
            return [TextContent(type="text", text=json.dumps({"success": success}))]
        
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def generate_suggestions(num: int) -> list:
    """Generate track suggestions using AI"""
    # TODO: Implement sophisticated track selection
    # For now, return placeholder
    return []

if __name__ == "__main__":
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    
    asyncio.run(main())