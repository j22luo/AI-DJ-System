from camera_capture import CameraRecorder
class VideoService():
    """API Wrapper to take a picture using webcam"""
    def __init__(self):
        self.cr = CameraRecorder(duration=1)
        self.cr.detect_camera

    async def get_picture_encoding(self):
        loop = asyncio.get_event_loop()