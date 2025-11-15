#!/usr/bin/env python3
"""
Camera Recorder
Captures images from laptop camera at 1-second intervals for 5 seconds.
"""

import time
import os
from datetime import datetime
import cv2
import numpy as np


class CameraRecorder:
    def __init__(self, output_dir="camera_images", duration=5, interval=1, camera_index=None, test_mode=False):
        """
        Initialize the camera recorder.

        Args:
            output_dir (str): Directory to save camera images
            duration (int): Total recording duration in seconds
            interval (int): Time interval between captures in seconds
            camera_index (int): Camera device index (None for auto-detect, 0 for default camera)
            test_mode (bool): If True, use mock camera for testing in headless environments
        """
        self.output_dir = output_dir
        self.duration = duration
        self.interval = interval
        self.camera_index = camera_index
        self.test_mode = test_mode
        self.images = []

    def setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        else:
            print(f"Using existing output directory: {self.output_dir}")

    def detect_camera(self):
        """
        Auto-detect available camera by trying different indices and backends.

        Returns:
            tuple: (camera_object, camera_index) or (None, None) if no camera found
        """
        # List of backends to try (in order of preference for Linux)
        import sys
        if sys.platform.startswith("win"):
            backends = [
                (cv2.CAP_DSHOW, "DSHOW"),
                (cv2.CAP_MSMF, "MSMF"),
                (cv2.CAP_ANY, "ANY"),
            ]
        else:
            backends = [
                (cv2.CAP_V4L2, "V4L2"),
                (cv2.CAP_ANY, "ANY"),
                (cv2.CAP_GSTREAMER, "GSTREAMER"),
            ]


        # Try camera indices 0-5
        for index in range(6):
            for backend, backend_name in backends:
                try:
                    camera = cv2.VideoCapture(index, backend)
                    if camera.isOpened() and self._validate_camera(camera):
                        # Test if we can actually read valid frames from the camera
                        if self._validate_camera(camera):
                            print(f"✓ Found working camera: index={index}, backend={backend_name}")
                            return camera, index
                        camera.release()
                except Exception as e:
                    pass

        return None, None

    def _validate_camera(self, camera):
        """
        Validate that a camera can provide valid frames.

        Args:
            camera: OpenCV VideoCapture object

        Returns:
            bool: True if camera provides valid frames, False otherwise
        """
        # Try reading a few frames to ensure camera is stable
        for attempt in range(3):
            ret, frame = camera.read()

            if not ret:
                return False

            # Validate frame properties
            if frame is None:
                return False

            # Check frame has valid shape
            if not hasattr(frame, 'shape') or len(frame.shape) != 3:
                return False

            # Check frame has reasonable dimensions
            height, width = frame.shape[:2]
            if height < 10 or width < 10 or height > 10000 or width > 10000:
                return False

            # Check frame data type
            if frame.dtype != np.uint8:
                return False

            # Check frame is not all zeros
            if np.sum(frame) == 0:
                continue  # First frame might be black, try next

            # Frame is valid
            return True

        return False

    def create_mock_camera(self):
        """
        Create a mock camera for testing in headless environments.

        Returns:
            MockCamera: A mock camera object that mimics cv2.VideoCapture
        """
        class MockCamera:
            def __init__(self):
                self.frame_count = 0

            def isOpened(self):
                return True

            def read(self):
                # Generate a test image with gradient and text
                self.frame_count += 1
                img = np.zeros((720, 1280, 3), dtype=np.uint8)

                # Create a gradient background
                for i in range(720):
                    color_val = int((i / 720) * 255)
                    img[i, :] = [color_val, 128, 255 - color_val]

                # Add text
                text = f"Mock Camera - Frame {self.frame_count}"
                cv2.putText(img, text, (400, 360), cv2.FONT_HERSHEY_SIMPLEX,
                           1.5, (255, 255, 255), 3)
                cv2.putText(img, "Test Mode Active", (450, 420),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                return True, img

            def set(self, prop, value):
                pass

            def release(self):
                pass

        return MockCamera()

    def capture_image(self, camera, filename):
        """
        Capture an image from the camera.

        Args:
            camera: OpenCV VideoCapture object
            filename (str): Name of the file to save the image

        Returns:
            str: Path to the saved image file
        """
        # Read frame from camera
        ret, frame = camera.read()

        if not ret:
            raise Exception("Failed to capture image from camera")

        # Validate frame before saving
        if frame is None:
            raise Exception("Camera returned None frame")
        
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        if not hasattr(frame, 'shape') or len(frame.shape) != 3:
            raise Exception(f"Invalid frame shape: {frame.shape if hasattr(frame, 'shape') else 'no shape'}")

        height, width = frame.shape[:2]
        if height < 10 or width < 10:
            raise Exception(f"Frame dimensions too small: {width}x{height}")

        # Save the image
        filepath = os.path.join(self.output_dir, filename)
        success = cv2.imwrite(filepath, frame)

        if not success:
            raise Exception(f"Failed to write image to {filepath}")

        return filepath

    def record(self):
        """
        Start recording from the camera.
        Captures images at specified intervals for the specified duration.
        """
        self.setup_output_directory()

        print(f"Initializing camera...")

        # Initialize camera
        camera = None

        if self.test_mode:
            print("Running in TEST MODE (no physical camera required)")
            camera = self.create_mock_camera()
            self.camera_index = -1  # Indicate mock camera
        elif self.camera_index is not None:
            # Try specific camera index with validation
            print(f"Attempting to open camera index {self.camera_index}...")
            import sys
            if sys.platform.startswith("win"):
                camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                camera = cv2.VideoCapture(self.camera_index)
            if camera.isOpened():
                # Validate that camera can provide valid frames
                if not self._validate_camera(camera):
                    print(f"⚠ Camera {self.camera_index} opened but cannot provide valid frames")
                    camera.release()
                    camera = None
            else:
                camera = None
        else:
            # Auto-detect camera
            print("Auto-detecting camera...")
            camera, detected_index = self.detect_camera()
            if camera:
                self.camera_index = detected_index

        # If no camera found, fall back to test mode
        if camera is None or not camera.isOpened():
            print("\n⚠ No physical camera detected!")
            print("Falling back to TEST MODE with mock camera...")
            print("To use a physical camera, ensure it's connected and not in use.\n")
            camera = self.create_mock_camera()
            self.test_mode = True
            self.camera_index = -1

        # Set camera properties for better quality (optional)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Warm up camera by reading and discarding a few frames
        if not self.test_mode:
            print("Warming up camera...")
            for i in range(5):
                ret, _ = camera.read()
                if not ret:
                    print(f"⚠ Warning: Failed to read warmup frame {i+1}/5")
            time.sleep(0.3)

        camera_type = "Mock Camera (Test Mode)" if self.test_mode else f"Camera {self.camera_index}"
        print(f"\n✓ Successfully initialized: {camera_type}")
        print(f"\nStarting camera recording...")
        print(f"Duration: {self.duration} seconds")
        print(f"Interval: {self.interval} second(s)")
        print(f"Expected captures: {self.duration // self.interval}")
        print("-" * 50)

        start_time = time.time()
        capture_count = 0

        try:
            while time.time() - start_time < self.duration:
                # Calculate current elapsed time
                elapsed = time.time() - start_time

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_{capture_count + 1:03d}_{timestamp}.jpg"

                # Capture image
                try:
                    filepath = self.capture_image(camera, filename)
                    capture_count += 1
                    self.images.append(filepath)
                    print(f"[{elapsed:.1f}s] Captured: {filename}")
                except Exception as e:
                    print(f"Error capturing image: {e}")

                # Wait for the next interval
                next_capture_time = start_time + (capture_count * self.interval)
                sleep_time = next_capture_time - time.time()

                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            # Release the camera
            camera.release()

        print("-" * 50)
        print(f"Recording complete!")
        print(f"Total captures: {capture_count}")
        print(f"Images saved to: {os.path.abspath(self.output_dir)}")

        return self.images


def main():
    """Main function to run the camera recorder."""
    import sys

    # Check for command line arguments
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    camera_index = None

    # Parse camera index from command line
    for arg in sys.argv[1:]:
        if arg.startswith("--camera="):
            try:
                camera_index = int(arg.split("=")[1])
            except ValueError:
                print(f"Invalid camera index: {arg}")
                sys.exit(1)

    # Create recorder with default settings: 5 seconds duration, 1 second interval
    recorder = CameraRecorder(
        output_dir="camera_images",
        duration=5,
        interval=1,
        camera_index=camera_index,
        test_mode=test_mode
    )

    # Start recording
    try:
        images = recorder.record()

        # Display captured files
        print("\nCaptured files:")
        for i, filepath in enumerate(images, 1):
            file_size = os.path.getsize(filepath) / 1024  # Size in KB
            print(f"  {i}. {filepath} ({file_size:.1f} KB)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("- Make sure your camera is not being used by another application")
        print("- Check that your camera is properly connected")
        print("- Try running with a specific camera: python camera_recorder.py --camera=0")
        print("- Use test mode for environments without a camera: python camera_recorder.py --test")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
