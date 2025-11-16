"""
Vision Analysis Service
Analyzes camera images to extract party metrics: crowd density, movement, energy levels, etc.
"""

import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class VisionAnalysisResult:
    """Results from analyzing a party scene image."""
    timestamp: str

    # Crowd metrics
    estimated_people_count: int
    crowd_density: float  # 0.0 to 1.0

    # Movement/Activity metrics
    motion_level: float  # 0.0 (still) to 1.0 (high movement)
    activity_zones: List[Dict]  # Areas of activity in the frame

    # Energy indicators
    overall_energy: float  # 0.0 to 1.0 derived from movement + crowd

    # Visual features
    brightness: float  # Average brightness 0.0 to 1.0
    color_vibrancy: float  # Color saturation metric 0.0 to 1.0

    # Spatial distribution
    crowd_clustering: float  # 0.0 (dispersed) to 1.0 (clustered)
    dance_floor_occupancy: float  # Estimated center area occupancy

    # Raw data for correlation
    raw_metrics: Dict


class VisionAnalyzer:
    """Analyzes camera images for party metrics."""

    def __init__(self):
        """Initialize the vision analyzer."""
        self.previous_frame = None
        self.previous_gray = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )

        # For optical flow motion detection
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Feature detection params
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

    def analyze_image(self, image_bytes: bytes) -> VisionAnalysisResult:
        """
        Analyze a single image for party metrics.

        Args:
            image_bytes: JPEG/PNG image bytes

        Returns:
            VisionAnalysisResult with all analyzed metrics
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image")

        # Convert to grayscale for many operations
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform all analysis
        people_count = self._estimate_people_count(frame, gray)
        crowd_density = self._calculate_crowd_density(frame, gray)
        motion_level = self._analyze_motion(gray)
        activity_zones = self._detect_activity_zones(frame, gray)
        brightness = self._calculate_brightness(gray)
        color_vibrancy = self._calculate_color_vibrancy(frame)
        crowd_clustering = self._analyze_clustering(gray)
        dance_floor_occupancy = self._estimate_dance_floor_occupancy(gray)

        # Calculate overall energy as weighted combination
        overall_energy = self._calculate_overall_energy(
            motion_level=motion_level,
            crowd_density=crowd_density,
            activity_zones=len(activity_zones)
        )

        # Store current frame for next motion analysis
        self.previous_frame = frame.copy()
        self.previous_gray = gray.copy()

        # Compile raw metrics for debugging/correlation
        raw_metrics = {
            "frame_shape": frame.shape,
            "motion_level": motion_level,
            "brightness": brightness,
            "color_vibrancy": color_vibrancy,
            "activity_zone_count": len(activity_zones)
        }

        return VisionAnalysisResult(
            timestamp=datetime.now().isoformat(),
            estimated_people_count=people_count,
            crowd_density=crowd_density,
            motion_level=motion_level,
            activity_zones=activity_zones,
            overall_energy=overall_energy,
            brightness=brightness,
            color_vibrancy=color_vibrancy,
            crowd_clustering=crowd_clustering,
            dance_floor_occupancy=dance_floor_occupancy,
            raw_metrics=raw_metrics
        )

    def _estimate_people_count(self, frame: np.ndarray, gray: np.ndarray) -> int:
        """
        Estimate number of people in frame using multiple detection methods.

        Uses a combination of:
        - HOG person detection (if available)
        - Contour-based blob detection
        - Edge density analysis
        """
        height, width = frame.shape[:2]

        # Method 1: Use HOG detector for person detection (CPU-based)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        try:
            # Detect people (returns bounding boxes)
            boxes, weights = hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
                hitThreshold=0
            )
            hog_count = len(boxes)
        except Exception:
            hog_count = 0

        # Method 2: Blob detection via background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size (assume person is at least this many pixels)
        min_area = (height * width) * 0.01  # 1% of frame
        max_area = (height * width) * 0.4   # 40% of frame (avoid full-frame noise)

        valid_blobs = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        blob_count = len(valid_blobs)

        # Method 3: Edge density heuristic (backup)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        # Rough heuristic: more edges = more people complexity
        edge_based_count = int(edge_density * 30)  # Calibrated multiplier

        # Combine methods with weights (HOG is most reliable if it works)
        if hog_count > 0:
            estimated_count = int(hog_count * 0.7 + blob_count * 0.2 + edge_based_count * 0.1)
        else:
            estimated_count = int(blob_count * 0.6 + edge_based_count * 0.4)

        # Clamp to reasonable range
        return max(0, min(estimated_count, 50))  # Max 50 people detectable

    def _calculate_crowd_density(self, frame: np.ndarray, gray: np.ndarray) -> float:
        """
        Calculate crowd density as a 0.0-1.0 metric.

        Based on:
        - Foreground pixel ratio
        - Feature point density
        - Visual complexity
        """
        height, width = frame.shape[:2]
        total_pixels = height * width

        # Foreground mask
        fg_mask = self.background_subtractor.apply(frame)
        fg_pixels = np.sum(fg_mask > 0)
        fg_ratio = fg_pixels / total_pixels

        # Feature point density (FAST corners)
        fast = cv2.FastFeatureDetector_create(threshold=20)
        keypoints = fast.detect(gray, None)
        feature_density = len(keypoints) / 1000  # Normalize (assume max ~1000 features)

        # Combine metrics
        density = (fg_ratio * 0.6 + min(feature_density, 1.0) * 0.4)

        return min(density, 1.0)

    def _analyze_motion(self, gray: np.ndarray) -> float:
        """
        Analyze motion level between current and previous frame.

        Returns:
            float: 0.0 (no motion) to 1.0 (high motion)
        """
        if self.previous_gray is None:
            # First frame, no motion data yet
            return 0.0

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Calculate magnitude of flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Average motion magnitude
        avg_motion = np.mean(magnitude)

        # Normalize (empirically, values typically range 0-10)
        motion_level = min(avg_motion / 5.0, 1.0)

        return motion_level

    def _detect_activity_zones(self, frame: np.ndarray, gray: np.ndarray) -> List[Dict]:
        """
        Detect zones of high activity in the frame.

        Returns:
            List of activity zones with positions and intensity
        """
        if self.previous_gray is None:
            return []

        # Calculate frame difference
        diff = cv2.absdiff(self.previous_gray, gray)

        # Threshold to get motion regions
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Dilate to connect nearby motion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours of motion regions
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert to activity zones
        height, width = frame.shape[:2]
        zones = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (height * width * 0.02):  # At least 2% of frame
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate intensity (how much motion in this zone)
                zone_motion = diff[y:y+h, x:x+w]
                intensity = np.mean(zone_motion) / 255.0

                zones.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "intensity": float(intensity),
                    "area_percent": float(area / (height * width))
                })

        # Sort by intensity
        zones.sort(key=lambda z: z["intensity"], reverse=True)

        # Return top 5 zones
        return zones[:5]

    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Calculate average brightness of the scene."""
        avg_brightness = np.mean(gray) / 255.0
        return avg_brightness

    def _calculate_color_vibrancy(self, frame: np.ndarray) -> float:
        """
        Calculate color vibrancy/saturation.

        Higher values indicate more colorful scene (e.g., party lights).
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Extract saturation channel
        saturation = hsv[:, :, 1]

        # Average saturation
        avg_saturation = np.mean(saturation) / 255.0

        return avg_saturation

    def _analyze_clustering(self, gray: np.ndarray) -> float:
        """
        Analyze how clustered vs dispersed the crowd is.

        Returns:
            float: 0.0 (dispersed) to 1.0 (highly clustered)
        """
        # Apply adaptive threshold to segment people/objects
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            return 0.0

        # Calculate centroids of largest contours
        centroids = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        if len(centroids) < 2:
            return 0.0

        # Calculate average distance between centroids
        distances = []
        for i, c1 in enumerate(centroids):
            for c2 in centroids[i+1:]:
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                distances.append(dist)

        avg_distance = np.mean(distances)

        # Normalize by frame diagonal
        height, width = gray.shape
        frame_diagonal = np.sqrt(height**2 + width**2)

        # Invert: closer centroids = more clustering
        clustering = 1.0 - min(avg_distance / frame_diagonal, 1.0)

        return clustering

    def _estimate_dance_floor_occupancy(self, gray: np.ndarray) -> float:
        """
        Estimate occupancy of central 'dance floor' area.

        Assumes center 50% of frame is the main activity area.
        """
        height, width = gray.shape

        # Define center region (middle 50% of frame)
        center_y1 = height // 4
        center_y2 = 3 * height // 4
        center_x1 = width // 4
        center_x2 = 3 * width // 4

        center_region = gray[center_y1:center_y2, center_x1:center_x2]

        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(center_region)

        # Calculate foreground ratio
        fg_pixels = np.sum(fg_mask > 0)
        total_pixels = center_region.size

        occupancy = fg_pixels / total_pixels

        return min(occupancy, 1.0)

    def _calculate_overall_energy(
        self,
        motion_level: float,
        crowd_density: float,
        activity_zones: int
    ) -> float:
        """
        Calculate overall party energy from component metrics.

        Energy combines:
        - Motion (40%) - how much movement
        - Crowd density (30%) - how many people
        - Activity zones (30%) - how distributed the activity is
        """
        # Normalize activity zones (assume max 5 zones)
        zone_factor = min(activity_zones / 5.0, 1.0)

        energy = (
            motion_level * 0.4 +
            crowd_density * 0.3 +
            zone_factor * 0.3
        )

        return min(energy, 1.0)

    def to_dict(self, result: VisionAnalysisResult) -> Dict:
        """Convert result to JSON-serializable dict."""
        return {
            "timestamp": result.timestamp,
            "crowd": {
                "estimated_people_count": result.estimated_people_count,
                "density": result.crowd_density,
                "clustering": result.crowd_clustering,
                "dance_floor_occupancy": result.dance_floor_occupancy
            },
            "movement": {
                "motion_level": result.motion_level,
                "activity_zones": result.activity_zones,
            },
            "energy": {
                "overall_energy": result.overall_energy,
            },
            "visual": {
                "brightness": result.brightness,
                "color_vibrancy": result.color_vibrancy,
            },
            "raw_metrics": result.raw_metrics
        }


# Singleton instance
_vision_analyzer = None

def get_vision_analyzer() -> VisionAnalyzer:
    """Get or create the singleton vision analyzer instance."""
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer()
    return _vision_analyzer
