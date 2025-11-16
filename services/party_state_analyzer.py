"""
Party State Analyzer
Correlates all sensor data (vision, audio, music) into a unified party state assessment.
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque


@dataclass
class PartyState:
    """Unified party state derived from all sensors."""
    timestamp: str

    # Overall metrics (0.0 to 1.0)
    overall_energy: float  # Combined energy from all sources
    crowd_engagement: float  # How engaged/active is the crowd
    party_momentum: str  # "heating_up", "peak", "cooling_down", "stable"

    # Component energies
    visual_energy: float  # From vision analysis
    audio_energy: float  # From audio analysis
    music_energy: float  # From track features

    # Alignment metrics
    music_crowd_alignment: float  # 0.0 to 1.0 - is music matching crowd?
    energy_variance: float  # How much do different sensors disagree?

    # Crowd state
    estimated_people: int
    crowd_density: float
    crowd_excitement: float  # From audio cheering + visual movement

    # Music state
    current_track_name: Optional[str]
    current_track_tempo: Optional[float]
    track_progress_pct: Optional[float]

    # Trends (compared to recent history)
    energy_trend: str  # "increasing", "decreasing", "stable"
    crowd_trend: str  # "growing", "shrinking", "stable"

    # Recommendations
    recommendation_confidence: float  # 0.0 to 1.0
    suggestions: List[str]  # Human-readable suggestions

    # Raw component data for debugging
    raw_vision: Optional[Dict]
    raw_audio: Optional[Dict]
    raw_music: Optional[Dict]


class PartyStateAnalyzer:
    """Analyzes and correlates all party data into unified state."""

    def __init__(self, history_size: int = 50):
        """
        Initialize the party state analyzer.

        Args:
            history_size: Number of historical states to keep for trend analysis
        """
        self.history_size = history_size

        # Historical data for trend analysis
        self.energy_history = deque(maxlen=history_size)
        self.crowd_history = deque(maxlen=history_size)
        self.state_history = deque(maxlen=history_size)

        # Party session tracking
        self.session_start_time = datetime.now()
        self.total_tracks_played = 0

    def analyze_party_state(
        self,
        vision_data: Optional[Dict] = None,
        audio_data: Optional[Dict] = None,
        music_data: Optional[Dict] = None
    ) -> PartyState:
        """
        Correlate all sensor data into unified party state.

        Args:
            vision_data: Output from vision analyzer (analyze_party_visuals)
            audio_data: Output from advanced audio analyzer (analyze_party_audio_advanced)
            music_data: Output from Spotify track analysis (get_current_track_audio_features)

        Returns:
            PartyState with all correlated metrics
        """
        # Extract component energies
        visual_energy = self._extract_visual_energy(vision_data)
        audio_energy = self._extract_audio_energy(audio_data)
        music_energy = self._extract_music_energy(music_data)

        # Calculate overall energy (weighted combination)
        overall_energy = self._calculate_overall_energy(
            visual_energy, audio_energy, music_energy
        )

        # Store in history
        self.energy_history.append(overall_energy)

        # Extract crowd metrics
        estimated_people = vision_data.get("crowd", {}).get("estimated_people_count", 0) if vision_data else 0
        crowd_density = vision_data.get("crowd", {}).get("density", 0.0) if vision_data else 0.0
        audio_excitement = audio_data.get("crowd", {}).get("excitement", 0.0) if audio_data else 0.0
        visual_motion = vision_data.get("movement", {}).get("motion_level", 0.0) if vision_data else 0.0

        crowd_excitement = (audio_excitement * 0.6 + visual_motion * 0.4)

        self.crowd_history.append(estimated_people)

        # Calculate engagement
        crowd_engagement = self._calculate_crowd_engagement(
            crowd_density, crowd_excitement, visual_motion
        )

        # Analyze trends
        energy_trend = self._analyze_trend(self.energy_history, "energy")
        crowd_trend = self._analyze_trend(self.crowd_history, "crowd")

        # Determine party momentum
        party_momentum = self._determine_momentum(
            overall_energy, energy_trend, crowd_engagement
        )

        # Calculate music-crowd alignment
        music_crowd_alignment = self._calculate_alignment(
            visual_energy, audio_energy, music_energy
        )

        # Calculate energy variance (disagreement between sensors)
        energy_variance = self._calculate_variance(
            visual_energy, audio_energy, music_energy
        )

        # Extract music info
        current_track_name = None
        current_track_tempo = None
        track_progress_pct = None

        if music_data:
            current_track_name = music_data.get("track_name")
            current_track_tempo = music_data.get("tempo")
            duration = music_data.get("duration_ms", 0)
            progress = music_data.get("progress_ms", 0)
            if duration > 0:
                track_progress_pct = (progress / duration) * 100

        # Generate suggestions
        suggestions, confidence = self._generate_suggestions(
            overall_energy=overall_energy,
            crowd_engagement=crowd_engagement,
            music_crowd_alignment=music_crowd_alignment,
            energy_trend=energy_trend,
            party_momentum=party_momentum,
            track_progress_pct=track_progress_pct,
            vision_data=vision_data,
            audio_data=audio_data,
            music_data=music_data
        )

        # Create party state
        party_state = PartyState(
            timestamp=datetime.now().isoformat(),
            overall_energy=overall_energy,
            crowd_engagement=crowd_engagement,
            party_momentum=party_momentum,
            visual_energy=visual_energy,
            audio_energy=audio_energy,
            music_energy=music_energy,
            music_crowd_alignment=music_crowd_alignment,
            energy_variance=energy_variance,
            estimated_people=estimated_people,
            crowd_density=crowd_density,
            crowd_excitement=crowd_excitement,
            current_track_name=current_track_name,
            current_track_tempo=current_track_tempo,
            track_progress_pct=track_progress_pct,
            energy_trend=energy_trend,
            crowd_trend=crowd_trend,
            recommendation_confidence=confidence,
            suggestions=suggestions,
            raw_vision=vision_data,
            raw_audio=audio_data,
            raw_music=music_data
        )

        # Store in history
        self.state_history.append(party_state)

        return party_state

    def _extract_visual_energy(self, vision_data: Optional[Dict]) -> float:
        """Extract energy metric from vision analysis."""
        if not vision_data:
            return 0.0

        return vision_data.get("energy", {}).get("overall_energy", 0.0)

    def _extract_audio_energy(self, audio_data: Optional[Dict]) -> float:
        """Extract energy metric from audio analysis."""
        if not audio_data:
            return 0.0

        return audio_data.get("energy", {}).get("audio_energy", 0.0)

    def _extract_music_energy(self, music_data: Optional[Dict]) -> float:
        """
        Extract energy metric from music/track data.

        Spotify doesn't provide direct "energy" in audio features,
        so we derive it from loudness, tempo, and segment data.
        """
        if not music_data:
            return 0.0

        # Use avg_segment_loudness as proxy (typically -60 to 0 dBFS)
        loudness = music_data.get("avg_segment_loudness", -40)
        loudness_normalized = min(max((loudness + 60) / 60, 0), 1.0)

        # Tempo factor (60-180 BPM range)
        tempo = music_data.get("tempo", 120)
        tempo_normalized = min(max((tempo - 60) / 120, 0), 1.0)

        # Section loudness variation (more dynamic = more energy)
        variation = music_data.get("section_loudness_variation", 5.0)
        variation_normalized = min(variation / 10.0, 1.0)

        # Weighted combination
        music_energy = (
            loudness_normalized * 0.5 +
            tempo_normalized * 0.3 +
            variation_normalized * 0.2
        )

        return min(music_energy, 1.0)

    def _calculate_overall_energy(
        self,
        visual: float,
        audio: float,
        music: float
    ) -> float:
        """
        Calculate overall party energy from components.

        Weights:
        - Audio: 40% (most reliable indicator of party state)
        - Visual: 35% (crowd movement is key)
        - Music: 25% (baseline from what's playing)
        """
        overall = (
            audio * 0.40 +
            visual * 0.35 +
            music * 0.25
        )

        return min(overall, 1.0)

    def _calculate_crowd_engagement(
        self,
        density: float,
        excitement: float,
        motion: float
    ) -> float:
        """
        Calculate how engaged the crowd is.

        Engagement = combination of density, excitement, and motion.
        """
        engagement = (
            density * 0.3 +
            excitement * 0.4 +
            motion * 0.3
        )

        return min(engagement, 1.0)

    def _analyze_trend(self, history: deque, metric_name: str) -> str:
        """
        Analyze trend in historical data.

        Returns: "increasing", "decreasing", or "stable"
        """
        if len(history) < 5:
            return "stable"

        # Compare recent average to older average
        recent = np.mean(list(history)[-10:])
        older = np.mean(list(history)[-20:-10]) if len(history) >= 20 else recent

        # Threshold depends on metric
        if metric_name == "energy":
            threshold = 0.1  # 10% change
        elif metric_name == "crowd":
            threshold = 2  # 2 people change
        else:
            threshold = 0.1

        diff = recent - older

        if diff > threshold:
            return "increasing"
        elif diff < -threshold:
            return "decreasing"
        else:
            return "stable"

    def _determine_momentum(
        self,
        overall_energy: float,
        energy_trend: str,
        crowd_engagement: float
    ) -> str:
        """
        Determine party momentum phase.

        Returns: "heating_up", "peak", "cooling_down", or "stable"
        """
        # Peak: high energy, high engagement
        if overall_energy > 0.7 and crowd_engagement > 0.7:
            if energy_trend == "stable":
                return "peak"
            elif energy_trend == "increasing":
                return "heating_up"
            else:
                return "cooling_down"

        # Heating up: increasing energy
        if energy_trend == "increasing":
            return "heating_up"

        # Cooling down: decreasing energy
        if energy_trend == "decreasing":
            return "cooling_down"

        # Otherwise stable
        return "stable"

    def _calculate_alignment(
        self,
        visual: float,
        audio: float,
        music: float
    ) -> float:
        """
        Calculate how well music aligns with crowd state.

        High alignment = music energy matches crowd energy
        Low alignment = mismatch (e.g., slow music but high crowd energy)
        """
        # Compare music energy to crowd energy (avg of visual + audio)
        crowd_energy = (visual + audio) / 2.0

        # Calculate difference
        diff = abs(music - crowd_energy)

        # Convert to alignment score (0 diff = 1.0 alignment, 1.0 diff = 0.0 alignment)
        alignment = 1.0 - diff

        return max(0.0, min(alignment, 1.0))

    def _calculate_variance(
        self,
        visual: float,
        audio: float,
        music: float
    ) -> float:
        """Calculate variance/disagreement between energy sources."""
        energies = [visual, audio, music]
        variance = np.var(energies)

        # Normalize (typical variance range 0-0.25)
        normalized_variance = min(variance / 0.25, 1.0)

        return normalized_variance

    def _generate_suggestions(
        self,
        overall_energy: float,
        crowd_engagement: float,
        music_crowd_alignment: float,
        energy_trend: str,
        party_momentum: str,
        track_progress_pct: Optional[float],
        vision_data: Optional[Dict],
        audio_data: Optional[Dict],
        music_data: Optional[Dict]
    ) -> tuple[List[str], float]:
        """
        Generate actionable DJ suggestions based on party state.

        Returns:
            (suggestions_list, confidence_score)
        """
        suggestions = []
        confidence = 0.8  # Base confidence

        # Check music-crowd alignment
        if music_crowd_alignment < 0.5:
            if overall_energy > 0.6 and music_data:
                music_energy = self._extract_music_energy(music_data)
                if music_energy < overall_energy - 0.2:
                    suggestions.append(
                        f"⚠️ Energy mismatch: Crowd energy is {overall_energy:.1%} but music energy is only {music_energy:.1%}. "
                        "Consider switching to a higher energy track."
                    )
                    confidence = 0.9
                elif music_energy > overall_energy + 0.2:
                    suggestions.append(
                        f"⚠️ Energy mismatch: Music energy is {music_energy:.1%} but crowd is only at {overall_energy:.1%}. "
                        "Consider a smoother/lower energy track."
                    )
                    confidence = 0.9

        # Check if crowd is losing interest
        if crowd_engagement < 0.4 and overall_energy < 0.5:
            suggestions.append(
                "⚠️ Low crowd engagement detected. Consider playing a popular/familiar track to re-engage the crowd."
            )
            confidence = 0.85

        # Check party momentum
        if party_momentum == "cooling_down":
            suggestions.append(
                "📉 Party is cooling down. Consider increasing energy with upbeat tracks or taking requests."
            )

        elif party_momentum == "heating_up":
            suggestions.append(
                "📈 Party is heating up! Maintain momentum with similar energy tracks."
            )

        elif party_momentum == "peak":
            suggestions.append(
                "🔥 Party is at peak energy! Keep the vibe going but prepare for transition."
            )

        # Check track progress (time to switch?)
        if track_progress_pct and track_progress_pct > 80:
            # Track is almost done
            if music_crowd_alignment < 0.6:
                suggestions.append(
                    f"Track is {track_progress_pct:.0f}% complete and crowd alignment is low. Prepare a better-matched next track."
                )
            else:
                suggestions.append(
                    f"Track is {track_progress_pct:.0f}% complete with good crowd response. Queue similar energy track."
                )

        # Check audio patterns
        if audio_data:
            cheering_prob = audio_data.get("content", {}).get("cheering_probability", 0.0)
            speech_prob = audio_data.get("content", {}).get("speech_probability", 0.0)

            if cheering_prob > 0.7:
                suggestions.append(
                    "🎉 High cheering detected! Crowd loves this vibe - note this for future reference."
                )
                confidence = 0.95

            if speech_prob > 0.6 and crowd_engagement < 0.5:
                suggestions.append(
                    "💬 High speech levels detected - people might be talking more than dancing. "
                    "Consider changing the music to recapture attention."
                )

        # Check visual patterns
        if vision_data:
            dance_floor_occ = vision_data.get("crowd", {}).get("dance_floor_occupancy", 0.0)
            motion_level = vision_data.get("movement", {}).get("motion_level", 0.0)

            if dance_floor_occ < 0.3 and motion_level < 0.3:
                suggestions.append(
                    "🚶 Low dance floor activity. Try a genre switch or crowd favorite to bring people back."
                )

            if dance_floor_occ > 0.7 and motion_level > 0.7:
                suggestions.append(
                    "💃 Dance floor is packed and moving! Excellent track selection."
                )

        # Default suggestion if no issues
        if not suggestions:
            suggestions.append(
                f"✅ Party state is good. Energy: {overall_energy:.1%}, Engagement: {crowd_engagement:.1%}, Alignment: {music_crowd_alignment:.1%}"
            )
            confidence = 0.7

        return suggestions, confidence

    def to_dict(self, state: PartyState) -> Dict:
        """Convert PartyState to JSON-serializable dict."""
        return {
            "timestamp": state.timestamp,
            "overall_metrics": {
                "overall_energy": state.overall_energy,
                "crowd_engagement": state.crowd_engagement,
                "party_momentum": state.party_momentum,
            },
            "component_energies": {
                "visual": state.visual_energy,
                "audio": state.audio_energy,
                "music": state.music_energy,
            },
            "alignment": {
                "music_crowd_alignment": state.music_crowd_alignment,
                "energy_variance": state.energy_variance,
            },
            "crowd": {
                "estimated_people": state.estimated_people,
                "density": state.crowd_density,
                "excitement": state.crowd_excitement,
            },
            "music": {
                "current_track": state.current_track_name,
                "tempo": state.current_track_tempo,
                "progress_pct": state.track_progress_pct,
            },
            "trends": {
                "energy_trend": state.energy_trend,
                "crowd_trend": state.crowd_trend,
            },
            "recommendations": {
                "confidence": state.recommendation_confidence,
                "suggestions": state.suggestions,
            },
            "raw_data": {
                "vision": state.raw_vision,
                "audio": state.raw_audio,
                "music": state.raw_music,
            }
        }


# Singleton instance
_party_state_analyzer = None

def get_party_state_analyzer() -> PartyStateAnalyzer:
    """Get or create the singleton party state analyzer instance."""
    global _party_state_analyzer
    if _party_state_analyzer is None:
        _party_state_analyzer = PartyStateAnalyzer()
    return _party_state_analyzer
