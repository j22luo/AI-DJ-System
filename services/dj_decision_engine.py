"""
DJ Decision Engine
Makes intelligent decisions about when to switch tracks, what to play next, and how to build playlists.
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import random


@dataclass
class TrackDecision:
    """Decision about track switching."""
    should_switch: bool
    confidence: float  # 0.0 to 1.0
    reason: str
    urgency: str  # "immediate", "soon", "normal", "no_rush"
    suggested_timing: Optional[float]  # Seconds until switch (if should_switch)


@dataclass
class TrackRecommendation:
    """Recommendation for next track."""
    track_id: str
    track_name: str
    artist: str
    confidence: float  # 0.0 to 1.0
    reasoning: str
    target_energy: float  # Desired energy level 0.0-1.0
    target_tempo: Optional[float]  # Desired BPM


@dataclass
class PlaylistPlan:
    """Plan for next N tracks with energy arc."""
    tracks: List[TrackRecommendation]
    energy_arc: List[float]  # Target energy for each track
    arc_strategy: str  # "build_up", "maintain", "cool_down", "wave"
    estimated_duration_minutes: float


class DJDecisionEngine:
    """Makes intelligent DJ decisions based on party state."""

    def __init__(self):
        """Initialize the DJ decision engine."""
        # Track history for preventing repetition
        self.recent_track_ids = deque(maxlen=20)
        self.recent_artists = deque(maxlen=15)
        self.recent_genres = deque(maxlen=10)

        # Decision history for learning
        self.decision_history = deque(maxlen=100)

        # Energy arc planning
        self.current_energy_target = 0.5
        self.energy_trajectory = "building"  # "building", "maintaining", "descending"

    def should_switch_track(
        self,
        party_state: Dict,
        track_info: Dict,
        time_since_track_start: float
    ) -> TrackDecision:
        """
        Decide if current track should be switched.

        Args:
            party_state: Output from get_party_state
            track_info: Current track info from Spotify
            time_since_track_start: Seconds since track started

        Returns:
            TrackDecision with switch recommendation
        """
        # Extract key metrics
        alignment = party_state.get("alignment", {}).get("music_crowd_alignment", 0.5)
        crowd_engagement = party_state.get("overall_metrics", {}).get("crowd_engagement", 0.5)
        party_momentum = party_state.get("overall_metrics", {}).get("party_momentum", "stable")
        overall_energy = party_state.get("overall_metrics", {}).get("overall_energy", 0.5)

        track_progress = track_info.get("progress_ms", 0) / 1000  # Convert to seconds
        track_duration = track_info.get("duration_ms", 180000) / 1000  # Convert to seconds
        track_progress_pct = (track_progress / track_duration * 100) if track_duration > 0 else 0

        # Decision factors
        should_switch = False
        reason = ""
        urgency = "normal"
        suggested_timing = None
        confidence = 0.5

        # Factor 1: Poor music-crowd alignment (strongest signal)
        if alignment < 0.3:
            should_switch = True
            reason = f"Poor music-crowd alignment ({alignment:.1%}). Track energy doesn't match crowd."
            urgency = "soon" if track_progress_pct > 30 else "immediate"
            confidence = 0.9
            suggested_timing = 10.0  # Switch in 10 seconds

        # Factor 2: Very low crowd engagement
        elif crowd_engagement < 0.25 and track_progress_pct > 25:
            should_switch = True
            reason = f"Very low crowd engagement ({crowd_engagement:.1%}). Crowd is losing interest."
            urgency = "soon"
            confidence = 0.85
            suggested_timing = 15.0

        # Factor 3: Track almost complete
        elif track_progress_pct > 90:
            should_switch = True
            reason = f"Track is {track_progress_pct:.0f}% complete. Prepare next track."
            urgency = "immediate"
            confidence = 1.0
            suggested_timing = track_duration - track_progress  # Switch when track ends

        # Factor 4: Moderate alignment issues + significant progress
        elif alignment < 0.5 and track_progress_pct > 50:
            should_switch = True
            reason = f"Moderate alignment issues ({alignment:.1%}) and track is halfway. Consider switching soon."
            urgency = "normal"
            confidence = 0.65
            suggested_timing = 30.0

        # Factor 5: Party momentum changing
        elif party_momentum == "cooling_down" and overall_energy < 0.4 and track_progress_pct > 40:
            should_switch = True
            reason = f"Party is cooling down (energy: {overall_energy:.1%}). Need energy boost."
            urgency = "normal"
            confidence = 0.7
            suggested_timing = 20.0

        # Factor 6: Track too long (>6 minutes) and past halfway
        elif track_duration > 360 and track_progress_pct > 50:
            should_switch = True
            reason = f"Long track ({track_duration/60:.1f} min) past halfway point. Keep energy fresh."
            urgency = "normal"
            confidence = 0.6
            suggested_timing = 25.0

        # Factor 7: Everything is going well, let it ride
        else:
            should_switch = False
            reason = f"Track performing well. Alignment: {alignment:.1%}, Engagement: {crowd_engagement:.1%}"
            confidence = 0.7

            # But still prepare if track is >70% done
            if track_progress_pct > 70:
                urgency = "no_rush"
                suggested_timing = track_duration - track_progress - 20  # 20 seconds before end

        return TrackDecision(
            should_switch=should_switch,
            confidence=confidence,
            reason=reason,
            urgency=urgency,
            suggested_timing=suggested_timing
        )

    def recommend_next_track(
        self,
        party_state: Dict,
        available_tracks: List[Dict],
        current_track: Optional[Dict] = None
    ) -> TrackRecommendation:
        """
        Recommend the best next track based on party state.

        Args:
            party_state: Output from get_party_state
            available_tracks: List of track dicts with id, name, artist, audio features
            current_track: Current track info (optional, for smooth transitions)

        Returns:
            TrackRecommendation with best next track
        """
        overall_energy = party_state.get("overall_metrics", {}).get("overall_energy", 0.5)
        crowd_engagement = party_state.get("overall_metrics", {}).get("crowd_engagement", 0.5)
        party_momentum = party_state.get("overall_metrics", {}).get("party_momentum", "stable")
        alignment = party_state.get("alignment", {}).get("music_crowd_alignment", 0.5)

        # Determine target energy based on party state
        target_energy = self._calculate_target_energy(
            current_energy=overall_energy,
            engagement=crowd_engagement,
            momentum=party_momentum,
            alignment=alignment
        )

        # Determine target tempo
        target_tempo = self._calculate_target_tempo(target_energy, party_state)

        # Score all available tracks
        scored_tracks = []
        for track in available_tracks:
            score, reasoning = self._score_track(
                track=track,
                target_energy=target_energy,
                target_tempo=target_tempo,
                current_track=current_track,
                party_state=party_state
            )
            scored_tracks.append((score, track, reasoning))

        # Sort by score (descending)
        scored_tracks.sort(key=lambda x: x[0], reverse=True)

        if not scored_tracks:
            # No tracks available - return dummy
            return TrackRecommendation(
                track_id="",
                track_name="No tracks available",
                artist="",
                confidence=0.0,
                reasoning="No tracks in playlist",
                target_energy=target_energy,
                target_tempo=target_tempo
            )

        # Get best track
        best_score, best_track, reasoning = scored_tracks[0]

        # Record decision
        self.recent_track_ids.append(best_track.get("id", ""))
        self.recent_artists.append(best_track.get("artist", ""))

        return TrackRecommendation(
            track_id=best_track.get("id", ""),
            track_name=best_track.get("name", "Unknown"),
            artist=best_track.get("artist", "Unknown"),
            confidence=min(best_score, 1.0),
            reasoning=reasoning,
            target_energy=target_energy,
            target_tempo=target_tempo
        )

    def generate_playlist_plan(
        self,
        party_state: Dict,
        available_tracks: List[Dict],
        num_tracks: int = 5,
        duration_minutes: Optional[float] = None
    ) -> PlaylistPlan:
        """
        Generate a planned sequence of tracks with energy arc strategy.

        Args:
            party_state: Output from get_party_state
            available_tracks: List of available tracks
            num_tracks: Number of tracks to plan
            duration_minutes: Target duration (optional)

        Returns:
            PlaylistPlan with track sequence and energy arc
        """
        overall_energy = party_state.get("overall_metrics", {}).get("overall_energy", 0.5)
        party_momentum = party_state.get("overall_metrics", {}).get("party_momentum", "stable")
        crowd_engagement = party_state.get("overall_metrics", {}).get("crowd_engagement", 0.5)

        # Determine energy arc strategy
        arc_strategy = self._determine_arc_strategy(
            current_energy=overall_energy,
            momentum=party_momentum,
            engagement=crowd_engagement
        )

        # Generate energy arc
        energy_arc = self._generate_energy_arc(
            start_energy=overall_energy,
            num_tracks=num_tracks,
            strategy=arc_strategy
        )

        # Select tracks to match energy arc
        planned_tracks = []
        used_track_ids = set(self.recent_track_ids)  # Avoid recent tracks
        current_track = None

        total_duration = 0

        for i, target_energy in enumerate(energy_arc):
            # Filter out already selected tracks
            available = [t for t in available_tracks if t.get("id") not in used_track_ids]

            if not available:
                # Ran out of unique tracks, reset
                used_track_ids = set(planned_tracks[-5:]) if planned_tracks else set()
                available = [t for t in available_tracks if t.get("id") not in used_track_ids]

            if not available:
                break  # Truly no tracks

            # Create mini party state for this step
            mini_state = party_state.copy()
            mini_state["overall_metrics"]["overall_energy"] = target_energy

            # Recommend track for this energy level
            recommendation = self.recommend_next_track(
                party_state=mini_state,
                available_tracks=available,
                current_track=current_track
            )

            planned_tracks.append(recommendation)
            used_track_ids.add(recommendation.track_id)
            current_track = {
                "id": recommendation.track_id,
                "name": recommendation.track_name,
                "artist": recommendation.artist
            }

            # Track duration (estimate 3.5 minutes if not available)
            total_duration += 3.5

        return PlaylistPlan(
            tracks=planned_tracks,
            energy_arc=energy_arc,
            arc_strategy=arc_strategy,
            estimated_duration_minutes=total_duration
        )

    def _calculate_target_energy(
        self,
        current_energy: float,
        engagement: float,
        momentum: str,
        alignment: float
    ) -> float:
        """Calculate target energy for next track."""
        target = current_energy

        # Adjust based on momentum
        if momentum == "heating_up":
            target += 0.15  # Increase energy
        elif momentum == "cooling_down":
            target += 0.2  # Boost to recover
        elif momentum == "peak":
            target += 0.05  # Slight increase to maintain
        # stable: no change

        # Adjust based on engagement
        if engagement < 0.3:
            target += 0.2  # Need more energy to re-engage
        elif engagement > 0.8:
            target += 0.0  # Keep it steady

        # Adjust based on alignment
        if alignment < 0.4:
            # Misalignment - move toward crowd energy
            target = current_energy  # Match current vibe better

        # Clamp to reasonable range
        return max(0.2, min(target, 0.95))

    def _calculate_target_tempo(self, target_energy: float, party_state: Dict) -> Optional[float]:
        """Calculate target tempo based on target energy."""
        # Rough mapping: energy 0.0-1.0 -> tempo 80-160 BPM
        base_tempo = 80 + (target_energy * 80)

        # Check if there's beat detection from audio
        audio_data = party_state.get("raw_data", {}).get("audio", {})
        if audio_data:
            detected_tempo = audio_data.get("rhythm", {}).get("estimated_tempo")
            if detected_tempo:
                # Crowd is moving to this tempo, stay close
                return detected_tempo

        return base_tempo

    def _score_track(
        self,
        track: Dict,
        target_energy: float,
        target_tempo: Optional[float],
        current_track: Optional[Dict],
        party_state: Dict
    ) -> Tuple[float, str]:
        """
        Score a track for how well it fits the current party state.

        Returns:
            (score, reasoning)
        """
        score = 0.0
        reasons = []

        track_id = track.get("id", "")

        # Penalize recent tracks
        if track_id in self.recent_track_ids:
            score -= 0.5
            reasons.append("Recently played (-0.5)")

        # Penalize recent artists
        artist = track.get("artist", "")
        if artist in self.recent_artists:
            score -= 0.2
            reasons.append(f"Artist {artist} played recently (-0.2)")

        # Energy matching (most important)
        # Derive track energy from audio features if available
        track_energy = self._estimate_track_energy(track)
        energy_diff = abs(track_energy - target_energy)
        energy_score = 1.0 - energy_diff
        score += energy_score * 0.5
        reasons.append(f"Energy match: {energy_score:.2f} (+{energy_score*0.5:.2f})")

        # Tempo matching
        if target_tempo:
            track_tempo = track.get("tempo", 120)
            tempo_diff = abs(track_tempo - target_tempo)
            tempo_score = max(0, 1.0 - (tempo_diff / 40))  # 40 BPM tolerance
            score += tempo_score * 0.2
            reasons.append(f"Tempo match: {tempo_score:.2f} (+{tempo_score*0.2:.2f})")

        # Popularity bonus (crowd pleasers)
        popularity = track.get("popularity", 50) / 100.0
        score += popularity * 0.2
        reasons.append(f"Popularity: {popularity:.2f} (+{popularity*0.2:.2f})")

        # Smooth transition from current track (if applicable)
        if current_track:
            transition_score = self._score_transition(current_track, track)
            score += transition_score * 0.1
            reasons.append(f"Transition: {transition_score:.2f} (+{transition_score*0.1:.2f})")

        reasoning = "; ".join(reasons)
        return score, reasoning

    def _estimate_track_energy(self, track: Dict) -> float:
        """Estimate track energy from available features."""
        # Try to use avg_segment_loudness if available
        loudness = track.get("avg_segment_loudness", track.get("loudness", -20))
        loudness_normalized = min(max((loudness + 60) / 60, 0), 1.0)

        tempo = track.get("tempo", 120)
        tempo_normalized = min(max((tempo - 60) / 120, 0), 1.0)

        # Weighted combination
        energy = loudness_normalized * 0.6 + tempo_normalized * 0.4

        return min(energy, 1.0)

    def _score_transition(self, from_track: Dict, to_track: Dict) -> float:
        """Score how smooth the transition would be between tracks."""
        score = 0.5  # Base score

        # Key compatibility (if available)
        from_key = from_track.get("key")
        to_key = to_track.get("key")

        if from_key is not None and to_key is not None:
            # Same key or fifth apart = smooth
            key_diff = abs(from_key - to_key)
            if key_diff == 0:
                score += 0.3
            elif key_diff in [5, 7]:  # Fifth relationships
                score += 0.2

        # Tempo compatibility
        from_tempo = from_track.get("tempo", 120)
        to_tempo = to_track.get("tempo", 120)

        tempo_diff = abs(from_tempo - to_tempo)
        if tempo_diff < 10:  # Very close
            score += 0.2
        elif tempo_diff < 20:  # Moderately close
            score += 0.1

        return min(score, 1.0)

    def _determine_arc_strategy(
        self,
        current_energy: float,
        momentum: str,
        engagement: float
    ) -> str:
        """Determine energy arc strategy for playlist."""
        if momentum == "heating_up":
            return "build_up"
        elif momentum == "cooling_down" and engagement < 0.4:
            return "build_up"  # Recover energy
        elif momentum == "peak":
            return "maintain"
        elif current_energy > 0.7:
            return "wave"  # Peak and valley for dynamic flow
        else:
            return "build_up"

    def _generate_energy_arc(
        self,
        start_energy: float,
        num_tracks: int,
        strategy: str
    ) -> List[float]:
        """Generate energy arc for playlist."""
        arc = []

        if strategy == "build_up":
            # Gradually increase energy
            for i in range(num_tracks):
                energy = start_energy + (i / num_tracks) * (0.9 - start_energy)
                arc.append(min(energy, 0.95))

        elif strategy == "maintain":
            # Keep energy steady with slight variations
            for i in range(num_tracks):
                variation = random.uniform(-0.05, 0.05)
                arc.append(min(max(start_energy + variation, 0.3), 0.95))

        elif strategy == "cool_down":
            # Gradually decrease energy
            for i in range(num_tracks):
                energy = start_energy - (i / num_tracks) * (start_energy - 0.3)
                arc.append(max(energy, 0.2))

        elif strategy == "wave":
            # Wave pattern for dynamic flow
            for i in range(num_tracks):
                # Sine wave pattern
                position = i / (num_tracks - 1) if num_tracks > 1 else 0
                wave = 0.15 * (1 if i % 2 == 0 else -1)  # Alternating peaks
                arc.append(min(max(start_energy + wave, 0.4), 0.95))

        else:  # Default to build_up
            return self._generate_energy_arc(start_energy, num_tracks, "build_up")

        return arc

    def to_dict_decision(self, decision: TrackDecision) -> Dict:
        """Convert TrackDecision to dict."""
        return {
            "should_switch": decision.should_switch,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "urgency": decision.urgency,
            "suggested_timing_seconds": decision.suggested_timing
        }

    def to_dict_recommendation(self, rec: TrackRecommendation) -> Dict:
        """Convert TrackRecommendation to dict."""
        return {
            "track_id": rec.track_id,
            "track_name": rec.track_name,
            "artist": rec.artist,
            "confidence": rec.confidence,
            "reasoning": rec.reasoning,
            "target_energy": rec.target_energy,
            "target_tempo": rec.target_tempo
        }

    def to_dict_playlist_plan(self, plan: PlaylistPlan) -> Dict:
        """Convert PlaylistPlan to dict."""
        return {
            "tracks": [self.to_dict_recommendation(t) for t in plan.tracks],
            "energy_arc": plan.energy_arc,
            "arc_strategy": plan.arc_strategy,
            "estimated_duration_minutes": plan.estimated_duration_minutes
        }


# Singleton instance
_dj_decision_engine = None

def get_dj_decision_engine() -> DJDecisionEngine:
    """Get or create the singleton DJ decision engine instance."""
    global _dj_decision_engine
    if _dj_decision_engine is None:
        _dj_decision_engine = DJDecisionEngine()
    return _dj_decision_engine
