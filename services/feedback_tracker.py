"""
Feedback Tracker
Tracks DJ decisions and their outcomes to enable learning and improvement.
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque


@dataclass
class TrackPlayback:
    """Record of a track playback session."""
    timestamp: str
    track_id: str
    track_name: str
    artist: str

    # Party state when track started
    starting_energy: float
    starting_engagement: float
    starting_momentum: str

    # Party state when track ended (if available)
    ending_energy: Optional[float] = None
    ending_engagement: Optional[float] = None
    ending_momentum: Optional[str] = None

    # Playback metrics
    played_duration_seconds: Optional[float] = None
    completed: bool = False  # Did track play to completion?
    skipped: bool = False  # Was track skipped early?

    # Crowd reaction metrics (if available)
    avg_cheering_level: Optional[float] = None
    avg_movement_level: Optional[float] = None
    crowd_size_change: Optional[int] = None  # People joined/left during track

    # Decision context
    was_recommended: bool = False
    recommendation_confidence: Optional[float] = None
    alignment_score: Optional[float] = None

    # Outcome assessment
    success_score: Optional[float] = None  # 0.0 to 1.0
    notes: str = ""


@dataclass
class DecisionOutcome:
    """Record of a DJ decision and its outcome."""
    timestamp: str
    decision_type: str  # "switch_track", "recommend_track", "generate_playlist"

    # Decision details
    decision_data: Dict

    # Outcome (measured after decision executed)
    outcome_energy_change: Optional[float] = None
    outcome_engagement_change: Optional[float] = None
    outcome_success: Optional[bool] = None
    outcome_notes: str = ""


class FeedbackTracker:
    """Tracks decisions and outcomes for learning."""

    def __init__(self, data_dir: str = "feedback_data"):
        """
        Initialize the feedback tracker.

        Args:
            data_dir: Directory to store feedback data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # In-memory tracking
        self.current_session_playbacks: List[TrackPlayback] = []
        self.current_session_decisions: List[DecisionOutcome] = []

        # Current track tracking
        self.current_track: Optional[TrackPlayback] = None
        self.track_start_time: Optional[datetime] = None

        # Historical statistics
        self.track_success_rates: Dict[str, List[float]] = defaultdict(list)  # track_id -> success scores
        self.artist_performance: Dict[str, List[float]] = defaultdict(list)  # artist -> success scores
        self.energy_transitions: List[Dict] = []  # Track energy transition effectiveness

        # Load existing data
        self._load_historical_data()

    def start_track(
        self,
        track_id: str,
        track_name: str,
        artist: str,
        party_state: Dict,
        was_recommended: bool = False,
        recommendation_confidence: Optional[float] = None
    ) -> None:
        """
        Record that a track has started playing.

        Args:
            track_id: Spotify track ID
            track_name: Track name
            artist: Artist name
            party_state: Current party state dict
            was_recommended: Whether this was an AI recommendation
            recommendation_confidence: Confidence of recommendation (if applicable)
        """
        # End previous track if any
        if self.current_track:
            self.end_track(party_state, completed=False, notes="New track started before previous ended")

        # Create new playback record
        overall_metrics = party_state.get("overall_metrics", {})
        alignment = party_state.get("alignment", {})

        self.current_track = TrackPlayback(
            timestamp=datetime.now().isoformat(),
            track_id=track_id,
            track_name=track_name,
            artist=artist,
            starting_energy=overall_metrics.get("overall_energy", 0.5),
            starting_engagement=overall_metrics.get("crowd_engagement", 0.5),
            starting_momentum=overall_metrics.get("party_momentum", "stable"),
            was_recommended=was_recommended,
            recommendation_confidence=recommendation_confidence,
            alignment_score=alignment.get("music_crowd_alignment", 0.5)
        )

        self.track_start_time = datetime.now()

        print(f"📀 Track started: {track_name} by {artist} (ID: {track_id[:10]}...)")

    def end_track(
        self,
        party_state: Dict,
        completed: bool = True,
        skipped: bool = False,
        notes: str = ""
    ) -> None:
        """
        Record that a track has ended.

        Args:
            party_state: Party state at end of track
            completed: Whether track played to completion
            skipped: Whether track was skipped early
            notes: Additional notes about the outcome
        """
        if not self.current_track:
            print("⚠️ end_track called but no current track")
            return

        # Calculate duration
        if self.track_start_time:
            duration = (datetime.now() - self.track_start_time).total_seconds()
            self.current_track.played_duration_seconds = duration

        # Record ending state
        overall_metrics = party_state.get("overall_metrics", {})

        self.current_track.ending_energy = overall_metrics.get("overall_energy", 0.5)
        self.current_track.ending_engagement = overall_metrics.get("crowd_engagement", 0.5)
        self.current_track.ending_momentum = overall_metrics.get("party_momentum", "stable")

        self.current_track.completed = completed
        self.current_track.skipped = skipped
        self.current_track.notes = notes

        # Extract crowd reaction metrics from party state
        crowd = party_state.get("crowd", {})
        audio_raw = party_state.get("raw_data", {}).get("audio", {})
        vision_raw = party_state.get("raw_data", {}).get("vision", {})

        if audio_raw:
            self.current_track.avg_cheering_level = audio_raw.get("crowd", {}).get("excitement", 0.0)

        if vision_raw:
            self.current_track.avg_movement_level = vision_raw.get("movement", {}).get("motion_level", 0.0)

        # Calculate success score
        self.current_track.success_score = self._calculate_success_score(self.current_track)

        # Record in session
        self.current_session_playbacks.append(self.current_track)

        # Update historical stats
        self.track_success_rates[self.current_track.track_id].append(self.current_track.success_score)
        self.artist_performance[self.current_track.artist].append(self.current_track.success_score)

        # Track energy transition
        if self.current_track.ending_energy is not None:
            energy_change = self.current_track.ending_energy - self.current_track.starting_energy
            self.energy_transitions.append({
                "track_id": self.current_track.track_id,
                "starting_energy": self.current_track.starting_energy,
                "ending_energy": self.current_track.ending_energy,
                "change": energy_change,
                "success": self.current_track.success_score
            })

        print(f"🏁 Track ended: {self.current_track.track_name} (Success: {self.current_track.success_score:.2f})")

        # Save to disk
        self._save_playback(self.current_track)

        # Clear current track
        self.current_track = None
        self.track_start_time = None

    def record_decision(
        self,
        decision_type: str,
        decision_data: Dict,
        party_state_before: Dict
    ) -> str:
        """
        Record a DJ decision for later outcome tracking.

        Args:
            decision_type: Type of decision ("switch_track", "recommend_track", etc.)
            decision_data: The decision details
            party_state_before: Party state before decision

        Returns:
            Decision ID for later outcome updates
        """
        decision = DecisionOutcome(
            timestamp=datetime.now().isoformat(),
            decision_type=decision_type,
            decision_data=decision_data
        )

        self.current_session_decisions.append(decision)

        decision_id = f"{decision_type}_{len(self.current_session_decisions)}"

        print(f"📝 Decision recorded: {decision_type} (ID: {decision_id})")

        return decision_id

    def update_decision_outcome(
        self,
        decision_id: str,
        party_state_after: Dict,
        success: bool,
        notes: str = ""
    ) -> None:
        """
        Update a decision with its outcome.

        Args:
            decision_id: ID returned from record_decision
            party_state_after: Party state after decision was executed
            success: Whether the decision was successful
            notes: Additional outcome notes
        """
        # Find decision by ID (simplified - using index)
        try:
            parts = decision_id.split("_")
            index = int(parts[-1]) - 1

            if 0 <= index < len(self.current_session_decisions):
                decision = self.current_session_decisions[index]
                decision.outcome_success = success
                decision.outcome_notes = notes

                print(f"✅ Decision outcome updated: {decision_id} -> {'Success' if success else 'Failure'}")
        except Exception as e:
            print(f"Error updating decision outcome: {e}")

    def get_track_success_rate(self, track_id: str) -> Optional[float]:
        """Get historical success rate for a track."""
        if track_id in self.track_success_rates:
            return sum(self.track_success_rates[track_id]) / len(self.track_success_rates[track_id])
        return None

    def get_artist_performance(self, artist: str) -> Optional[float]:
        """Get historical performance for an artist."""
        if artist in self.artist_performance:
            return sum(self.artist_performance[artist]) / len(self.artist_performance[artist])
        return None

    def get_best_performing_tracks(self, top_n: int = 10) -> List[Dict]:
        """Get best performing tracks from history."""
        track_scores = []

        for track_id, scores in self.track_success_rates.items():
            if len(scores) >= 2:  # At least 2 plays to be meaningful
                avg_score = sum(scores) / len(scores)
                track_scores.append({
                    "track_id": track_id,
                    "avg_success_score": avg_score,
                    "play_count": len(scores)
                })

        # Sort by score
        track_scores.sort(key=lambda x: x["avg_success_score"], reverse=True)

        return track_scores[:top_n]

    def get_session_summary(self) -> Dict:
        """Get summary of current session."""
        total_tracks = len(self.current_session_playbacks)

        if total_tracks == 0:
            return {
                "total_tracks": 0,
                "message": "No tracks played yet this session"
            }

        avg_success = sum(t.success_score for t in self.current_session_playbacks if t.success_score) / total_tracks
        completed_count = sum(1 for t in self.current_session_playbacks if t.completed)
        skipped_count = sum(1 for t in self.current_session_playbacks if t.skipped)

        energy_changes = [
            t.ending_energy - t.starting_energy
            for t in self.current_session_playbacks
            if t.ending_energy is not None
        ]

        return {
            "total_tracks": total_tracks,
            "avg_success_score": avg_success,
            "completed_count": completed_count,
            "skipped_count": skipped_count,
            "avg_energy_change": sum(energy_changes) / len(energy_changes) if energy_changes else 0,
            "tracks": [
                {
                    "track_name": t.track_name,
                    "artist": t.artist,
                    "success_score": t.success_score,
                    "completed": t.completed
                }
                for t in self.current_session_playbacks
            ]
        }

    def _calculate_success_score(self, playback: TrackPlayback) -> float:
        """
        Calculate success score for a track playback.

        Success factors:
        - Energy improvement (or maintenance if already high)
        - Engagement improvement
        - Track completion
        - High crowd excitement
        """
        score = 0.5  # Base score

        # Factor 1: Completion (+0.2)
        if playback.completed:
            score += 0.2
        elif playback.skipped:
            score -= 0.3

        # Factor 2: Energy trajectory (+/- 0.2)
        if playback.ending_energy is not None:
            energy_change = playback.ending_energy - playback.starting_energy

            if playback.starting_energy < 0.5:
                # Low energy starting point - reward increase
                score += energy_change * 0.4
            else:
                # High energy starting point - reward maintenance
                if energy_change > -0.1:  # Maintained or increased
                    score += 0.2
                else:
                    score -= 0.2

        # Factor 3: Engagement (+/- 0.2)
        if playback.ending_engagement is not None:
            engagement_change = playback.ending_engagement - playback.starting_engagement
            score += engagement_change * 0.4

        # Factor 4: Crowd excitement (+0.2)
        if playback.avg_cheering_level and playback.avg_cheering_level > 0.6:
            score += 0.2

        # Factor 5: Movement level (+0.1)
        if playback.avg_movement_level and playback.avg_movement_level > 0.6:
            score += 0.1

        # Clamp to 0-1
        return max(0.0, min(score, 1.0))

    def _save_playback(self, playback: TrackPlayback) -> None:
        """Save playback record to disk."""
        filename = f"playback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{playback.track_id[:8]}.json"
        filepath = os.path.join(self.data_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(asdict(playback), f, indent=2)
        except Exception as e:
            print(f"Error saving playback: {e}")

    def _load_historical_data(self) -> None:
        """Load historical playback data from disk."""
        if not os.path.exists(self.data_dir):
            return

        try:
            for filename in os.listdir(self.data_dir):
                if filename.startswith("playback_") and filename.endswith(".json"):
                    filepath = os.path.join(self.data_dir, filename)

                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        playback = TrackPlayback(**data)

                        # Add to statistics
                        if playback.success_score is not None:
                            self.track_success_rates[playback.track_id].append(playback.success_score)
                            self.artist_performance[playback.artist].append(playback.success_score)

            print(f"📊 Loaded historical data: {len(self.track_success_rates)} tracks, {len(self.artist_performance)} artists")

        except Exception as e:
            print(f"Error loading historical data: {e}")


# Singleton instance
_feedback_tracker = None

def get_feedback_tracker() -> FeedbackTracker:
    """Get or create the singleton feedback tracker instance."""
    global _feedback_tracker
    if _feedback_tracker is None:
        _feedback_tracker = FeedbackTracker()
    return _feedback_tracker
