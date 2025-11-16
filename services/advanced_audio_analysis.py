"""
Advanced Audio Analysis Service
Analyzes microphone audio for speech, cheering, crowd patterns, and party energy.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import scipy.signal as signal
from collections import deque


@dataclass
class AudioAnalysisResult:
    """Results from advanced audio analysis."""
    timestamp: str

    # Volume metrics
    dbfs: float  # Current dBFS level
    volume_trend: str  # "increasing", "decreasing", "stable"

    # Content classification
    speech_probability: float  # 0.0 to 1.0
    music_probability: float  # 0.0 to 1.0
    cheering_probability: float  # 0.0 to 1.0

    # Crowd metrics
    crowd_noise_level: float  # 0.0 to 1.0
    crowd_excitement: float  # 0.0 to 1.0 (derived from volume spikes + frequency patterns)

    # Frequency analysis
    dominant_frequency: float  # Hz
    bass_energy: float  # 0.0 to 1.0
    mid_energy: float  # 0.0 to 1.0
    high_energy: float  # 0.0 to 1.0

    # Temporal patterns
    beat_detected: bool
    estimated_tempo: Optional[float]  # BPM if detectable

    # Overall party audio energy
    audio_energy: float  # 0.0 to 1.0

    # Raw metrics
    raw_metrics: Dict


class AdvancedAudioAnalyzer:
    """Analyzes microphone audio for party metrics."""

    def __init__(self, sample_rate: int = 22050, buffer_seconds: float = 5.0):
        """
        Initialize the advanced audio analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
            buffer_seconds: How many seconds of history to keep
        """
        self.sample_rate = sample_rate
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(sample_rate * buffer_seconds)

        # Rolling buffers for trend analysis
        self.dbfs_history = deque(maxlen=50)  # Last 50 measurements
        self.energy_history = deque(maxlen=50)

        # Frequency band definitions (Hz)
        self.BASS_RANGE = (20, 250)
        self.MID_RANGE = (250, 4000)
        self.HIGH_RANGE = (4000, 11000)

        # Speech frequency range (fundamental + harmonics)
        self.SPEECH_RANGE = (80, 3500)

        # Cheering typically has high energy bursts with wide frequency spread
        self.CHEER_FREQ_RANGE = (500, 8000)

    def analyze_audio(
        self,
        audio_samples: np.ndarray,
        current_dbfs: float,
        spectral_centroid: float
    ) -> AudioAnalysisResult:
        """
        Perform advanced analysis on audio samples.

        Args:
            audio_samples: Raw audio sample data (mono)
            current_dbfs: Current dBFS level from basic analysis
            spectral_centroid: Spectral centroid from basic analysis

        Returns:
            AudioAnalysisResult with all analyzed metrics
        """
        # Store history
        self.dbfs_history.append(current_dbfs)

        # Calculate volume trend
        volume_trend = self._calculate_volume_trend()

        # Perform FFT for frequency analysis
        fft_result = np.fft.rfft(audio_samples)
        fft_magnitude = np.abs(fft_result)
        fft_freqs = np.fft.rfftfreq(len(audio_samples), 1.0 / self.sample_rate)

        # Band energy analysis
        bass_energy = self._calculate_band_energy(fft_magnitude, fft_freqs, self.BASS_RANGE)
        mid_energy = self._calculate_band_energy(fft_magnitude, fft_freqs, self.MID_RANGE)
        high_energy = self._calculate_band_energy(fft_magnitude, fft_freqs, self.HIGH_RANGE)

        # Dominant frequency
        dominant_freq = fft_freqs[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0.0

        # Content classification
        speech_prob = self._estimate_speech_probability(
            fft_magnitude, fft_freqs, current_dbfs, spectral_centroid
        )
        music_prob = self._estimate_music_probability(
            bass_energy, mid_energy, high_energy, audio_samples
        )
        cheering_prob = self._estimate_cheering_probability(
            fft_magnitude, fft_freqs, current_dbfs, audio_samples
        )

        # Crowd metrics
        crowd_noise = self._estimate_crowd_noise(speech_prob, cheering_prob, current_dbfs)
        crowd_excitement = self._estimate_crowd_excitement(
            cheering_prob, volume_trend, current_dbfs
        )

        # Beat detection
        beat_detected, estimated_tempo = self._detect_beat(audio_samples)

        # Overall audio energy
        audio_energy = self._calculate_audio_energy(
            current_dbfs, bass_energy, crowd_noise, crowd_excitement
        )

        self.energy_history.append(audio_energy)

        # Compile raw metrics
        raw_metrics = {
            "sample_count": len(audio_samples),
            "fft_peak_freq": float(dominant_freq),
            "dbfs_history_len": len(self.dbfs_history),
            "spectral_centroid": float(spectral_centroid)
        }

        return AudioAnalysisResult(
            timestamp=datetime.now().isoformat(),
            dbfs=current_dbfs,
            volume_trend=volume_trend,
            speech_probability=speech_prob,
            music_probability=music_prob,
            cheering_probability=cheering_prob,
            crowd_noise_level=crowd_noise,
            crowd_excitement=crowd_excitement,
            dominant_frequency=dominant_freq,
            bass_energy=bass_energy,
            mid_energy=mid_energy,
            high_energy=high_energy,
            beat_detected=beat_detected,
            estimated_tempo=estimated_tempo,
            audio_energy=audio_energy,
            raw_metrics=raw_metrics
        )

    def _calculate_volume_trend(self) -> str:
        """
        Calculate if volume is increasing, decreasing, or stable.

        Returns:
            "increasing", "decreasing", or "stable"
        """
        if len(self.dbfs_history) < 5:
            return "stable"

        # Compare recent average to older average
        recent = np.mean(list(self.dbfs_history)[-10:])
        older = np.mean(list(self.dbfs_history)[-20:-10]) if len(self.dbfs_history) >= 20 else recent

        diff = recent - older

        if diff > 2.0:  # dBFS increase
            return "increasing"
        elif diff < -2.0:  # dBFS decrease
            return "decreasing"
        else:
            return "stable"

    def _calculate_band_energy(
        self,
        fft_magnitude: np.ndarray,
        fft_freqs: np.ndarray,
        freq_range: Tuple[float, float]
    ) -> float:
        """
        Calculate energy in a frequency band.

        Returns:
            float: 0.0 to 1.0 normalized energy
        """
        # Find indices within frequency range
        mask = (fft_freqs >= freq_range[0]) & (fft_freqs <= freq_range[1])
        band_energy = np.sum(fft_magnitude[mask] ** 2)

        # Normalize by total energy
        total_energy = np.sum(fft_magnitude ** 2)

        if total_energy > 0:
            return min(band_energy / total_energy, 1.0)
        return 0.0

    def _estimate_speech_probability(
        self,
        fft_magnitude: np.ndarray,
        fft_freqs: np.ndarray,
        dbfs: float,
        spectral_centroid: float
    ) -> float:
        """
        Estimate probability that audio contains speech.

        Speech characteristics:
        - Energy in 80-3500 Hz range (fundamental + formants)
        - Spectral centroid typically 500-2500 Hz for speech
        - Moderate volume (not too loud, not too quiet)
        - Irregular temporal pattern
        """
        # Energy in speech frequency range
        speech_energy = self._calculate_band_energy(fft_magnitude, fft_freqs, self.SPEECH_RANGE)

        # Spectral centroid in speech range
        centroid_score = 0.0
        if 500 <= spectral_centroid <= 2500:
            # Peak score at 1500 Hz
            centroid_score = 1.0 - abs(spectral_centroid - 1500) / 1500
            centroid_score = max(0.0, centroid_score)

        # Volume in speech range (-40 to -10 dBFS)
        volume_score = 0.0
        if -40 <= dbfs <= -10:
            # Peak score at -25 dBFS
            volume_score = 1.0 - abs(dbfs + 25) / 15
            volume_score = max(0.0, volume_score)

        # Weighted combination
        speech_prob = (
            speech_energy * 0.5 +
            centroid_score * 0.3 +
            volume_score * 0.2
        )

        return min(speech_prob, 1.0)

    def _estimate_music_probability(
        self,
        bass_energy: float,
        mid_energy: float,
        high_energy: float,
        audio_samples: np.ndarray
    ) -> float:
        """
        Estimate probability that audio contains music.

        Music characteristics:
        - Strong bass component
        - Balanced frequency distribution
        - Regular temporal patterns (rhythm)
        - Higher overall energy
        """
        # Music typically has strong bass
        bass_score = bass_energy

        # Balanced frequency distribution
        # Calculate variance - lower variance = more balanced
        energies = [bass_energy, mid_energy, high_energy]
        energy_variance = np.var(energies)
        balance_score = 1.0 - min(energy_variance, 1.0)

        # Regularity check via autocorrelation
        # Regular patterns have stronger autocorrelation
        if len(audio_samples) > 1000:
            autocorr = np.correlate(audio_samples[:1000], audio_samples[:1000], mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            # Normalize
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]
                # Check for peaks (indicating periodicity)
                peaks, _ = signal.find_peaks(autocorr, height=0.3, distance=20)
                regularity_score = min(len(peaks) / 5.0, 1.0)
            else:
                regularity_score = 0.0
        else:
            regularity_score = 0.0

        # Weighted combination
        music_prob = (
            bass_score * 0.4 +
            balance_score * 0.3 +
            regularity_score * 0.3
        )

        return min(music_prob, 1.0)

    def _estimate_cheering_probability(
        self,
        fft_magnitude: np.ndarray,
        fft_freqs: np.ndarray,
        dbfs: float,
        audio_samples: np.ndarray
    ) -> float:
        """
        Estimate probability that audio contains cheering/crowd excitement.

        Cheering characteristics:
        - Sudden volume spikes
        - Wide frequency spread
        - High energy in mid-high frequencies
        - Irregular bursts
        """
        # Energy in cheering frequency range
        cheer_energy = self._calculate_band_energy(fft_magnitude, fft_freqs, self.CHEER_FREQ_RANGE)

        # Loud volume (cheering is typically loud)
        volume_score = 0.0
        if dbfs > -20:  # Loud
            volume_score = min((dbfs + 20) / 10, 1.0)

        # Check for sudden spikes in amplitude
        if len(self.dbfs_history) >= 5:
            recent_dbfs = list(self.dbfs_history)[-5:]
            max_recent = max(recent_dbfs)
            avg_older = np.mean(list(self.dbfs_history)[:-5]) if len(self.dbfs_history) > 5 else max_recent
            spike_score = min(max(max_recent - avg_older, 0) / 10, 1.0)
        else:
            spike_score = 0.0

        # High frequency spread (cheering has wide spectrum)
        # Calculate spectral flatness
        if len(fft_magnitude) > 0 and np.min(fft_magnitude) > 0:
            geometric_mean = np.exp(np.mean(np.log(fft_magnitude + 1e-10)))
            arithmetic_mean = np.mean(fft_magnitude)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            spread_score = spectral_flatness  # Higher = more noise-like (cheering)
        else:
            spread_score = 0.0

        # Weighted combination
        cheering_prob = (
            cheer_energy * 0.3 +
            volume_score * 0.3 +
            spike_score * 0.25 +
            spread_score * 0.15
        )

        return min(cheering_prob, 1.0)

    def _estimate_crowd_noise(
        self,
        speech_prob: float,
        cheering_prob: float,
        dbfs: float
    ) -> float:
        """
        Estimate overall crowd noise level.

        Combines speech and cheering probabilities with volume.
        """
        # Crowd noise is combination of speech and cheering
        content_score = max(speech_prob, cheering_prob)

        # Volume factor (louder = more crowd)
        volume_factor = min(max(dbfs + 40, 0) / 30, 1.0)  # -40 to -10 dBFS range

        crowd_noise = (content_score * 0.7 + volume_factor * 0.3)

        return min(crowd_noise, 1.0)

    def _estimate_crowd_excitement(
        self,
        cheering_prob: float,
        volume_trend: str,
        dbfs: float
    ) -> float:
        """
        Estimate crowd excitement level.

        Excitement indicated by:
        - High cheering probability
        - Increasing volume
        - High overall volume
        """
        # Base excitement from cheering
        excitement = cheering_prob

        # Volume trend bonus
        if volume_trend == "increasing":
            excitement += 0.2
        elif volume_trend == "decreasing":
            excitement -= 0.1

        # Absolute volume bonus
        if dbfs > -15:  # Very loud
            excitement += 0.15

        return min(max(excitement, 0.0), 1.0)

    def _detect_beat(self, audio_samples: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Detect if there's a beat and estimate tempo.

        Returns:
            (beat_detected, estimated_tempo_bpm)
        """
        if len(audio_samples) < 1000:
            return False, None

        # Calculate onset strength (energy in spectral flux)
        # Use envelope detection
        envelope = np.abs(signal.hilbert(audio_samples))

        # Find peaks in envelope (potential beats)
        # Normalize first
        if np.max(envelope) > 0:
            envelope = envelope / np.max(envelope)

        peaks, properties = signal.find_peaks(
            envelope,
            height=0.3,
            distance=int(self.sample_rate * 0.3)  # Min 0.3s between beats (200 BPM max)
        )

        if len(peaks) < 2:
            return False, None

        # Calculate intervals between peaks
        intervals = np.diff(peaks) / self.sample_rate  # Convert to seconds

        # Median interval
        median_interval = np.median(intervals)

        # Convert to BPM
        if median_interval > 0:
            estimated_bpm = 60.0 / median_interval

            # Typical music range is 60-180 BPM
            if 60 <= estimated_bpm <= 180:
                return True, estimated_bpm

        return len(peaks) >= 3, None  # Beat detected but can't estimate tempo

    def _calculate_audio_energy(
        self,
        dbfs: float,
        bass_energy: float,
        crowd_noise: float,
        crowd_excitement: float
    ) -> float:
        """
        Calculate overall audio energy metric.

        Combines:
        - Volume (dBFS)
        - Bass energy (music energy)
        - Crowd noise
        - Crowd excitement
        """
        # Normalize dBFS to 0-1 (-40 to 0 dBFS range)
        volume_normalized = min(max(dbfs + 40, 0) / 40, 1.0)

        energy = (
            volume_normalized * 0.3 +
            bass_energy * 0.3 +
            crowd_noise * 0.2 +
            crowd_excitement * 0.2
        )

        return min(energy, 1.0)

    def to_dict(self, result: AudioAnalysisResult) -> Dict:
        """Convert result to JSON-serializable dict."""
        return {
            "timestamp": result.timestamp,
            "volume": {
                "dbfs": result.dbfs,
                "trend": result.volume_trend
            },
            "content": {
                "speech_probability": result.speech_probability,
                "music_probability": result.music_probability,
                "cheering_probability": result.cheering_probability
            },
            "crowd": {
                "noise_level": result.crowd_noise_level,
                "excitement": result.crowd_excitement
            },
            "frequency": {
                "dominant_frequency": result.dominant_frequency,
                "bass_energy": result.bass_energy,
                "mid_energy": result.mid_energy,
                "high_energy": result.high_energy
            },
            "rhythm": {
                "beat_detected": result.beat_detected,
                "estimated_tempo": result.estimated_tempo
            },
            "energy": {
                "audio_energy": result.audio_energy
            },
            "raw_metrics": result.raw_metrics
        }


# Singleton instance
_audio_analyzer = None

def get_audio_analyzer(sample_rate: int = 22050) -> AdvancedAudioAnalyzer:
    """Get or create the singleton audio analyzer instance."""
    global _audio_analyzer
    if _audio_analyzer is None:
        _audio_analyzer = AdvancedAudioAnalyzer(sample_rate=sample_rate)
    return _audio_analyzer
