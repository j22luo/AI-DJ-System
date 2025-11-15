"""
Mic dBFS + FFT Frequency Monitor (Server-Friendly)

- Captures microphone audio in real time.
- Every 0.01 s (~100 Hz), computes:
    - dBFS loudness (decibels relative to full scale)
    - FFT-based spectral centroid (Hz) with simple smoothing
- Keeps a sliding window of the last `history_seconds` of data.

Designed for server/MCP usage:
    - start() / stop()
    - get_recent_data()
    - get_graph_snapshot()
    - get_graph_image() -> PNG bytes (for Claude / clients)

Requirements:
    pip install numpy sounddevice matplotlib
"""

import time
from datetime import datetime
from typing import List, Tuple, Optional, Deque, Dict, Any
from collections import deque
import io

import numpy as np
import sounddevice as sd

import matplotlib
matplotlib.use("Agg")  # headless backend for server environments
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Optional helper: list devices (for manual testing)
# -------------------------------------------------------------

def list_input_devices() -> None:
    """Print all input-capable devices with indices."""
    devices = sd.query_devices()
    print("\nAvailable audio input devices:")
    print("=" * 60)
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"[{i}] {d['name']}")
    print("=" * 60 + "\n")


# -------------------------------------------------------------
# Core class: MicDBFFrequencyMonitor
# -------------------------------------------------------------

class MicDBFFrequencyMonitor:
    """
    Real-time microphone monitor.

    Each sample stored as:
        (timestamp_iso: str, dbfs: float, spectral_centroid_hz: float)

    Public API:
        start()               -> begin capturing
        stop()                -> stop capturing
        get_recent_data()     -> raw tuples
        get_graph_snapshot()  -> JSON-like dict for last history_seconds
        get_graph_image()     -> PNG bytes (graph of last history_seconds)
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 22050,
        block_duration: float = 0.01,   # 10ms blocks (~100 Hz)
        history_seconds: float = 5.0,   # keep last 5 seconds
    ):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.block_duration = block_duration

        self.block_size = int(sample_rate * block_duration)
        self.max_blocks = max(1, int(history_seconds / block_duration))

        # Deque of (timestamp_iso, dbfs, centroid_hz)
        self._history: Deque[Tuple[str, float, float]] = deque(
            maxlen=self.max_blocks
        )

        self._stream: Optional[sd.InputStream] = None
        self._running = False

        # For snapshot metadata
        self.history_seconds = history_seconds

    # ---------------------------------------------------------
    # Internal metrics
    # ---------------------------------------------------------

    def _compute_dbfs(self, audio: np.ndarray) -> float:
        """
        Compute dBFS from audio samples in [-1, 1].
        Returns negative values; 0 dBFS is full-scale.
        """
        if len(audio) == 0:
            return -80.0

        rms = np.sqrt(np.mean(audio ** 2))
        if rms <= 1e-12:
            return -80.0

        dbfs = 20.0 * np.log10(rms)
        return float(dbfs)

    def _compute_frequency_fft(self, audio: np.ndarray) -> float:
        """
        Spectral centroid via FFT, with a little smoothing
        to make it less spiky.
        """
        if len(audio) == 0:
            return 0.0

        # FFT magnitude spectrum
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)

        # Simple moving average smoothing to reduce peaks
        window = 5
        if len(spectrum) > window:
            spectrum = np.convolve(spectrum, np.ones(window) / window, mode="same")

        total = np.sum(spectrum)
        if total <= 0:
            return 0.0

        centroid = float(np.sum(freqs * spectrum) / total)
        return centroid

    # ---------------------------------------------------------
    # Audio callback
    # ---------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio callback status:", status)

        # indata: (frames, channels)
        if indata.shape[1] > 1:
            audio = np.mean(indata, axis=1)
        else:
            audio = indata[:, 0]

        audio = audio.astype(np.float32, copy=False)

        dbfs = self._compute_dbfs(audio)
        freq = self._compute_frequency_fft(audio)
        ts = datetime.now().isoformat()

        self._history.append((ts, dbfs, freq))

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def start(self) -> None:
        """Start capturing from the microphone."""
        if self._running:
            print("MicDBFFrequencyMonitor is already running.")
            return

        self._running = True

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=self.block_size,
            device=self.device_index,
        )
        self._stream.start()

        print(
            f"MicDBFFrequencyMonitor started "
            f"(device={self.device_index}, sr={self.sample_rate}, "
            f"block={self.block_duration}s)"
        )

    def stop(self) -> None:
        """Stop capturing from the microphone."""
        if not self._running:
            return

        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        print("MicDBFFrequencyMonitor stopped.")

    def get_recent_data(self) -> List[Tuple[str, float, float]]:
        """
        Returns a list of:
            (timestamp_iso, dbfs_level, spectral_centroid_hz)
        for roughly the last `history_seconds` seconds.
        """
        return list(self._history)

    def clear_history(self) -> None:
        """Clear all stored samples."""
        self._history.clear()

    def get_graph_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Return a snapshot of the last `history_seconds` worth of data,
        in a graph-friendly format:

        {
          "duration_seconds": float,
          "sample_interval_seconds": float,
          "points": [
            {"t": -4.99, "dbfs": -32.1, "freq_hz": 1500.2},
            ...
            {"t": -0.01, "dbfs": -20.7, "freq_hz": 1850.9}
          ],
          "dbfs_axis": {"min": -80.0, "max": 0.0},
          "freq_axis": {"min": 0.0, "max": sample_rate/2}
        }

        `t` is time relative to "now" in seconds (negative = past).
        """
        data = self.get_recent_data()
        if not data:
            return None

        n = len(data)
        # last sample ~ 0s, previous negative
        times = [-(n - 1 - i) * self.block_duration for i in range(n)]

        dbfs_values = [d[1] for d in data]
        freq_values = [d[2] for d in data]

        # Axis ranges
        db_min = min(dbfs_values + [-80.0])
        db_max = max(dbfs_values + [-10.0])
        db_min = min(db_min, -80.0)
        db_max = max(db_max, 0.0)

        freq_min = 0.0
        freq_max = float(self.sample_rate) / 2.0

        points = [
            {
                "t": float(t),
                "dbfs": float(db),
                "freq_hz": float(f),
            }
            for t, db, f in zip(times, dbfs_values, freq_values)
        ]

        return {
            "duration_seconds": self.history_seconds,
            "sample_interval_seconds": self.block_duration,
            "points": points,
            "dbfs_axis": {"min": db_min, "max": db_max},
            "freq_axis": {"min": freq_min, "max": freq_max},
        }

    def get_graph_image(self, width: int = 800, height: int = 400) -> Optional[bytes]:
        """
        Render a PNG image of the last `history_seconds` of data.

        Returns:
            PNG bytes, or None if no data yet.

        The image shows:
          - Top subplot: dBFS over time (last N samples)
          - Bottom subplot: frequency (spectral centroid in Hz) over time
        """
        snapshot = self.get_graph_snapshot()
        if snapshot is None:
            return None

        points = snapshot["points"]
        if not points:
            return None

        t_vals = [p["t"] for p in points]
        db_vals = [p["dbfs"] for p in points]
        freq_vals = [p["freq_hz"] for p in points]

        db_axis = snapshot["dbfs_axis"]
        freq_axis = snapshot["freq_axis"]

        dpi = 100
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(width / dpi, height / dpi),
            dpi=dpi,
            sharex=True
        )

        # Top: dBFS
        ax1.plot(t_vals, db_vals, linewidth=1.0)
        ax1.set_ylabel("dBFS")
        ax1.set_ylim(db_axis["min"], db_axis["max"])
        ax1.set_title(f"Last {snapshot['duration_seconds']:.1f}s – Loudness (dBFS)")
        ax1.grid(True, linestyle=":", linewidth=0.5)

        # Bottom: Frequency
        ax2.plot(t_vals, freq_vals, linewidth=1.0)
        ax2.set_ylabel("Hz")
        ax2.set_xlabel("Time (s, relative to now)")
        ax2.set_ylim(freq_axis["min"], freq_axis["max"])
        ax2.set_title(f"Last {snapshot['duration_seconds']:.1f}s – Spectral Centroid")
        ax2.grid(True, linestyle=":", linewidth=0.5)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


# -------------------------------------------------------------
# Example usage when run directly (manual test)
# -------------------------------------------------------------
if __name__ == "__main__":
    print("MicDBFFrequencyMonitor demo\n")
    list_input_devices()

    choice = input("Enter device index to use (Enter = default): ").strip()
    device_idx = int(choice) if choice else None

    monitor = MicDBFFrequencyMonitor(
        device_index=device_idx,
        sample_rate=22050,
        block_duration=0.01,
        history_seconds=5.0,
    )

    monitor.start()
    print("Capturing... press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1.0)
            snap = monitor.get_graph_snapshot()
            if snap is None or not snap["points"]:
                print("No data yet.")
            else:
                last = snap["points"][-1]
                print(
                    f"Latest: t={last['t']:.2f}s, "
                    f"dBFS={last['dbfs']:.1f}, "
                    f"freq={last['freq_hz']:.1f} Hz"
                )
    except KeyboardInterrupt:
        print("\nStopping...")
        monitor.stop()
        print("Done.")
