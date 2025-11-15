"""
Mic dBFS + FFT Frequency Monitor (with live graph)

Features:
- 0.01 second updates (100 Hz)
- FFT smoothing to reduce peaks
- dBFS loudness measurement
- Spectral centroid computed from FFT magnitude spectrum
- Sliding window of recent data
- Live matplotlib graph (optional)
"""

import time
from datetime import datetime
from typing import List, Tuple, Optional
from collections import deque

import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -------------------------------------------------------------
# Device listing helper
# -------------------------------------------------------------

def list_input_devices():
    devices = sd.query_devices()
    print("\nAvailable audio input devices:")
    print("=" * 60)
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"[{i}] {d['name']}")
    print("=" * 60)


# -------------------------------------------------------------
# Core class: MicDBFFrequencyMonitor
# -------------------------------------------------------------

class MicDBFFrequencyMonitor:
    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 22050,
        block_duration: float = 0.01,   # 100 updates per second
        history_seconds: float = 5.0
    ):
        """
        0.01s block = 100Hz update rate.
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.block_duration = block_duration

        self.block_size = int(sample_rate * block_duration)
        self.max_blocks = int(history_seconds / block_duration)

        self._history: deque[Tuple[str, float, float]] = deque(
            maxlen=self.max_blocks
        )

        self._stream = None
        self._running = False

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    def _compute_dbfs(self, audio: np.ndarray) -> float:
        if len(audio) == 0:
            return -80.0

        rms = np.sqrt(np.mean(audio ** 2))
        if rms <= 1e-12:
            return -80.0

        return float(20 * np.log10(rms))

    def _compute_frequency_fft(self, audio: np.ndarray) -> float:
        """
        Spectral centroid computed AFTER FFT smoothing.
        """
        if len(audio) == 0:
            return 0.0

        # FFT computation
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)

        # Smoothing to reduce peaks (moving average)
        window = 5
        if len(spectrum) > window:
            spectrum = np.convolve(spectrum, np.ones(window) / window, mode="same")

        # Weighted centroid
        if spectrum.sum() == 0:
            return 0.0

        centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
        return centroid

    # ---------------------------------------------------------
    # Audio callback
    # ---------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)

        audio = indata[:, 0].astype(np.float32)

        dbfs = self._compute_dbfs(audio)
        freq = self._compute_frequency_fft(audio)
        ts = datetime.now().isoformat()

        self._history.append((ts, dbfs, freq))

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def start(self):
        if self._running:
            print("Already running.")
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
            f"Mic started (device={self.device_index}, block={self.block_duration}s, "
            f"sample_rate={self.sample_rate})"
        )

    def stop(self):
        if not self._running:
            return
        self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()

        print("Mic stopped.")

    def get_recent_data(self) -> List[Tuple[str, float, float]]:
        return list(self._history)

    def clear_history(self):
        self._history.clear()

    # ---------------------------------------------------------
    # Live Graph
    # ---------------------------------------------------------

    def plot_live(self):
        """
        Shows a live graph of dBFS and frequency centroid.
        """
        plt.style.use("ggplot")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        ax1.set_title("Live dBFS (Loudness)")
        ax1.set_ylim(-80, 0)
        ax1.set_ylabel("dBFS")

        ax2.set_title("Live Frequency (FFT Spectral Centroid)")
        ax2.set_ylim(0, self.sample_rate // 2)
        ax2.set_ylabel("Hz")

        line1, = ax1.plot([], [], lw=2)
        line2, = ax2.plot([], [], lw=2)

        def update(frame):
            data = self.get_recent_data()

            if len(data) == 0:
                return line1, line2

            timestamps, db_vals, freq_vals = zip(*data)

            # X axis = last N samples (no timestamp needed)
            x = np.arange(len(db_vals))

            line1.set_data(x, db_vals)
            line2.set_data(x, freq_vals)

            ax1.set_xlim(0, len(db_vals))
            ax2.set_xlim(0, len(freq_vals))

            return line1, line2

        ani = FuncAnimation(fig, update, interval=50)
        plt.tight_layout()
        plt.show()


# -------------------------------------------------------------
# Standalone Test Runner
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Mic dBFS + FFT Frequency Monitor Demo")
    print("====================================\n")

    list_input_devices()

    choice = input("Enter device index (Enter = default): ").strip()
    device_idx = int(choice) if choice else None

    monitor = MicDBFFrequencyMonitor(
        device_index=device_idx,
        sample_rate=22050,
        block_duration=0.01,
        history_seconds=5.0
    )

    monitor.start()

    print("Capturing... live graph will open.")
    print("Close graph window or press Ctrl+C to stop.\n")

    try:
        monitor.plot_live()
    except KeyboardInterrupt:
        pass

    monitor.stop()
