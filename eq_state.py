"""Shared EQ state for the live mixer GUI and audio control layer.

This file owns the thread-safe band settings that the GUI edits and the
audio engine snapshots before sending new filter settings to the C runtime.
"""

import threading
import math

class EQState:
    """Thread-safe list of EQ bands for one mixer channel."""

    def __init__(self):
        # A simple version counter lets callers notice when settings changed.
        self.lock = threading.Lock()
        self.version = 0
        self.min_freq = 20.0
        self.max_freq = 20000.0
        self.min_gain = -18.0
        self.max_gain = 18.0

        self.bands = [
            {"freq": 100.0, "gain_db": 0.0, "q": 1.0, "type": "bell", "order": 2, "enabled": True, "original_freq": 100.0},
            {"freq": 1000.0, "gain_db": 0.0, "q": 1.0, "type": "bell", "order": 2, "enabled": True, "original_freq": 1000.0},
            {"freq": 8000.0, "gain_db": 0.0, "q": 1.0, "type": "bell", "order": 2, "enabled": True, "original_freq": 8000.0},
        ]

    def get_snapshot(self):
        """Return a safe copy of the current version and band list."""
        with self.lock:
            return self.version, [b.copy() for b in self.bands]

    def update_band(self, index, **kwargs):
        """Update one band and bump the version for rebuild-aware callers."""
        with self.lock:
            for k, v in kwargs.items():
                self.bands[index][k] = v
            self.version += 1

    def _new_band_freq(self):
        """Place a new band in the largest empty log-frequency gap."""
        if not self.bands:
            return math.sqrt(self.min_freq * self.max_freq)

        freqs = sorted(float(b["freq"]) for b in self.bands)
        candidates = [self.min_freq] + freqs + [self.max_freq]

        max_gap = -1.0
        best_freq = freqs[-1]
        for i in range(len(candidates) - 1):
            left = max(self.min_freq, candidates[i])
            right = min(self.max_freq, candidates[i + 1])
            if right <= left:
                continue
            gap = math.log10(right) - math.log10(left)
            if gap > max_gap:
                max_gap = gap
                best_freq = 10 ** ((math.log10(left) + math.log10(right)) / 2.0)

        return float(best_freq)

    def add_band(self):
        """Append a neutral bell band and return its index."""
        with self.lock:
            freq = self._new_band_freq()
            self.bands.append(
                {
                    "freq": freq,
                    "gain_db": 0.0,
                    "q": 1.0,
                    "type": "bell",
                    "order": 2,
                    "enabled": True,
                }
            )
            self.version += 1
            return len(self.bands) - 1

    def remove_band(self, index=None):
        """Remove a band, while keeping at least one editable band alive."""
        with self.lock:
            if len(self.bands) <= 1:
                return None

            if index is None:
                index = len(self.bands) - 1
            index = max(0, min(int(index), len(self.bands) - 1))

            self.bands.pop(index)
            self.version += 1
            return index
