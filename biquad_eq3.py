"""Python biquad helpers used for EQ preview curves and offline filtering.

The realtime mixer uses matching C code, while this module builds scipy SOS
filters so the GUI can draw accurate frequency responses.
"""

import math
import numpy as np
from scipy.signal import sosfilt


class Biquad:
    """One normalized second-order filter section."""

    def __init__(self, b0, b1, b2, a0, a1, a2):
        self.set_coefficients(b0, b1, b2, a0, a1, a2)
        self.reset()

    def set_coefficients(self, b0, b1, b2, a0, a1, a2):
        """Normalize coefficients so scipy can process them as one SOS row."""
        b0_n = b0 / a0
        b1_n = b1 / a0
        b2_n = b2 / a0
        a1_n = a1 / a0
        a2_n = a2 / a0

        # One biquad represented as one SOS row: [b0, b1, b2, a0, a1, a2].
        self.sos = np.array([[b0_n, b1_n, b2_n, 1.0, a1_n, a2_n]], dtype=np.float32)

    def reset(self):
        """Clear the delay state for a fresh stream."""
        self.zi = np.zeros((1, 2), dtype=np.float32)

    def process_block(self, x_block):
        """Process one block while preserving filter history."""
        x = np.asarray(x_block, dtype=np.float32)
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y


class SOSFilter:
    """General cascade wrapper for already-built SOS rows."""

    def __init__(self, sos):
        self.sos = np.asarray(sos, dtype=np.float32)
        self.reset()

    def reset(self):
        """Clear one delay row per SOS section."""
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float32)

    def process_block(self, x_block):
        """Run a block through the full SOS cascade."""
        x = np.asarray(x_block, dtype=np.float32)
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y


class EQChain:
    """A complete channel EQ made from zero or more biquad filters."""

    def __init__(self, filters=None):
        # Flatten individual filters into one SOS matrix for efficient filtering.
        self.filters = filters if filters is not None else []
        if self.filters:
            self.sos = np.vstack([filt.sos for filt in self.filters])
            self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float32)
        else:
            self.sos = None
            self.zi = None

    def reset(self):
        """Reset all filter memory in the chain."""
        if self.sos is not None:
            self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float32)

    def process_block(self, x_block):
        """Return input unchanged when no filters are active."""
        if self.sos is None:
            return np.asarray(x_block, dtype=np.float32)
        x = np.asarray(x_block, dtype=np.float32)
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y


def make_peaking_eq(fs, f0, q, gain_db):
    """Create a bell/peaking EQ filter using RBJ cookbook coefficients."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    return Biquad(b0, b1, b2, a0, a1, a2)


def make_lowpass(fs, f0, q=0.707):
    """Create a second-order low-pass section."""
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)

    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    return Biquad(b0, b1, b2, a0, a1, a2)


def make_highpass(fs, f0, q=0.707):
    """Create a second-order high-pass section."""
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)

    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    return Biquad(b0, b1, b2, a0, a1, a2)


def make_low_shelf(fs, f0, gain_db, slope=1.0):
    """Create a low-shelf EQ section."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)

    alpha = sin_w0 / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    two_sqrtA_alpha = 2.0 * math.sqrt(A) * alpha

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrtA_alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrtA_alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrtA_alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrtA_alpha

    return Biquad(b0, b1, b2, a0, a1, a2)


def make_high_shelf(fs, f0, gain_db, slope=1.0):
    """Create a high-shelf EQ section."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)

    alpha = sin_w0 / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    two_sqrtA_alpha = 2.0 * math.sqrt(A) * alpha

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrtA_alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrtA_alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrtA_alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrtA_alpha

    return Biquad(b0, b1, b2, a0, a1, a2)


def butterworth_section_qs(order):
    """Return Q values for an even-order Butterworth cascade."""
    order = max(2, int(order))
    if order % 2 != 0:
        order += 1

    sections = order // 2
    qs = []
    for section_index in range(1, sections + 1):
        q = 1.0 / (2.0 * math.sin(((2 * section_index) - 1) * math.pi / (2.0 * order)))
        qs.append(q)

    return qs

def build_eq_chain(fs, bands):
    """Convert GUI band dictionaries into an EQChain."""
    filters = []
    for band in bands:
        if band.get("enabled", True) and band.get("type", "bell") == "bell":
            filters.append(
                make_peaking_eq(
                    fs,
                    f0=float(band["freq"]),
                    q=max(0.1, float(band["q"])),
                    gain_db=float(band["gain_db"]),
                )
            )
        elif band.get("enabled", True) and band.get("type", "bell") == "low_shelf":
            filters.append(
                make_low_shelf(
                    fs,
                    f0=float(band["freq"]),
                    gain_db=float(band["gain_db"]),
                    slope=1.0,
                )
            )

        elif band.get("enabled", True) and band.get("type", "bell") == "high_shelf":
            filters.append(
                make_high_shelf(
                    fs,
                    f0=float(band["freq"]),
                    gain_db=float(band["gain_db"]),
                    slope=1.0,
                )
            )
        elif band.get("enabled", True) and band.get("type", "bell") in ("lowpass", "lp"):
            order = int(band.get("order", 2))
            cutoff = float(band["freq"])
            for q in butterworth_section_qs(order):
                filters.append(make_lowpass(fs, f0=cutoff, q=q))
        elif band.get("enabled", True) and band.get("type", "bell") in ("highpass", "hp"):
            order = int(band.get("order", 2))
            cutoff = float(band["freq"])
            for q in butterworth_section_qs(order):
                filters.append(make_highpass(fs, f0=cutoff, q=q))
    return EQChain(filters)
