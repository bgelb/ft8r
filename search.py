# Basic candidate search for FT8 Costas sequence
import math
import numpy as np
from typing import List, Tuple

from utils import (
    RealSamples,
    COSTAS_SEQUENCE,
    TONE_SPACING_IN_HZ,
    COSTAS_START_OFFSET_SEC,
)


def dft_mag(
    samples: np.ndarray,
    sample_rate: int,
    start_dt_in_samples: int,
    freq_in_hz: float,
    length_in_samples: int,
) -> float:
    """Return DFT magnitude for ``length_in_samples`` samples starting at
    ``start_dt_in_samples``.

    The computation uses a vectorized dot product which is considerably
    faster than iterating over each sample in Python.
    """
    n = np.arange(length_in_samples)
    angle = -2j * math.pi * freq_in_hz / sample_rate
    segment = samples[start_dt_in_samples : start_dt_in_samples + length_in_samples]
    exps = np.exp(angle * n)
    return abs(np.dot(segment, exps))


def find_candidates(
    samples_in: RealSamples,
    freq_range_in_hz: List[float],
    dt_range_in_samples: List[int],
    threshold: float,
) -> List[Tuple[float, float, float]]:
    """Search ``samples_in`` for possible Costas sync locations.

    Parameters
    ----------
    samples_in:
        Audio samples with associated sample rate.
    freq_range_in_hz:
        List of base frequencies to search in Hz.
    dt_range_in_samples:
        List of sample offsets for candidate positions.
    threshold:
        Minimum accumulated DFT magnitude to be considered a candidate.

    Returns
    -------
    List[Tuple[float, float, float]]
        Tuples of ``(score, time_offset_sec, base_frequency)`` sorted by score.

    This implementation ignores some aspects of the WSJT-X search such as
    forward error correction metrics and uses a coarse DFT rather than an
    FFT-based correlator, so it is far less sensitive.  The ``time_offset_sec``
    returned for each candidate is adjusted by ``COSTAS_START_OFFSET_SEC`` so
    it lines up with the timestamp reported by WSJT-X.
    """
    samples = samples_in.samples
    sample_rate = samples_in.sample_rate_in_hz
    sym_len = int(round(sample_rate / TONE_SPACING_IN_HZ))
    results = []
    for start in dt_range_in_samples:
        if start + sym_len * len(COSTAS_SEQUENCE) > len(samples):
            continue
        for freq in freq_range_in_hz:
            score = 0.0
            for idx, tone in enumerate(COSTAS_SEQUENCE):
                f = freq + tone * TONE_SPACING_IN_HZ
                score += dft_mag(samples, sample_rate, start + idx * sym_len, f, sym_len)
            if score >= threshold:
                dt = start / sample_rate - COSTAS_START_OFFSET_SEC
                results.append((score, dt, freq))
    results.sort(reverse=True)
    return results
