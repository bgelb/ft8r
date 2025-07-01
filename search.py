# Basic candidate search for FT8 Costas sequence
import numpy as np
from typing import List, Tuple

from utils import (
    RealSamples,
    COSTAS_SEQUENCE,
    TONE_SPACING_IN_HZ,
    COSTAS_START_OFFSET_SEC,
)

# Number of FFT bins used per symbol bin. This controls the zero-padding
# applied to each symbol prior to the FFT and therefore the frequency
# resolution of the search.
FREQ_SEARCH_OVERSAMPLING_RATIO = 2

# COSTAS sequence indices scaled by the oversampling ratio.  These indices
# point to the FFT bins for each Costas tone when the FFT length is
# ``sym_len * FREQ_SEARCH_OVERSAMPLING_RATIO``.
EXPANDED_COSTAS_SEQUENCE = [
    tone * FREQ_SEARCH_OVERSAMPLING_RATIO for tone in COSTAS_SEQUENCE
]
def find_candidates(
    samples_in: RealSamples,
    max_freq_bin: int,
    max_dt_in_symbols: int,
    threshold: float,
) -> List[Tuple[float, float, float]]:
    """Search ``samples_in`` for possible Costas sync locations.

    Parameters
    ----------
    samples_in:
        Audio samples with associated sample rate.
    max_freq_bin:
        Highest base frequency bin to search, expressed in units of the
        FT8 tone spacing. All base frequencies from 0 to ``max_freq_bin`` are
        examined.
    max_dt_in_symbols:
        Maximum candidate start offset in whole symbols. The search only
        examines offsets that are aligned to the 1920-sample symbol width so
        a single set of FFTs can be reused for every candidate position.
    threshold:
        Minimum accumulated DFT magnitude to be considered a candidate.

    Returns
    -------
    List[Tuple[float, float, float]]
        Tuples of ``(score, time_offset_sec, base_frequency)`` sorted by score.

    This implementation ignores several aspects of the WSJT-X search such as
    forward error correction metrics.  The ``time_offset_sec`` returned for
    each candidate is adjusted by ``COSTAS_START_OFFSET_SEC`` so it lines up
    with the timestamp reported by WSJT-X.
    """
    samples = samples_in.samples
    sample_rate = samples_in.sample_rate_in_hz
    sym_len = int(round(sample_rate / TONE_SPACING_IN_HZ))
    fft_len = sym_len * FREQ_SEARCH_OVERSAMPLING_RATIO

    num_ffts = min(
        max_dt_in_symbols + len(COSTAS_SEQUENCE),
        (len(samples) - sym_len) // sym_len + 1,
    )

    ffts = []
    for idx in range(num_ffts):
        start = idx * sym_len
        seg = samples[start : start + sym_len]
        seg = np.pad(seg, (0, fft_len - sym_len))
        ffts.append(np.fft.rfft(seg) / sym_len)
    ffts = np.asarray(ffts)

    results = []
    for dt_sym in range(max_dt_in_symbols + 1):
        if dt_sym + len(COSTAS_SEQUENCE) > len(ffts):
            break
        for base_tone_bin in range(max_freq_bin + 1):
            score = 0.0
            base_expanded_tone_bin = base_tone_bin * FREQ_SEARCH_OVERSAMPLING_RATIO
            for sync_idx, expanded_sync_bin in enumerate(EXPANDED_COSTAS_SEQUENCE):
                bin_idx = base_expanded_tone_bin + expanded_sync_bin
                score += abs(ffts[dt_sym + sync_idx][bin_idx])
            if score >= threshold:
                dt = dt_sym * sym_len / sample_rate - COSTAS_START_OFFSET_SEC
                base_freq = base_tone_bin * TONE_SPACING_IN_HZ
                results.append((score, dt, base_freq))
    results.sort(reverse=True)
    return results
