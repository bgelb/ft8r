# Basic candidate search for FT8 Costas sequence
import numpy as np
from scipy.signal import correlate2d
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

# Number of candidate start offsets evaluated per symbol period.  A value of
# ``4`` means each subsequent Costas sync position begins one quarter of a
# symbol after the previous one.
TIME_SEARCH_OVERSAMPLING_RATIO = 4

# COSTAS sequence indices scaled by the oversampling ratio.  These indices
# point to the FFT bins for each Costas tone when the FFT length is
# ``sym_len * FREQ_SEARCH_OVERSAMPLING_RATIO``.
EXPANDED_COSTAS_SEQUENCE = [
    tone * FREQ_SEARCH_OVERSAMPLING_RATIO for tone in COSTAS_SEQUENCE
]

# Convolution kernel used to locate the Costas sync sequence.  It accounts for
# both the frequency and time oversampling patterns so that candidate search can
# be implemented as a 2‑D cross-correlation against the FFT magnitude matrix.
_KERNEL_TIME_LEN = TIME_SEARCH_OVERSAMPLING_RATIO * (len(COSTAS_SEQUENCE) - 1) + 1
# Number of tone bins spanned by the Costas kernel in the frequency dimension.
COSTAS_KERNEL_NUM_TONES = 8
_KERNEL_FREQ_LEN = (
    COSTAS_KERNEL_NUM_TONES
    + (COSTAS_KERNEL_NUM_TONES - 1) * (FREQ_SEARCH_OVERSAMPLING_RATIO - 1)
)
_COSTAS_KERNEL = np.zeros((_KERNEL_TIME_LEN, _KERNEL_FREQ_LEN))
for sym_idx, bin_offset in enumerate(EXPANDED_COSTAS_SEQUENCE):
    _COSTAS_KERNEL[sym_idx * TIME_SEARCH_OVERSAMPLING_RATIO, bin_offset] = 1.0

# Kernel used to measure noise power in the unused Costas bins.
_NOISE_KERNEL = np.zeros_like(_COSTAS_KERNEL)
for sym_idx, bin_offset in enumerate(EXPANDED_COSTAS_SEQUENCE):
    row = sym_idx * TIME_SEARCH_OVERSAMPLING_RATIO
    for tone in range(COSTAS_KERNEL_NUM_TONES):
        col = tone * FREQ_SEARCH_OVERSAMPLING_RATIO
        if tone != COSTAS_SEQUENCE[sym_idx]:
            _NOISE_KERNEL[row, col] = 1.0

# Offsets of the three Costas sync blocks within an FT8 transmission.  Each
# value is the starting symbol index for one 7-symbol Costas sequence.  The
# offsets are measured relative to the nominal start of the transmission.
COSTAS_BLOCK_OFFSETS = [0, 36, 72]

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
        Minimum ratio of Costas power to noise power required for a
        position to be considered a candidate.

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
    step = sym_len // TIME_SEARCH_OVERSAMPLING_RATIO

    offsets = [off * TIME_SEARCH_OVERSAMPLING_RATIO for off in COSTAS_BLOCK_OFFSETS]
    max_offset = offsets[-1]

    max_dt_idx = max_dt_in_symbols * TIME_SEARCH_OVERSAMPLING_RATIO
    required_ffts = max_dt_idx + max_offset + _KERNEL_TIME_LEN
    available_ffts = (len(samples) - sym_len) // step + 1
    num_ffts = min(required_ffts, available_ffts)

    # Build a matrix of overlapping windows so all FFTs can be computed in one
    # vectorized operation. ``sliding_window_view`` creates ``sym_len`` sample
    # segments spaced ``step`` samples apart without copying the underlying
    # data.
    segs = np.lib.stride_tricks.sliding_window_view(samples, sym_len)[::step]
    segs = segs[:num_ffts]
    segs = np.pad(segs, ((0, 0), (0, fft_len - sym_len)))
    ffts = np.fft.rfft(segs, axis=1) / sym_len

    fft_pwr = np.abs(ffts) ** 2

    active_map = correlate2d(fft_pwr, _COSTAS_KERNEL, mode="valid")
    noise_map = correlate2d(fft_pwr, _NOISE_KERNEL, mode="valid")
    scores_map = active_map / (noise_map + 1e-12)

    num_windows = scores_map.shape[0]
    max_dt_idx = min(max_dt_idx, num_windows - 1)
    max_base_bin = min(
        max_freq_bin,
        (scores_map.shape[1] - 1) // FREQ_SEARCH_OVERSAMPLING_RATIO,
    )

    # Gather all offsets for the three Costas blocks at once. ``idx`` has shape
    # ``(3, max_dt_idx + 1)`` where each row corresponds to one Costas block
    # offset.  Rows that would fall past the end of ``active_map`` are clipped
    # and later ignored using ``valid``.
    idx = np.array(offsets)[:, None] + np.arange(max_dt_idx + 1)
    valid = idx < num_windows
    idx = np.clip(idx, 0, num_windows - 1)

    active = active_map[idx] * valid[:, :, None]
    noise = noise_map[idx] * valid[:, :, None]
    active = active.sum(axis=0)
    noise = noise.sum(axis=0)

    # ``num_blocks`` records how many Costas blocks contributed at each time
    # offset (1 to 3).
    num_blocks = valid.sum(axis=0)
    scores = (active / (noise + 1e-12)) * num_blocks[:, None]

    # Only evaluate base frequencies aligned to actual FT8 tones.
    scores = scores[:, : (max_base_bin + 1) * FREQ_SEARCH_OVERSAMPLING_RATIO + 1]
    scores = scores[:, ::FREQ_SEARCH_OVERSAMPLING_RATIO]

    dts = (
        np.arange(max_dt_idx + 1) * step / sample_rate - COSTAS_START_OFFSET_SEC
    )
    freqs = np.arange(max_base_bin + 1) * TONE_SPACING_IN_HZ

    mask = scores >= threshold
    dt_idx, freq_idx = np.nonzero(mask)
    results = [
        (float(scores[i, j]), float(dts[i]), float(freqs[j]))
        for i, j in zip(dt_idx, freq_idx)
    ]

    results.sort(reverse=True)
    return results
