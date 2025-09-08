# Basic candidate search for FT8 Costas sequence
import numpy as np
from scipy.ndimage import maximum_filter, median_filter
from typing import List, Tuple

from utils import (
    RealSamples,
    COSTAS_SEQUENCE,
    TONE_SPACING_IN_HZ,
    COSTAS_START_OFFSET_SEC,
)
from utils.prof import PROFILER

# Default global cap for runtime candidate count (tunable via env).
# Back-to-basics default tuned via K sweep (short e2e saturates by ~1000).
DEFAULT_MAX_CANDIDATES = 1000


def _costas_active_noise_maps(fft_pwr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (active_map, noise_map) for the Costas correlation.

    Computes the Costas sync correlation using indexed gathers and sums over the
    sparse Costas kernels instead of dense 2‑D convolution. The result matches
    correlate2d(..., mode="valid") using the logical active/noise kernels.

    Parameters
    ----------
    fft_pwr:
        2‑D array of shape (num_rows, num_cols) with FFT magnitudes squared for
        the oversampled time/frequency grid.

    Returns
    -------
    (active_map, noise_map):
        Arrays of shape (R, B) where R is the number of valid starting rows for
        a 7‑symbol Costas window, and B is the number of base frequency columns.
    """

    num_rows, num_cols = fft_pwr.shape
    # Number of valid starting rows for a 7‑symbol Costas window
    rows_valid = num_rows - _KERNEL_TIME_LEN + 1
    if rows_valid <= 0:
        return (np.empty((0, 0)), np.empty((0, 0)))

    # Base (non‑oversampled) frequency columns we evaluate
    max_base_cols = (num_cols - _KERNEL_FREQ_LEN + 1) // FREQ_SEARCH_OVERSAMPLING_RATIO
    if max_base_cols <= 0:
        return (np.empty((rows_valid, 0)), np.empty((rows_valid, 0)))
    base_cols = np.arange(max_base_cols) * FREQ_SEARCH_OVERSAMPLING_RATIO  # (B,)

    # Gather the 7 time rows used by the Costas kernel
    time_offsets = np.array(
        [i * TIME_SEARCH_OVERSAMPLING_RATIO for i in range(len(COSTAS_SEQUENCE))]
    )  # (7,)
    row_idx = time_offsets[:, None] + np.arange(rows_valid)[None, :]  # (7, R)

    # Per‑row tone offsets in columns (oversampled)
    tone_offsets = np.array(COSTAS_SEQUENCE) * FREQ_SEARCH_OVERSAMPLING_RATIO  # (7,)

    active_accum = None
    noise_accum = None
    for s in range(len(COSTAS_SEQUENCE)):
        rows = fft_pwr[row_idx[s]]  # (R, num_cols)

        # Active tone bin for this symbol (one tone per row)
        cols_active = base_cols + tone_offsets[s]  # (B,)
        vals_active = rows[:, cols_active]  # (R, B)
        active_accum = vals_active if active_accum is None else (active_accum + vals_active)

        # Sum of all 8 tones for this symbol (bins spaced by oversampling ratio)
        cols_all = base_cols[None, :] + (
            np.arange(COSTAS_KERNEL_NUM_TONES)[:, None] * FREQ_SEARCH_OVERSAMPLING_RATIO
        )  # (8, B)
        vals_all = rows[:, cols_all]  # (R, 8, B)
        sum_all = vals_all.sum(axis=1)  # (R, B)
        vals_noise = sum_all - vals_active  # exclude active tone from noise sum
        noise_accum = vals_noise if noise_accum is None else (noise_accum + vals_noise)

    return active_accum, noise_accum

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


def candidate_score_map(
    samples_in: RealSamples,
    max_freq_bin: int,
    max_dt_in_symbols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Costas power ratio scores for all search offsets."""

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

    with PROFILER.section("search.sliding_windows"):
        segs = np.lib.stride_tricks.sliding_window_view(samples, sym_len)[::step]
        segs = segs[:num_ffts]
        segs = np.pad(segs, ((0, 0), (0, fft_len - sym_len)))
    with PROFILER.section("search.rfft_grid"):
        ffts = np.fft.rfft(segs, axis=1) / sym_len

    with PROFILER.section("search.fft_power"):
        fft_pwr = np.abs(ffts) ** 2

    # Optional local whitening/normalization to improve contrast in busy bands.
    # Enabled by default; disable via FT8R_WHITEN_ENABLE=0. Uses robust per-tile scaling.
    if os.getenv("FT8R_WHITEN_ENABLE", "1") not in ("0", "", "false", "False"):
        eps = float(os.getenv("FT8R_WHITEN_EPS", "1e-12"))
        alpha = float(os.getenv("FT8R_WHITEN_MAD_ALPHA", "3.0"))
        rows_per_sec = sample_rate / (sym_len // TIME_SEARCH_OVERSAMPLING_RATIO)
        default_dt_rows = max(1, int(round(0.5 * rows_per_sec)))
        default_df_bins = max(1, int(round(100.0 / TONE_SPACING_IN_HZ)))
        tile_dt = int(os.getenv("FT8R_WHITEN_TILE_DT", str(default_dt_rows)))
        tile_df = int(os.getenv("FT8R_WHITEN_TILE_FREQ", str(default_df_bins)))
        H, W = fft_pwr.shape
        Hp = ((H + tile_dt - 1) // tile_dt) * tile_dt
        Wp = ((W + tile_df - 1) // tile_df) * tile_df
        with PROFILER.section("search.whiten.tile.pad"):
            pad_h = Hp - H
            pad_w = Wp - W
            if pad_h or pad_w:
                fft_pad = np.pad(fft_pwr, ((0, pad_h), (0, pad_w)), mode="edge")
            else:
                fft_pad = fft_pwr
        with PROFILER.section("search.whiten.tile.block_median"):
            blocks = fft_pad.reshape(Hp // tile_dt, tile_dt, Wp // tile_df, tile_df)
            med_tiles = np.median(blocks, axis=(1, 3))  # (Hb, Wb)
            med_map = np.repeat(np.repeat(med_tiles, tile_dt, axis=0), tile_df, axis=1)
            med_map = med_map[:H, :W]
        with PROFILER.section("search.whiten.tile.block_mad"):
            dev_pad = np.abs(fft_pad - np.repeat(np.repeat(med_tiles, tile_dt, axis=0), tile_df, axis=1))
            dev_blocks = dev_pad.reshape(Hp // tile_dt, tile_dt, Wp // tile_df, tile_df)
            mad_tiles = np.median(dev_blocks, axis=(1, 3))
            mad_map = np.repeat(np.repeat(mad_tiles, tile_dt, axis=0), tile_df, axis=1)
            mad_map = mad_map[:H, :W]
        with PROFILER.section("search.whiten.tile.apply"):
            scale = med_map + alpha * mad_map + eps
            fft_pwr = fft_pwr / scale

    # Vectorized Costas correlation: sum target tone bins vs non-target bins
    with PROFILER.section("search.correlate"):
        active_map, noise_map = _costas_active_noise_maps(fft_pwr)
    with PROFILER.section("search.ratio"):
        scores_map = active_map / (noise_map + 1e-12)

    num_windows = scores_map.shape[0]
    max_dt_idx = min(max_dt_idx, num_windows - 1)
    # scores_map is already at base-bin resolution; do not divide by OSF
    max_base_bin = min(max_freq_bin, scores_map.shape[1] - 1)

    idx = np.array(offsets)[:, None] + np.arange(max_dt_idx + 1)
    valid = idx < num_windows
    idx = np.clip(idx, 0, num_windows - 1)

    active = active_map[idx] * valid[:, :, None]
    noise = noise_map[idx] * valid[:, :, None]
    active = active.sum(axis=0)
    noise = noise.sum(axis=0)

    num_blocks = valid.sum(axis=0)
    scores = (active / (noise + 1e-12)) * num_blocks[:, None]

    # scores_map is already at base-bin resolution in our vectorized path
    scores = scores[:, : (max_base_bin + 1)]

    dts = (
        np.arange(max_dt_idx + 1) * step / sample_rate - COSTAS_START_OFFSET_SEC
    )
    freqs = np.arange(max_base_bin + 1) * TONE_SPACING_IN_HZ

    return scores, dts, freqs


import os


    


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


 


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def default_candidate_budget() -> int:
    """Return the default candidate budget honoring ``FT8R_MAX_CANDIDATES``.

    ``0`` or negatives are treated as uncapped and map to ``DEFAULT_MAX_CANDIDATES``
    for safety in production paths; callers who truly need uncapped should pass
    an explicit large budget derived from their grid size.
    """
    try:
        b = int(os.getenv("FT8R_MAX_CANDIDATES", str(DEFAULT_MAX_CANDIDATES)))
    except Exception:
        b = DEFAULT_MAX_CANDIDATES
    return DEFAULT_MAX_CANDIDATES if b <= 0 else b


 


def budget_tile_candidates(
    scores: np.ndarray,
    dts: np.ndarray,
    freqs: np.ndarray,
    base_threshold: float,
    *,
    budget: int,
) -> List[Tuple[float, float, float]]:
    """Select coarse candidates from score grid.

    Modes:
    - FT8R_COARSE_MODE=topk: global Top-K by score (no threshold)
    - default: global Top-K among cells >= base_threshold
    """
    mode = os.getenv("FT8R_COARSE_MODE", "").strip().lower()
    if budget <= 0:
        budget = scores.size
    flat = scores.ravel()
    if mode == "topk":
        # Pure Top‑K, ignore threshold
        if budget < flat.size:
            idx_part = np.argpartition(flat, -budget)[-budget:]
            idx_sorted = idx_part[np.argsort(flat[idx_part])[::-1]]
        else:
            idx_sorted = np.argsort(flat)[::-1]
        h, w = scores.shape
        out: List[Tuple[float, float, float]] = []
        for idx in idx_sorted:
            i = int(idx // w); j = int(idx % w)
            out.append((float(scores[i, j]), float(dts[i]), float(freqs[j])))
        return out
    else:
        # Threshold‑gated Top‑K
        mask = scores >= base_threshold
        idxs = np.nonzero(mask.ravel())[0]
        if idxs.size == 0:
            return []
        if budget < idxs.size:
            part = idxs[np.argpartition(flat[idxs], -budget)[-budget:]]
            order = part[np.argsort(flat[part])[::-1]]
        else:
            order = idxs[np.argsort(flat[idxs])[::-1]]
        h, w = scores.shape
        out: List[Tuple[float, float, float]] = []
        for idx in order:
            i = int(idx // w); j = int(idx % w)
            out.append((float(scores[i, j]), float(dts[i]), float(freqs[j])))
        return out

def find_candidates(
    samples_in: RealSamples,
    max_freq_bin: int,
    max_dt_in_symbols: int,
    threshold: float,
) -> List[Tuple[float, float, float]]:
    """Search ``samples_in`` for possible Costas sync locations.

    Candidates correspond to local maxima in the Costas power ratio map above
    ``threshold``.

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
    scores, dts, freqs = candidate_score_map(
        samples_in,
        max_freq_bin,
        max_dt_in_symbols,
    )

    # Always use budgeted per-tile selection to control candidate counts
    B = default_candidate_budget()
    return budget_tile_candidates(scores, dts, freqs, threshold, budget=B)
