# Basic candidate search for FT8 Costas sequence
import numpy as np
from scipy.ndimage import maximum_filter
from typing import List, Tuple

from utils import (
    RealSamples,
    COSTAS_SEQUENCE,
    TONE_SPACING_IN_HZ,
    COSTAS_START_OFFSET_SEC,
)
from utils.prof import PROFILER


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
    if os.getenv("FT8R_WHITEN_ENABLE", "0") not in ("0", "", "false", "False"):
        eps = float(os.getenv("FT8R_WHITEN_EPS", "1e-12"))
        mode = os.getenv("FT8R_WHITEN_MODE", "global").strip().lower()
        if mode == "tile":
            # Tile-based robust scaling: divide by (median + alpha*MAD) per tile.
            alpha = float(os.getenv("FT8R_WHITEN_MAD_ALPHA", "3.0"))
            # Derive reasonable defaults for tile sizes if not specified.
            # Time tile default: ~0.5s worth of rows.
            rows_per_sec = sample_rate / (sym_len // TIME_SEARCH_OVERSAMPLING_RATIO)
            default_dt_rows = max(1, int(round(0.5 * rows_per_sec)))
            default_df_bins = max(1, int(round(100.0 / TONE_SPACING_IN_HZ)))  # ~100 Hz
            tile_dt = int(os.getenv("FT8R_WHITEN_TILE_DT", str(default_dt_rows)))
            tile_df = int(os.getenv("FT8R_WHITEN_TILE_FREQ", str(default_df_bins)))
            H, W = fft_pwr.shape
            with PROFILER.section("search.whiten.tile"):
                for i0 in range(0, H, tile_dt):
                    i1 = min(i0 + tile_dt, H)
                    blk = fft_pwr[i0:i1]
                    for j0 in range(0, W, tile_df):
                        j1 = min(j0 + tile_df, W)
                        sub = blk[:, j0:j1]
                        med = np.median(sub)
                        mad = np.median(np.abs(sub - med))
                        scale = med + alpha * mad + eps
                        fft_pwr[i0:i1, j0:j1] = sub / scale
        else:
            with PROFILER.section("search.whiten.freq"):
                col_med = np.median(fft_pwr, axis=0)
                col_med = np.maximum(col_med, eps)
                fft_pwr = fft_pwr / col_med
            with PROFILER.section("search.whiten.time"):
                row_med = np.median(fft_pwr, axis=1, keepdims=True)
                row_med = np.maximum(row_med, eps)
                fft_pwr = fft_pwr / row_med

    # Vectorized Costas correlation: sum target tone bins vs non-target bins
    with PROFILER.section("search.correlate"):
        active_map, noise_map = _costas_active_noise_maps(fft_pwr)
    with PROFILER.section("search.ratio"):
        scores_map = active_map / (noise_map + 1e-12)

    num_windows = scores_map.shape[0]
    max_dt_idx = min(max_dt_idx, num_windows - 1)
    max_base_bin = min(
        max_freq_bin,
        (scores_map.shape[1] - 1) // FREQ_SEARCH_OVERSAMPLING_RATIO,
    )

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


def peak_candidates(
    scores: np.ndarray,
    dts: np.ndarray,
    freqs: np.ndarray,
    threshold: float,
) -> List[Tuple[float, float, float]]:
    """Return local maxima from ``scores`` above ``threshold``."""

    # Max value including the center.
    neighborhood = np.ones((3, 3), dtype=bool)
    with PROFILER.section("search.max_full"):
        max_full = maximum_filter(scores, footprint=neighborhood, mode="constant", cval=-np.inf)
    # Max value of neighbors excluding the center element.
    neighbor_foot = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    with PROFILER.section("search.max_neighbors"):
        max_neighbors = maximum_filter(scores, footprint=neighbor_foot, mode="constant", cval=-np.inf)

    mask = (scores >= threshold) & (scores == max_full) & (scores > max_neighbors)
    with PROFILER.section("search.nonzero"):
        dt_idx, freq_idx = np.nonzero(mask)
    results = [
        (float(scores[i, j]), float(dts[i]), float(freqs[j]))
        for i, j in zip(dt_idx, freq_idx)
    ]
    results.sort(reverse=True)
    return results


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


## Removed experimental tile_topk in favor of budgeted per-tile selection.


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


## Removed experimental adaptive replacement path in favor of budgeted selection.


def budget_tile_candidates(
    scores: np.ndarray,
    dts: np.ndarray,
    freqs: np.ndarray,
    base_threshold: float,
    *,
    budget: int,
) -> List[Tuple[float, float, float]]:
    """Adaptive per-tile thresholding to a global budget.

    Lowers per-tile thresholds together from a high quantile to a lower bound,
    adding up to K per tile, until reaching the global candidate budget.
    """
    h, w = scores.shape
    td = max(1, _env_int("FT8R_COARSE_ADAPTIVE_TILE_DT", 8))
    tf = max(1, _env_int("FT8R_COARSE_ADAPTIVE_TILE_FREQ", 4))
    k = max(1, _env_int("FT8R_COARSE_ADAPTIVE_PER_TILE_K", 2))
    t_min = _env_float("FT8R_COARSE_ADAPTIVE_THRESH_MIN", 0.7)
    q_start = _env_float("FT8R_COARSE_ADAPTIVE_Q_START", 0.98)
    q_min = _env_float("FT8R_COARSE_ADAPTIVE_Q_MIN", 0.80)
    q_step = _env_float("FT8R_COARSE_ADAPTIVE_Q_STEP", 0.05)

    # Prepare sorted indices per tile
    tiles = []
    for i0 in range(0, h, td):
        i1 = min(i0 + td, h)
        for j0 in range(0, w, tf):
            j1 = min(j0 + tf, w)
            block = scores[i0:i1, j0:j1]
            if block.size == 0:
                continue
            flat_scores = block.ravel()
            order = np.argsort(flat_scores)[::-1]
            tiles.append((i0, i1, j0, j1, block, flat_scores, order))

    selected_mask = np.zeros_like(scores, dtype=bool)
    results: List[Tuple[float, float, float]] = []
    selected_per_tile = [0] * len(tiles)

    q = q_start
    while q >= q_min and len(results) < budget:
        for idx, (i0, i1, j0, j1, block, flat_scores, order) in enumerate(tiles):
            if selected_per_tile[idx] >= k or len(results) >= budget:
                continue
            # Compute tile threshold at current quantile
            q_th = float(np.quantile(block, q))
            T_tile = max(base_threshold, q_th, t_min)
            # Try to add highest-scoring points above T_tile
            for fid in order:
                if selected_per_tile[idx] >= k or len(results) >= budget:
                    break
                s = float(flat_scores[fid])
                if s < T_tile:
                    break
                di = fid // (j1 - j0)
                dj = fid % (j1 - j0)
                i = i0 + di
                j = j0 + dj
                if selected_mask[i, j]:
                    continue
                # Enforce small 3x3 separation globally
                ii0 = max(0, i - 1)
                ii1 = min(h, i + 2)
                jj0 = max(0, j - 1)
                jj1 = min(w, j + 2)
                if selected_mask[ii0:ii1, jj0:jj1].any():
                    continue
                selected_mask[i, j] = True
                results.append((s, float(dts[i]), float(freqs[j])))
                selected_per_tile[idx] += 1
        q -= q_step

    results.sort(reverse=True)
    return results

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

    mode = os.getenv("FT8R_COARSE_MODE", "peak").strip().lower()
    if mode == "budget":
        # Target a global budget. Fall back to peak if budget invalid.
        try:
            B = int(os.getenv("FT8R_MAX_CANDIDATES", "1500"))
        except Exception:
            B = 1500
        return budget_tile_candidates(scores, dts, freqs, threshold, budget=B if B > 0 else 1500)
    return peak_candidates(scores, dts, freqs, threshold)
