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


 


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


 


def budget_tile_candidates(
    scores: np.ndarray,
    dts: np.ndarray,
    freqs: np.ndarray,
    base_threshold: float,
    *,
    budget: int,
) -> List[Tuple[float, float, float]]:
    """Adaptive per-tile thresholding with efficient quantile sweep.

    Functionally mirrors the original quantile-lowering loop but avoids
    recomputing quantiles and rescanning entries multiple times. For each tile
    we maintain a pointer into its descending score list and advance it as the
    joint threshold is lowered from q_start to q_min. A global 3x3 suppression
    mask enforces separation, and at most ``k`` entries are accepted per tile.
    """
    h, w = scores.shape
    td = max(1, _env_int("FT8R_COARSE_ADAPTIVE_TILE_DT", 8))
    tf = max(1, _env_int("FT8R_COARSE_ADAPTIVE_TILE_FREQ", 4))
    k = max(1, _env_int("FT8R_COARSE_ADAPTIVE_PER_TILE_K", 2))
    t_min = _env_float("FT8R_COARSE_ADAPTIVE_THRESH_MIN", 0.7)
    q_start = _env_float("FT8R_COARSE_ADAPTIVE_Q_START", 0.98)
    q_min = _env_float("FT8R_COARSE_ADAPTIVE_Q_MIN", 0.80)
    q_step = _env_float("FT8R_COARSE_ADAPTIVE_Q_STEP", 0.05)

    T_fixed = max(base_threshold, t_min)

    # Prepare per-tile sorted scores and coordinates
    tiles = []
    for i0 in range(0, h, td):
        i1 = min(i0 + td, h)
        for j0 in range(0, w, tf):
            j1 = min(j0 + tf, w)
            block = scores[i0:i1, j0:j1]
            if block.size == 0:
                continue
            flat = block.ravel()
            order = np.argsort(flat)[::-1]
            s_desc = flat[order]
            # Coordinates for each entry in s_desc
            width = (j1 - j0)
            di = order // width
            dj = order % width
            is_arr = i0 + di
            js_arr = j0 + dj
            neg_s = -s_desc  # for searchsorted on ascending
            tiles.append({
                "s": s_desc,
                "neg": neg_s,
                "is": is_arr,
                "js": js_arr,
                "ptr": 0,
                "taken": 0,
            })

    selected_mask = np.zeros_like(scores, dtype=bool)
    results: List[Tuple[float, float, float]] = []

    def count_ge(tile, thr_val: float) -> int:
        # number of entries with s >= thr_val using -s ascending
        return int(np.searchsorted(tile["neg"], -thr_val, side="right"))

    q = q_start
    while q >= q_min and len(results) < budget:
        for tile in tiles:
            if tile["taken"] >= k or len(results) >= budget:
                continue
            s_desc = tile["s"]
            n = s_desc.shape[0]
            if n == 0:
                continue
            # Quantile threshold from the tile's own distribution
            asc_idx = int(np.floor(q * (n - 1)))
            # value at quantile q of ascending array equals s_desc[n-1-asc_idx]
            q_th = float(s_desc[n - 1 - asc_idx])
            T_tile = max(T_fixed, q_th)
            limit = min(n, count_ge(tile, T_tile))
            # Admit new entries between ptr and limit
            ptr = tile["ptr"]
            while ptr < limit and tile["taken"] < k and len(results) < budget:
                i = int(tile["is"][ptr])
                j = int(tile["js"][ptr])
                s = float(s_desc[ptr])
                # Enforce global 3x3 non-overlap
                if not selected_mask[i, j]:
                    ii0 = max(0, i - 1)
                    ii1 = min(h, i + 2)
                    jj0 = max(0, j - 1)
                    jj1 = min(w, j + 2)
                    if not selected_mask[ii0:ii1, jj0:jj1].any():
                        selected_mask[i, j] = True
                        tile["taken"] += 1
                        results.append((s, float(dts[i]), float(freqs[j])))
                ptr += 1
            tile["ptr"] = ptr
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

    # Always use budgeted per-tile selection to control candidate counts
    try:
        B = int(os.getenv("FT8R_MAX_CANDIDATES", "1500"))
    except Exception:
        B = 1500
    return budget_tile_candidates(scores, dts, freqs, threshold, budget=B if B > 0 else 1500)
