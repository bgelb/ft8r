import numpy as np
from functools import lru_cache

import ldpc
from typing import Tuple, List, Dict

from utils import (
    RealSamples,
    ComplexSamples,
    COSTAS_SEQUENCE,
    TONE_SPACING_IN_HZ,
    COSTAS_START_OFFSET_SEC,
    FT8_SYMBOL_LENGTH_IN_SEC,
    FT8_SYMBOLS_PER_MESSAGE,
    LDPC_174_91_H,
    check_crc,
    decode77,
)

from search import find_candidates
from utils.prof import PROFILER
import os

# Symbol positions occupied by the three 7-symbol Costas sequences.
COSTAS_POSITIONS = list(range(7)) + list(range(36, 43)) + list(range(72, 79))


# Map tone indices to 3-bit values using Gray coding.
GRAY_MAP = [
    0b000,  # tone 0
    0b001,  # tone 1
    0b011,  # tone 2
    0b010,  # tone 3
    0b110,  # tone 4
    0b100,  # tone 5
    0b101,  # tone 6
    0b111,  # tone 7
]

# LDPC parity matrix copied from the WSJT-X source tree
# (see ``utils/ldpc_matrix.py``).  Embedding it here avoids relying on the
# WSJT-X archive at runtime.

_LDPC_DECODER = ldpc.BpOsdDecoder(
    LDPC_174_91_H,
    error_rate=0.1,
    input_vector_type="received_vector",
    osd_method="OSD_CS",
    osd_order=2,
)


# Note: For the FT8 (174,91) code used here, the transmitted codeword order
# matches the decoder/LDPC parity matrix column order. The first 91 bits are
# the 77 message bits followed by the 14 CRC bits.

# Number of FT8 tone spacings carried in the downsampled slice (eight active
# tones plus one spacing of transition band on either side)
SLICE_SPAN_TONES = 10
# Width of the narrow band extracted around each candidate (Hz)
SLICE_BANDWIDTH_HZ = SLICE_SPAN_TONES * TONE_SPACING_IN_HZ
# Number of seconds included in the FFTs used for downsampling
FFT_DURATION_SEC = 16
# Target sample rate for the downsampled baseband signal (Hz)
BASEBAND_RATE_HZ = 200
# Lengths of the forward and inverse FFTs
_FFT_SLICE_LEN = int(BASEBAND_RATE_HZ * FFT_DURATION_SEC)
_EDGE_TAPER_LEN = 101
# Precompute fixed edge taper window used for band slicing
_EDGE_TAPER = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, _EDGE_TAPER_LEN))

# Early rejection by average |LLR| was experimentally disabled and is removed
_MIN_LLR_AVG = 0.0
_ENV_MAX = os.getenv("FT8R_MAX_CANDIDATES", "").strip()
# Default cap chosen from empirical candidate distribution (p99≈1244, max≈1260)
# across bundled samples, with headroom. Set to 0 to disable capping.
_MAX_CANDIDATES = 1500 if _ENV_MAX == "" else int(_ENV_MAX)
# Allow bypassing CRC gating for troubleshooting only. Defaults to disabled.
_ALLOW_CRC_FAIL = os.getenv("FT8R_ALLOW_CRC_FAIL", "0") not in ("0", "", "false", "False")
# Offset of the band edges relative to ``freq`` expressed in tone spacings.
# ``freq`` corresponds to tone 0 so the bottom edge lies 1.5 tone spacings
# below it and the top edge is ``SLICE_SPAN_TONES - 1.5`` spacings above.
LOWER_EDGE_OFFSET_TONES = 1.5
UPPER_EDGE_OFFSET_TONES = SLICE_SPAN_TONES - LOWER_EDGE_OFFSET_TONES


def _prepare_full_fft(samples_in: RealSamples):
    sample_rate = samples_in.sample_rate_in_hz
    full_fft_len = int(sample_rate * FFT_DURATION_SEC)
    audio = samples_in.samples
    if len(audio) >= full_fft_len:
        audio = audio[:full_fft_len]
    else:
        audio = np.pad(audio, (0, full_fft_len - len(audio)))
    with PROFILER.section("baseband.rfft_full"):
        full_fft = np.fft.rfft(audio)
    bin_spacing_hz = sample_rate / full_fft_len
    return full_fft, bin_spacing_hz, full_fft_len




def downsample_to_baseband(
    samples_in: RealSamples,
    freq: float,
    *,
    precomputed_fft: tuple | None = None,
) -> ComplexSamples:
    """Extract a narrow band around ``freq`` and decimate to ``BASEBAND_RATE_HZ``.

    The returned audio contains ``SLICE_BANDWIDTH_HZ`` of spectrum centred on
    ``freq`` and is shifted so that ``freq`` is at DC. The result is sampled at
    :data:`BASEBAND_RATE_HZ`.
    """

    sample_rate = samples_in.sample_rate_in_hz
    if precomputed_fft is None:
        full_fft, bin_spacing_hz, full_fft_len = _prepare_full_fft(samples_in)
    else:
        full_fft, bin_spacing_hz, full_fft_len = precomputed_fft
    symbol_rate_hz = TONE_SPACING_IN_HZ

    bottom_freq = freq - LOWER_EDGE_OFFSET_TONES * symbol_rate_hz
    top_freq = freq + UPPER_EDGE_OFFSET_TONES * symbol_rate_hz

    start_bin = int(round(bottom_freq / bin_spacing_hz))
    end_bin = int(round(top_freq / bin_spacing_hz))

    slice_fft = np.zeros(_FFT_SLICE_LEN, dtype=complex)
    slice_bins = full_fft[start_bin:end_bin]
    slice_len = len(slice_bins)
    slice_fft[:slice_len] = slice_bins

    slice_fft[:_EDGE_TAPER_LEN] *= _EDGE_TAPER
    slice_fft[slice_len - _EDGE_TAPER_LEN : slice_len] *= _EDGE_TAPER[::-1]

    candidate_bin = int(round(freq / bin_spacing_hz))
    bin_shift = candidate_bin - start_bin
    slice_fft = np.roll(slice_fft, -bin_shift)

    with PROFILER.section("baseband.irfft_slice"):
        time_series = np.fft.ifft(slice_fft)
    time_series *= 1 / np.sqrt(full_fft_len * _FFT_SLICE_LEN)

    return ComplexSamples(time_series, sample_rate_in_hz=BASEBAND_RATE_HZ)


def _symbol_samples(sample_rate: float) -> int:
    """Return the number of samples per FT8 symbol for ``sample_rate``."""
    return int(round(sample_rate * FT8_SYMBOL_LENGTH_IN_SEC))


def _tone_bases(sample_rate: float, sym_len: int, freq_offset_hz: float = 0.0) -> np.ndarray:
    """Return a matrix of tone responses for one symbol."""
    time_idx = np.arange(sym_len) / sample_rate
    return np.exp(
        -2j
        * np.pi
        * (freq_offset_hz + np.arange(8) * TONE_SPACING_IN_HZ)[:, None]
        * time_idx
    )


@lru_cache(maxsize=None)
def _zero_offset_bases(sample_rate: int, sym_len: int) -> np.ndarray:
    """Cache zero-offset tone bases for given sample_rate and sym_len."""
    return _tone_bases(sample_rate, sym_len, 0.0)


def _symbol_matrix(samples: np.ndarray, start: int, sym_len: int) -> np.ndarray:
    """Return ``samples`` sliced and reshaped into a symbol matrix."""
    seg = samples[start : start + sym_len * FT8_SYMBOLS_PER_MESSAGE]
    return seg.reshape(FT8_SYMBOLS_PER_MESSAGE, sym_len)


def _costas_energy(
    samples: ComplexSamples, start: int, freq_offset_hz: float
) -> float:
    """Return the power in the Costas symbols at ``start`` with ``freq_offset_hz``."""

    sample_rate = samples.sample_rate_in_hz
    sym_len = _symbol_samples(sample_rate)
    seg = _symbol_matrix(samples.samples, start, sym_len)

    bases = _tone_bases(sample_rate, sym_len, freq_offset_hz)

    resp = np.abs(bases @ seg.T) ** 2

    tones = COSTAS_SEQUENCE * 3
    energy = resp[tones, COSTAS_POSITIONS].sum()
    return float(energy)


def _costas_energy_with_bases(
    samples: ComplexSamples, start: int, bases: np.ndarray
) -> float:
    sample_rate = samples.sample_rate_in_hz
    sym_len = _symbol_samples(sample_rate)
    seg = _symbol_matrix(samples.samples, start, sym_len)
    resp = np.abs(bases @ seg.T) ** 2
    tones = COSTAS_SEQUENCE * 3
    energy = resp[tones, COSTAS_POSITIONS].sum()
    return float(energy)


def fine_time_sync(samples: ComplexSamples, dt: float, search: int) -> float:
    """Return refined ``dt`` by maximizing Costas energy around ``dt``.

    Batched across the integer offset window using a sliding view to avoid
    rebuilding symbol matrices in Python.
    """

    sample_rate = samples.sample_rate_in_hz
    sym_len = _symbol_samples(sample_rate)
    base_start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))

    seg_len = sym_len * FT8_SYMBOLS_PER_MESSAGE

    # Determine valid offset range so windows stay in-bounds
    min_start = 0
    max_start = len(samples.samples) - seg_len
    start0 = base_start - search
    start1 = base_start + search
    v0 = max(start0, min_start)
    v1 = min(start1, max_start)
    if v1 < v0:
        return dt

    # Build a compact strided view over exactly the needed windows
    count = v1 - v0 + 1
    base = samples.samples[v0 : v0 + seg_len + count - 1]
    stride = base.strides[0]
    windows = np.lib.stride_tricks.as_strided(
        base, shape=(count, seg_len), strides=(stride, stride)
    )
    seg_all = windows.reshape(count, FT8_SYMBOLS_PER_MESSAGE, sym_len)

    bases = _zero_offset_bases(sample_rate, sym_len)  # (8, sym_len)

    with PROFILER.section("align.costas_energy_time"):
        # resp: (O, 8, 79) = (8, sym_len) x (O, 79, sym_len)
        resp = np.abs(np.einsum("ks,ons->okn", bases, seg_all)) ** 2
        tones = np.array(COSTAS_SEQUENCE * 3)
        pos = np.array(COSTAS_POSITIONS)
        energies = resp[:, tones, pos].sum(axis=1)  # (O,)

    # Find best within the valid window and map back to absolute offset
    best_local = int(np.argmax(energies))
    best_off = (v0 + best_local) - base_start

    # Sub-sample refinement using parabolic interpolation around the peak
    frac = 0.0
    if 0 < best_local < len(energies) - 1:
        y1, y2, y3 = energies[best_local - 1], energies[best_local], energies[best_local + 1]
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) > 1e-12:
            frac = 0.5 * (y1 - y3) / denom
            frac = float(np.clip(frac, -0.5, 0.5))

    return dt + (best_off + frac) / sample_rate


def fine_freq_sync(
    samples: ComplexSamples, dt: float, search_hz: float, step_hz: float
) -> float:
    """Return frequency offset maximizing Costas energy with sub-bin refinement."""

    sample_rate = samples.sample_rate_in_hz
    start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))
    freqs = np.arange(-search_hz, search_hz + step_hz / 2, step_hz)
    sym_len = _symbol_samples(sample_rate)
    time_idx = np.arange(sym_len) / sample_rate
    bases0 = _zero_offset_bases(sample_rate, sym_len)

    seg = _symbol_matrix(samples.samples, start, sym_len)

    with PROFILER.section("align.costas_energy_freq"):
        shifts = np.exp(-2j * np.pi * freqs[:, None] * time_idx[None, :])
        bases = bases0[None, :, :] * shifts[:, None, :]
        resp = np.abs(bases @ seg.T) ** 2
        pos_idx = np.array(COSTAS_POSITIONS)
        tone_idx = np.array(COSTAS_SEQUENCE * 3)
        energies = resp[:, tone_idx, pos_idx].sum(axis=1)

    best = int(np.argmax(energies))
    frac = 0.0
    if 0 < best < len(energies) - 1:
        y1, y2, y3 = energies[best - 1], energies[best], energies[best + 1]
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) > 1e-18:
            frac = 0.5 * (y1 - y3) / denom
            frac = float(np.clip(frac, -0.5, 0.5))

    return float(freqs[best] + frac * step_hz)


 


def fine_sync_candidate(
    samples_in: RealSamples, freq: float, dt: float, *, precomputed_fft: tuple | None = None
) -> Tuple[ComplexSamples, float, float]:
    """Return finely aligned baseband, ``dt`` and ``freq`` for ``samples_in``."""

    with PROFILER.section("align.downsample_pre"):
        bb = downsample_to_baseband(samples_in, freq, precomputed_fft=precomputed_fft)

    with PROFILER.section("align.fine_time"):
        dt = fine_time_sync(bb, dt, 10)
    # ``find_candidates`` only locates frequencies to the nearest FT8 tone
    # spacing (6.25 Hz).  Some of the test samples include signals that fall
    # roughly halfway between these coarse bins.  Increase the fine frequency
    # search span and resolution so such offsets can be recovered.
    with PROFILER.section("align.fine_freq"):
        df = fine_freq_sync(bb, dt, 5.0, 0.25)
    freq += df

    with PROFILER.section("align.downsample_post"):
        bb = downsample_to_baseband(samples_in, freq, precomputed_fft=precomputed_fft)

    with PROFILER.section("align.fine_time_post"):
        dt = fine_time_sync(bb, dt, 4)

    # One more narrow frequency refinement around the current estimate can help
    # on some signals, but it marginally hurts others. Keep the pipeline simple
    # and robust by skipping this optional pass here.

    sym_len = _symbol_samples(bb.sample_rate_in_hz)
    start = int(round((dt + COSTAS_START_OFFSET_SEC) * bb.sample_rate_in_hz))
    end = start + sym_len * FT8_SYMBOLS_PER_MESSAGE
    trimmed = bb.samples[start:end]

    return ComplexSamples(trimmed, bb.sample_rate_in_hz), dt, freq


 


def soft_demod(samples_in: ComplexSamples) -> np.ndarray:
    """Return log-likelihood ratios for each payload bit.

    ``samples_in`` must contain exactly the 79 FT8 symbols beginning with the
    first Costas sync symbol.  All residual frequency or time offsets should be
    removed before calling this function.  :func:`fine_sync_candidate` performs
    the required alignment steps.
    """

    samples = samples_in.samples
    sample_rate = samples_in.sample_rate_in_hz
    sym_len = _symbol_samples(sample_rate)
    start = 0
    bases = _zero_offset_bases(sample_rate, sym_len)

    # Arrange the input samples into one matrix containing every symbol.  Each
    # row corresponds to one symbol worth of data.  This allows the tone
    # responses for all symbols to be computed in a single matrix
    # multiplication.
    with PROFILER.section("demod.symbol_matrix"):
        seg = _symbol_matrix(samples, start, sym_len)

    # ``resp`` has shape ``(8, FT8_SYMBOLS_PER_MESSAGE)`` and contains the
    # magnitude response of each tone for every symbol.
    with PROFILER.section("demod.tone_response"):
        resp = np.abs(bases @ seg.T)

    # Remove the Costas symbols used for synchronization.
    payload_resp = np.delete(resp, COSTAS_POSITIONS, axis=1)

    # Normalize to per-symbol probabilities.
    with PROFILER.section("demod.normalize"):
        probs = payload_resp / payload_resp.sum(axis=0, keepdims=True)

    # Pre-build Gray-code bit masks to compute log-likelihood ratios with
    # broadcasting. ``gray_bits`` has shape ``(3, 8)`` where each row selects the
    # tones contributing a ``1`` for that bit position.
    gray_bits = np.array(
        [[(g >> (2 - b)) & 1 for g in GRAY_MAP] for b in range(3)], dtype=bool
    )

    mask = gray_bits[:, :, None]
    with PROFILER.section("demod.llr"):
        ones = np.where(mask, probs[None, :, :], 0).sum(axis=1)
        zeros = np.where(~mask, probs[None, :, :], 0).sum(axis=1)
        llrs = np.log(ones + 1e-12) - np.log(zeros + 1e-12)

    return llrs.T.ravel()


def naive_hard_decode(llrs: np.ndarray) -> str:
    """Return a hard-decision bitstring from log-likelihood ratios."""

    bits = ["1" if llr > 0 else "0" for llr in llrs]
    return "".join(bits)


def ldpc_decode(llrs: np.ndarray) -> str:
    """LDPC decode using soft information ``llrs``.

    Parameters
    ----------
    llrs:
        Log-likelihood ratios for each of the 174 coded bits where positive
        values favour ``1`` and negative values favour ``0``.
    """

    hard = (llrs > 0).astype(np.uint8)
    # Convert log-likelihood ratios to bit error probabilities.  ``update_channel_probs``
    # expects the probability of each received bit being flipped.  ``llrs`` encode the
    # log of ``p(1) / p(0)`` so ``1 / (1 + exp(abs(llr)))`` is the probability that the
    # hard decision is wrong.
    # Scaling the LLR magnitude reduces the effective bit error probability
    # which helps the belief propagation decoder converge on weak signals.
    # The factor 7.0 mirrors the reliability scaling used by WSJT-X.  It was
    # tuned empirically by trying a range of values and selecting the one that
    # best matched WSJT-X's decode rate on the sample set.
    error_prob = 1.0 / (np.exp(7.0 * np.abs(llrs)) + 1.0)
    with PROFILER.section("ldpc.update_channel_probs"):
        _LDPC_DECODER.update_channel_probs(error_prob)

    syndrome = (LDPC_174_91_H @ hard) % 2
    syndrome = syndrome.astype(np.uint8)
    with PROFILER.section("ldpc.decode"):
        err_est = _LDPC_DECODER.decode(syndrome)
    corrected = np.bitwise_xor(err_est.astype(np.uint8), hard)
    # Optional development-time parity assertion to verify decoder correctness
    if os.getenv("FT8R_ASSERT_PARITY", "0") not in ("0", "", "false", "False"):
        syn2 = (LDPC_174_91_H @ corrected) % 2
        if int(syn2.sum()) != 0:
            raise AssertionError("LDPC decode produced non-zero parity syndrome")
    bits = "".join("1" if b else "0" for b in corrected.astype(int))
    return bits


# No CRC-guided flipping: only structurally correct mapping is used.


def _dedup_decodes(records: List[Dict]) -> List[Dict]:
    """Return ``records`` with near-duplicate decodes removed.

    Messages with identical decoded text (and identical payload bits when
    available) that fall within one FT8 tone bin and one symbol period of each
    other are considered duplicates.  The decode with the highest average
    absolute LLR is kept to represent the group.
    """

    by_msg: Dict[tuple[str, str | None], List[Dict]] = {}
    for rec in records:
        key = (rec.get("message", ""), rec.get("bits"))
        by_msg.setdefault(key, []).append(rec)

    deduped: List[Dict] = []
    for recs in by_msg.values():
        recs.sort(key=lambda r: abs(r.get("llr", 0.0)), reverse=True)
        groups: List[List[Dict]] = []
        for r in recs:
            placed = False
            for g in groups:
                if all(
                    abs(r["freq"] - k["freq"]) <= TONE_SPACING_IN_HZ
                    and abs(r["dt"] - k["dt"]) <= FT8_SYMBOL_LENGTH_IN_SEC
                    for k in g
                ):
                    g.append(r)
                    placed = True
                    break
            if not placed:
                groups.append([r])
        for g in groups:
            deduped.append(g[0])

    for rec in deduped:
        rec.pop("llr", None)

    deduped.sort(key=lambda r: r["score"], reverse=True)
    return deduped


def decode_full_period(samples_in: RealSamples, threshold: float = 1.0, *, include_bits: bool = False):
    """Decode all FT8 signals present in ``samples_in``.

    The audio is searched for Costas sync peaks and every candidate above
    ``threshold`` is downsampled, synchronized and decoded.  Only candidates
    that pass the CRC check are returned.

    Parameters
    ----------
    samples_in:
        Audio samples for one 15 second FT8 cycle.
    threshold:
        Minimum Costas power ratio for a location to be considered.

    Returns
    -------
    List[Dict[str, float | str]]
        Each dictionary contains ``message`` (the decoded text), ``score``,
        ``freq`` and ``dt`` for one valid decode.
    """

    sample_rate = samples_in.sample_rate_in_hz

    # Compute search parameters matching those used in the test helpers.
    sym_len = int(sample_rate / TONE_SPACING_IN_HZ)
    # Extend the search range slightly above the 2.5 kHz used in earlier
    # versions so signals up to 3 kHz offset are considered.  Some of the
    # bundled sample WAVs contain transmissions above 2.5 kHz which were
    # previously ignored.
    max_freq_bin = int(3000 / TONE_SPACING_IN_HZ)
    # Search the entire audio span for possible start offsets.
    max_dt_samples = len(samples_in.samples) - int(sample_rate * COSTAS_START_OFFSET_SEC)
    max_dt_symbols = -(-max_dt_samples // sym_len)

    with PROFILER.section("search.find_candidates"):
        candidates = find_candidates(
            samples_in, max_freq_bin, max_dt_symbols, threshold=threshold
        )

    results = []
    # Precompute FFT of the full window once per period for reuse across candidates
    precomputed_fft = _prepare_full_fft(samples_in)
    if _MAX_CANDIDATES > 0:
        candidates = candidates[:_MAX_CANDIDATES]

    for score, dt, freq in candidates:
        start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))
        end = start + sym_len * FT8_SYMBOLS_PER_MESSAGE
        margin = int(round(10 * sample_rate / BASEBAND_RATE_HZ))
        if start - margin < 0 or end + margin > len(samples_in.samples):
            continue
        # Run refined alignment path
        try:
            with PROFILER.section("align.pipeline_refined"):
                bb, dt_f, freq_f = fine_sync_candidate(
                    samples_in, freq, dt, precomputed_fft=precomputed_fft
                )
            with PROFILER.section("demod.soft"):
                llrs = soft_demod(bb)
            # Compute mean |LLR| for diagnostics only
            mu0 = float(np.mean(np.abs(llrs)))
            # Try hard-decision CRC first
            hard_bits = naive_hard_decode(llrs)
            method = "hard"
            if check_crc(hard_bits):
                decoded_bits = hard_bits
            else:
                # Light df microsearch before LDPC: try hard-only small df nudges
                try:
                    micro_df_span = float(os.getenv("FT8R_MICRO_LIGHT_DF_SPAN", "1.0"))
                    micro_df_step = float(os.getenv("FT8R_MICRO_LIGHT_DF_STEP", "0.5"))
                except Exception:
                    micro_df_span, micro_df_step = 1.0, 0.5
                best_mu = mu0
                best_tuple = (bb, dt_f, freq_f, llrs)
                samples = bb.samples
                sr = bb.sample_rate_in_hz
                t_idx = np.arange(samples.shape[0]) / sr
                dfs = np.arange(-micro_df_span, micro_df_span + 1e-9, micro_df_step)
                hard_passed = False
                for df in dfs:
                    if abs(float(df)) < 1e-12:
                        continue
                    with PROFILER.section("align.freq_nudge"):
                        rot = np.exp(-2j * np.pi * float(df) * t_idx)
                        nudged = samples * rot
                        bb2 = ComplexSamples(nudged, sr)
                        dt2, freq2 = dt_f, freq_f + float(df)
                    with PROFILER.section("demod.soft"):
                        llrs2 = soft_demod(bb2)
                    mu2 = float(np.mean(np.abs(llrs2)))
                    # No early gate; only tracking best-mu for later LDPC
                    hard2 = naive_hard_decode(llrs2)
                    if check_crc(hard2):
                        decoded_bits = hard2
                        method = "hard"
                        bb, dt_f, freq_f, llrs = bb2, dt2, freq2, llrs2
                        hard_passed = True
                        break
                    if mu2 > best_mu:
                        best_mu = mu2
                        best_tuple = (bb2, dt2, freq2, llrs2)
                # Optional dt microsearch: adjust symbol window by small ± offsets
                if not hard_passed and os.getenv("FT8R_MICRO_DT_ENABLE", "1") not in ("0","","false","False"):
                    sym_len = _symbol_samples(sr)
                    # work on current best_tuple as base
                    bb_b, dt_b, freq_b, llrs_b = best_tuple
                    base_samples = bb_b.samples
                    # dt offsets in symbols
                    dt_sym_offs = np.arange(-1.0, 1.0 + 1e-9, 0.5)
                    best_dt_mu = float(np.mean(np.abs(llrs_b)))
                    best_dt_tuple = (bb_b, dt_b, freq_b, llrs_b)
                    for dto in dt_sym_offs:
                        if abs(dto) < 1e-12:
                            continue
                        # shift window by integer samples
                        shift = int(round(dto * sym_len))
                        if shift == 0:
                            continue
                        if shift > 0:
                            seg = np.pad(base_samples, (0, shift))[: base_samples.shape[0]]
                            seg = seg[shift:]
                        else:
                            sh = -shift
                            seg = np.pad(base_samples, (sh, 0))[0: base_samples.shape[0]]
                            seg = seg[: base_samples.shape[0]-sh]
                        if seg.shape[0] != base_samples.shape[0]:
                            continue
                        bb_dt = ComplexSamples(seg, sr)
                        llrs_dt = soft_demod(bb_dt)
                        mu_dt = float(np.mean(np.abs(llrs_dt)))
                        hard_dt = naive_hard_decode(llrs_dt)
                        if check_crc(hard_dt):
                            decoded_bits = hard_dt
                            method = "hard"
                            bb, dt_f, freq_f, llrs = bb_dt, dt_b + dto*FT8_SYMBOL_LENGTH_IN_SEC, freq_b, llrs_dt
                            hard_passed = True
                            break
                        if mu_dt > best_dt_mu:
                            best_dt_mu = mu_dt
                            best_dt_tuple = (bb_dt, dt_b + dto*FT8_SYMBOL_LENGTH_IN_SEC, freq_b, llrs_dt)
                    if not hard_passed and best_dt_mu > best_mu:
                        best_tuple = best_dt_tuple
                if not hard_passed:
                    bb_b, dt_b, freq_b, llrs_b = best_tuple
                    with PROFILER.section("ldpc.total"):
                        decoded_bits = ldpc_decode(llrs_b)
                    method = "ldpc"
                    bb, dt_f, freq_f, llrs = bb_b, dt_b, freq_b, llrs_b
            # Enforce CRC gating after LDPC/hard decision (after optional microsearch)
            if _ALLOW_CRC_FAIL or check_crc(decoded_bits):
                text = decode77(decoded_bits[:77])
                mu = float(np.mean(np.abs(llrs)))
                rec = {
                    "message": text,
                    "score": score,
                    "freq": freq_f,
                    "dt": dt_f,
                    "method": method,
                    "llr": mu,
                }
                if include_bits:
                    rec["bits"] = decoded_bits
                results.append(rec)
        except Exception:
            pass

    return _dedup_decodes(results)
