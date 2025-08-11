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

# Optional early rejection threshold based on average |LLR|.
try:
    _MIN_LLR_AVG = float(os.getenv("FT8R_MIN_LLR_AVG", "0"))
except Exception:
    _MIN_LLR_AVG = 0.0
_ENV_MAX = os.getenv("FT8R_MAX_CANDIDATES", "").strip()
# Default cap chosen from empirical candidate distribution (p99≈1244, max≈1260)
# across bundled samples, with headroom. Set to 0 to disable capping.
_MAX_CANDIDATES = 1500 if _ENV_MAX == "" else int(_ENV_MAX)
_DISABLE_LEGACY = os.getenv("FT8R_DISABLE_LEGACY", "0") not in ("0", "", "false", "False")
_RUN_BOTH_ALIGNMENTS = os.getenv("FT8R_RUN_BOTH", "1") not in ("0", "false", "False")
# Successive interference cancellation controls (default off to avoid regressions)
try:
    _SIC_PASSES = int(os.getenv("FT8R_SIC_PASSES", "0"))
except Exception:
    _SIC_PASSES = 0
try:
    _SIC_ALPHA = float(os.getenv("FT8R_SIC_ALPHA", "0.7"))
except Exception:
    _SIC_ALPHA = 0.7
_REQUIRE_FINAL_CRC = True
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


# Map 3-bit Gray code value to tone index and back
_GRAY_TO_TONE = {GRAY_MAP[i]: i for i in range(8)}
_TONE_TO_GRAY = {i: GRAY_MAP[i] for i in range(8)}


def _bits_to_tone_sequence(bits_174: str) -> List[int]:
    """Return 79-length tone index sequence including Costas positions.

    The 174 coded bits are interpreted as 58 groups of 3 bits in time order
    across payload symbols (Costas removed). Each 3-bit group is Gray-decoded
    to a tone index 0..7.
    """
    if len(bits_174) != 174:
        raise ValueError("bits_174 must be 174 bits long")
    tones: List[int] = []
    payload_idx = 0
    costas_iter = iter(COSTAS_SEQUENCE * 3)
    for sym in range(FT8_SYMBOLS_PER_MESSAGE):
        if sym in COSTAS_POSITIONS:
            tones.append(next(costas_iter))
        else:
            b0 = 1 if bits_174[payload_idx + 0] == "1" else 0
            b1 = 1 if bits_174[payload_idx + 1] == "1" else 0
            b2 = 1 if bits_174[payload_idx + 2] == "1" else 0
            gray = (b0 << 2) | (b1 << 1) | b2
            tone = _GRAY_TO_TONE.get(gray, 0)
            tones.append(tone)
            payload_idx += 3
    return tones


def _estimate_symbol_phasors(bb: ComplexSamples, tone_seq: List[int]) -> np.ndarray:
    """Estimate complex phasor per symbol for a given tone sequence.

    For each symbol, project the symbol-length segment onto the complex basis
    vector for the selected tone. Returns an array of length 79 of complex
    coefficients whose product with the basis reconstructs the symbol.
    """
    sample_rate = bb.sample_rate_in_hz
    sym_len = _symbol_samples(sample_rate)
    seg = _symbol_matrix(bb.samples, 0, sym_len)
    bases = _zero_offset_bases(sample_rate, sym_len)
    phasors = np.zeros(FT8_SYMBOLS_PER_MESSAGE, dtype=complex)
    for s in range(FT8_SYMBOLS_PER_MESSAGE):
        tone = tone_seq[s]
        # Use conjugate dot product, normalize by symbol length for stability
        phasors[s] = np.vdot(bases[tone], seg[s]) / sym_len
    return phasors


def _synthesize_original_signal(
    samples_in: RealSamples,
    dt: float,
    freq_hz: float,
    tone_seq: List[int],
    phasors: np.ndarray,
    alpha: float,
) -> Tuple[int, np.ndarray]:
    """Return (start_index, signal) synthesized at original rate to subtract.

    The returned `signal` has length equal to 79 symbols at the original
    sample rate. It is scaled by `alpha`.
    """
    orig_sr = samples_in.sample_rate_in_hz
    sym_len_orig = _symbol_samples(orig_sr)
    start_idx = int(round((dt + COSTAS_START_OFFSET_SEC) * orig_sr))
    total_len = sym_len_orig * FT8_SYMBOLS_PER_MESSAGE
    out = np.zeros(total_len)
    # Precompute time vector for one symbol at original rate
    t = np.arange(sym_len_orig) / orig_sr
    for s in range(FT8_SYMBOLS_PER_MESSAGE):
        tone = tone_seq[s]
        # Complex exponential at absolute frequency
        omega = 2.0 * np.pi * (freq_hz + tone * TONE_SPACING_IN_HZ)
        carrier = np.cos(omega * t)  # in-phase component
        # Use phasor's angle to add quadrature if needed
        # phasor = a * (cos(phi) + j sin(phi))
        a = float(np.abs(phasors[s]))
        phi = float(np.angle(phasors[s]))
        # Real signal matching in-phase/quadrature via cos(phi) and sin(phi)
        synth = a * (np.cos(phi) * carrier - np.sin(phi) * np.sin(omega * t))
        out[s * sym_len_orig : (s + 1) * sym_len_orig] += alpha * synth
    return start_idx, out


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
    """Return refined ``dt`` by maximizing Costas energy around ``dt``."""

    sample_rate = samples.sample_rate_in_hz
    base_start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))

    offsets = range(-search, search + 1)
    bases = _zero_offset_bases(sample_rate, _symbol_samples(sample_rate))
    with PROFILER.section("align.costas_energy_time"):
        energies = [
            _costas_energy_with_bases(samples, base_start + off, bases) for off in offsets
        ]
    best = int(np.argmax(energies))
    best_off = offsets[best]

    # Sub-sample refinement using parabolic interpolation around the peak.
    frac = 0.0
    if 0 < best < len(energies) - 1:
        y1, y2, y3 = energies[best - 1], energies[best], energies[best + 1]
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) > 1e-12:
            frac = 0.5 * (y1 - y3) / denom
            # Limit to ±0.5 sample to avoid instability on flat tops
            frac = float(np.clip(frac, -0.5, 0.5))

    return dt + (best_off + frac) / sample_rate


def _fine_time_sync_integer(samples: ComplexSamples, dt: float, search: int) -> float:
    """Return refined ``dt`` using integer-sample peak only (legacy behavior)."""

    sample_rate = samples.sample_rate_in_hz
    base_start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))
    offsets = range(-search, search + 1)
    with PROFILER.section("align.costas_energy_time_legacy"):
        energies = [
            _costas_energy(samples, base_start + off, 0.0) for off in offsets
        ]
    best = int(np.argmax(energies))
    best_off = offsets[best]
    return dt + best_off / sample_rate


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
    with PROFILER.section("align.costas_energy_freq"):
        energies = []
        for f in freqs:
            shift = np.exp(-2j * np.pi * f * time_idx)
            bases = bases0 * shift[None, :]
            energies.append(_costas_energy_with_bases(samples, start, bases))
    best = int(np.argmax(energies))

    # Sub-bin refinement via parabolic interpolation around the peak.
    frac = 0.0
    if 0 < best < len(energies) - 1:
        y1, y2, y3 = energies[best - 1], energies[best], energies[best + 1]
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) > 1e-18:
            frac = 0.5 * (y1 - y3) / denom
            frac = float(np.clip(frac, -0.5, 0.5))

    return float(freqs[best] + frac * step_hz)


def _fine_freq_sync_maxbin(
    samples: ComplexSamples, dt: float, search_hz: float, step_hz: float
) -> float:
    """Return frequency offset using maximum bin only (legacy behavior)."""
    sample_rate = samples.sample_rate_in_hz
    start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))
    freqs = np.arange(-search_hz, search_hz + step_hz / 2, step_hz)
    sym_len = _symbol_samples(sample_rate)
    time_idx = np.arange(sym_len) / sample_rate
    bases0 = _zero_offset_bases(sample_rate, sym_len)
    with PROFILER.section("align.costas_energy_freq_legacy"):
        energies = []
        for f in freqs:
            shift = np.exp(-2j * np.pi * f * time_idx)
            bases = bases0 * shift[None, :]
            energies.append(_costas_energy_with_bases(samples, start, bases))
    best = int(np.argmax(energies))
    return float(freqs[best])


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


def _fine_sync_candidate_legacy(
    samples_in: RealSamples, freq: float, dt: float, *, precomputed_fft: tuple | None = None
) -> Tuple[ComplexSamples, float, float]:
    """Legacy alignment without sub-sample refinement.

    This mirrors the earlier implementation to act as a safe fallback for
    borderline signals where fractional interpolation occasionally harms
    demodulation.
    """

    bb = downsample_to_baseband(samples_in, freq, precomputed_fft=precomputed_fft)
    dt = _fine_time_sync_integer(bb, dt, 10)
    df = _fine_freq_sync_maxbin(bb, dt, 5.0, 0.25)
    freq += df
    bb = downsample_to_baseband(samples_in, freq, precomputed_fft=precomputed_fft)
    dt = _fine_time_sync_integer(bb, dt, 4)

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
    bits = "".join("1" if b else "0" for b in corrected.astype(int))
    return bits


def decode_full_period(samples_in: RealSamples, threshold: float = 1.0):
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

    results: List[Dict[str, float | str]] = []
    # Work on a residual copy for SIC passes
    residual = RealSamples(samples_in.samples.copy(), sample_rate)
    for sic_pass in range(0, max(1, _SIC_PASSES + 1)):
        with PROFILER.section("search.find_candidates"):
            candidates = find_candidates(
                residual, max_freq_bin, max_dt_symbols, threshold=threshold
            )
        if _MAX_CANDIDATES > 0:
            candidates = candidates[:_MAX_CANDIDATES]

        # Precompute FFT once per pass for all candidates
        precomputed_fft = _prepare_full_fft(residual)

        accepted_for_subtraction: List[Tuple[float, float, float, List[int], np.ndarray]] = []

        for score, dt, freq in candidates:
            start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))
            end = start + sym_len * FT8_SYMBOLS_PER_MESSAGE
            margin = int(round(10 * sample_rate / BASEBAND_RATE_HZ))
            if start - margin < 0 or end + margin > len(residual.samples):
                continue
            # Try refined path first; optionally also run legacy afterward
            decoded_any = False
            local_bb = None
            dt_f = dt
            freq_f = freq
            decoded_bits = None
            try:
                with PROFILER.section("align.pipeline_refined"):
                    local_bb, dt_f, freq_f = fine_sync_candidate(
                        residual, freq, dt, precomputed_fft=precomputed_fft
                    )
                with PROFILER.section("demod.soft"):
                    llrs = soft_demod(local_bb)
                if _MIN_LLR_AVG > 0.0 and float(np.mean(np.abs(llrs))) < _MIN_LLR_AVG:
                    raise RuntimeError("low_llr")
                hard_bits = naive_hard_decode(llrs)
                if check_crc(hard_bits):
                    decoded_bits = hard_bits
                else:
                    with PROFILER.section("ldpc.total"):
                        decoded_bits = ldpc_decode(llrs)
                    # Optional final CRC gate after LDPC; drop if invalid when enabled
                    if _REQUIRE_FINAL_CRC and not check_crc(decoded_bits):
                        raise RuntimeError("crc_fail")
                text = decode77(decoded_bits[:77])
                results.append({
                    "message": text,
                    "score": score,
                    "freq": freq_f,
                    "dt": dt_f,
                })
                decoded_any = True
            except Exception:
                decoded_any = False

            if not _DISABLE_LEGACY and (_RUN_BOTH_ALIGNMENTS or not decoded_any):
                try:
                    with PROFILER.section("align.pipeline_legacy"):
                        local_bb, dt_f, freq_f = _fine_sync_candidate_legacy(
                            residual, freq, dt, precomputed_fft=precomputed_fft
                        )
                    with PROFILER.section("demod.soft"):
                        llrs = soft_demod(local_bb)
                    if _MIN_LLR_AVG > 0.0 and float(np.mean(np.abs(llrs))) < _MIN_LLR_AVG:
                        raise RuntimeError("low_llr")
                    hard_bits = naive_hard_decode(llrs)
                    if check_crc(hard_bits):
                        decoded_bits = hard_bits
                    else:
                        with PROFILER.section("ldpc.total"):
                            decoded_bits = ldpc_decode(llrs)
                        if _REQUIRE_FINAL_CRC and not check_crc(decoded_bits):
                            raise RuntimeError("crc_fail")
                    text = decode77(decoded_bits[:77])
                    results.append({
                        "message": text,
                        "score": score,
                        "freq": freq_f,
                        "dt": dt_f,
                    })
                    decoded_any = True
                except Exception:
                    decoded_any = False

            # Queue for subtraction only if we successfully decoded and have bb
            if decoded_any and decoded_bits is not None and local_bb is not None and _SIC_PASSES > 0:
                tone_seq = _bits_to_tone_sequence(decoded_bits)
                phasors = _estimate_symbol_phasors(local_bb, tone_seq)
                accepted_for_subtraction.append((dt_f, freq_f, score, tone_seq, phasors))

        # After processing all candidates, subtract in batch and proceed to next pass
        if _SIC_PASSES > 0 and len(accepted_for_subtraction) > 0 and sic_pass < _SIC_PASSES:
            # Optionally limit number subtracted per pass by score (highest first)
            accepted_for_subtraction.sort(key=lambda x: x[2], reverse=True)
            residual_samples = residual.samples.copy()
            for dt_f, freq_f, _score, tone_seq, phasors in accepted_for_subtraction:
                start_idx, synth = _synthesize_original_signal(
                    residual, dt_f, freq_f, tone_seq, phasors, _SIC_ALPHA
                )
                end_idx = start_idx + len(synth)
                if start_idx < 0 or end_idx > len(residual_samples):
                    continue
                residual_samples[start_idx:end_idx] -= synth
            residual = RealSamples(residual_samples, sample_rate)

        # Stop if no SIC or no new candidates to subtract
        if _SIC_PASSES == 0 or len(accepted_for_subtraction) == 0:
            # Either we are not doing SIC or nothing new was decoded to cancel
            if sic_pass > 0:
                break
            # If zero passes configured, we only ran once
            if _SIC_PASSES == 0:
                break

    return results

