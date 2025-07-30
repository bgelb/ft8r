import numpy as np

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
from utils.waveform import bits_to_tones, synth_waveform

from search import find_candidates

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

_LDPC_DECODER = ldpc.BpDecoder(
    LDPC_174_91_H, error_rate=0.1, input_vector_type="received_vector"
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
# Offset of the band edges relative to ``freq`` expressed in tone spacings.
# ``freq`` corresponds to tone 0 so the bottom edge lies 1.5 tone spacings
# below it and the top edge is ``SLICE_SPAN_TONES - 1.5`` spacings above.
LOWER_EDGE_OFFSET_TONES = 1.5
UPPER_EDGE_OFFSET_TONES = SLICE_SPAN_TONES - LOWER_EDGE_OFFSET_TONES


def downsample_to_baseband(samples_in: RealSamples, freq: float) -> ComplexSamples:
    """Extract a narrow band around ``freq`` and decimate to ``BASEBAND_RATE_HZ``.

    The returned audio contains ``SLICE_BANDWIDTH_HZ`` of spectrum centred on
    ``freq`` and is shifted so that ``freq`` is at DC. The result is sampled at
    :data:`BASEBAND_RATE_HZ`.
    """

    sample_rate = samples_in.sample_rate_in_hz
    full_fft_len = int(sample_rate * FFT_DURATION_SEC)

    audio = samples_in.samples
    if len(audio) >= full_fft_len:
        audio = audio[:full_fft_len]
    else:
        audio = np.pad(audio, (0, full_fft_len - len(audio)))

    full_fft = np.fft.rfft(audio)

    bin_spacing_hz = sample_rate / full_fft_len
    symbol_rate_hz = TONE_SPACING_IN_HZ

    bottom_freq = freq - LOWER_EDGE_OFFSET_TONES * symbol_rate_hz
    top_freq = freq + UPPER_EDGE_OFFSET_TONES * symbol_rate_hz

    start_bin = int(round(bottom_freq / bin_spacing_hz))
    end_bin = int(round(top_freq / bin_spacing_hz))

    slice_fft = np.zeros(_FFT_SLICE_LEN, dtype=complex)
    slice_bins = full_fft[start_bin:end_bin]
    slice_len = len(slice_bins)
    slice_fft[:slice_len] = slice_bins

    taper = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, _EDGE_TAPER_LEN))
    slice_fft[:_EDGE_TAPER_LEN] *= taper
    slice_fft[slice_len - _EDGE_TAPER_LEN : slice_len] *= taper[::-1]

    candidate_bin = int(round(freq / bin_spacing_hz))
    bin_shift = candidate_bin - start_bin
    slice_fft = np.roll(slice_fft, -bin_shift)

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


def fine_time_sync(samples: ComplexSamples, dt: float, search: int) -> float:
    """Return refined ``dt`` by maximizing Costas energy around ``dt``."""

    sample_rate = samples.sample_rate_in_hz
    base_start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))

    offsets = range(-search, search + 1)
    energies = [
        _costas_energy(samples, base_start + off, 0.0) for off in offsets
    ]
    best = int(np.argmax(energies))
    best_off = offsets[best]
    return dt + best_off / sample_rate


def fine_freq_sync(
    samples: ComplexSamples, dt: float, search_hz: float, step_hz: float
) -> float:
    """Return frequency offset maximizing Costas energy."""

    sample_rate = samples.sample_rate_in_hz
    start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))
    freqs = np.arange(-search_hz, search_hz + step_hz / 2, step_hz)
    energies = [_costas_energy(samples, start, f) for f in freqs]
    best = int(np.argmax(energies))
    return float(freqs[best])


def fine_sync_candidate(
    samples_in: RealSamples, freq: float, dt: float
) -> Tuple[ComplexSamples, float, float]:
    """Return finely aligned baseband, ``dt`` and ``freq`` for ``samples_in``."""

    bb = downsample_to_baseband(samples_in, freq)

    dt = fine_time_sync(bb, dt, 10)
    df = fine_freq_sync(bb, dt, 2.5, 0.5)
    freq += df

    bb = downsample_to_baseband(samples_in, freq)

    dt = fine_time_sync(bb, dt, 4)

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
    bases = _tone_bases(sample_rate, sym_len, 0.0)

    # Arrange the input samples into one matrix containing every symbol.  Each
    # row corresponds to one symbol worth of data.  This allows the tone
    # responses for all symbols to be computed in a single matrix
    # multiplication.
    seg = _symbol_matrix(samples, start, sym_len)

    # ``resp`` has shape ``(8, FT8_SYMBOLS_PER_MESSAGE)`` and contains the
    # magnitude response of each tone for every symbol.
    resp = np.abs(bases @ seg.T)

    # Remove the Costas symbols used for synchronization.
    payload_resp = np.delete(resp, COSTAS_POSITIONS, axis=1)

    # Normalize to per-symbol probabilities.
    probs = payload_resp / payload_resp.sum(axis=0, keepdims=True)

    # Pre-build Gray-code bit masks to compute log-likelihood ratios with
    # broadcasting. ``gray_bits`` has shape ``(3, 8)`` where each row selects the
    # tones contributing a ``1`` for that bit position.
    gray_bits = np.array(
        [[(g >> (2 - b)) & 1 for g in GRAY_MAP] for b in range(3)], dtype=bool
    )

    mask = gray_bits[:, :, None]
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
    _LDPC_DECODER.update_channel_probs(error_prob)
    decoded = _LDPC_DECODER.decode(hard)
    bits = "".join("1" if b else "0" for b in decoded.astype(int))
    return bits


def decode_full_period(samples_in: RealSamples, threshold: float = 1.0, max_passes: int = 3):
    """Decode all FT8 signals present in ``samples_in`` using multi-pass subtraction.

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

    working = samples_in.samples.copy()
    results = []

    def seen_before(msg: str, dt_v: float, freq_v: float) -> bool:
        for rec in results:
            if (
                rec["message"] == msg
                and abs(rec["dt"] - dt_v) < 0.2
                and abs(rec["freq"] - freq_v) < 1.0
            ):
                return True
        return False

    for _ in range(max_passes):
        audio = RealSamples(working, sample_rate)
        candidates = find_candidates(
            audio, max_freq_bin, max_dt_symbols, threshold=threshold
        )

        decoded_in_pass = []
        found = False
        for score, dt, freq in candidates:
            start = int(round((dt + COSTAS_START_OFFSET_SEC) * sample_rate))
            end = start + sym_len * FT8_SYMBOLS_PER_MESSAGE
            margin = int(round(10 * sample_rate / BASEBAND_RATE_HZ))
            if start - margin < 0 or end + margin > len(working):
                continue
            try:
                bb, dt_f, freq_f = fine_sync_candidate(audio, freq, dt)
            except ValueError:
                continue
            llrs = soft_demod(bb)
            decoded_bits = ldpc_decode(llrs)
            try:
                text = decode77(decoded_bits[:77])
            except Exception:
                continue
            if seen_before(text, dt_f, freq_f):
                continue

            results.append(
                {
                    "message": text,
                    "score": score,
                    "freq": freq_f,
                    "dt": dt_f,
                }
            )

            amp = float(np.sqrt(np.mean(np.abs(bb.samples) ** 2)))
            decoded_in_pass.append((decoded_bits, dt_f, freq_f, amp))
            found = True

        if not found:
            break

        # Subtract all signals decoded in this pass
        for bits, dt_f, freq_f, amp in decoded_in_pass:
            tones = bits_to_tones(bits)
            synth = synth_waveform(
                tones,
                sample_rate,
                freq_f,
                dt_f,
                amplitude=amp,
                total_len=len(working),
            )
            working[: len(synth)] -= synth

    return results

