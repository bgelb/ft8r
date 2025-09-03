from search import find_candidates, candidate_score_map, peak_candidates
import numpy as np
import os
from utils import (
    read_wav,
    RealSamples,
    COSTAS_START_OFFSET_SEC,
    FT8_SYMBOL_LENGTH_IN_SEC,
    FT8_SYMBOLS_PER_MESSAGE,
    check_crc,
)
from demod import (
    soft_demod,
    naive_hard_decode,
    fine_sync_candidate,
)
import random

from tests.utils import (
    generate_ft8_wav,
    default_search_params,
    ft8code_bits,
    DEFAULT_SEARCH_THRESHOLD,
    DEFAULT_FREQ_EPS,
    DEFAULT_DT_EPS,
)


def test_candidate_search(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    wav = generate_ft8_wav(msg, tmp_path, snr=0)
    audio = read_wav(str(wav))
    max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)
    candidates = find_candidates(
        audio,
        max_freq_bin,
        max_dt_symbols,
        threshold=DEFAULT_SEARCH_THRESHOLD,
    )
    assert candidates, "no candidates found"
    score, dt, freq = candidates[0]
    assert abs(freq - 1500) < DEFAULT_FREQ_EPS
    assert abs(dt - 0.0) < DEFAULT_DT_EPS
    assert score > DEFAULT_SEARCH_THRESHOLD
    assert any(abs(f - 1500) <= 40 for _, _, f in candidates)


def test_candidate_search_noise():
    sample_rate_in_hz = 12000
    max_freq_bin, max_dt_symbols = default_search_params(sample_rate_in_hz)
    random.seed(123)
    total_len_sec = COSTAS_START_OFFSET_SEC + FT8_SYMBOLS_PER_MESSAGE * FT8_SYMBOL_LENGTH_IN_SEC
    num_samples = int(total_len_sec * sample_rate_in_hz)
    noise = [random.uniform(-1e-3, 1e-3) for _ in range(num_samples)]
    noise_audio = RealSamples(noise, sample_rate_in_hz)
    cands = find_candidates(noise_audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD)
    assert not cands


def test_naive_demod(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    wav = generate_ft8_wav(msg, tmp_path, snr=0)
    audio = read_wav(str(wav))
    max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)
    cand = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD)[0]
    _, dt, freq = cand
    bb, dt_f, freq_f = fine_sync_candidate(audio, freq, dt)
    llrs = soft_demod(bb)
    decoded_bits = naive_hard_decode(llrs)
    expected_bits = ft8code_bits(msg)
    assert decoded_bits == expected_bits
    assert check_crc(decoded_bits)


def test_naive_demod_low_snr(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    wav = generate_ft8_wav(msg, tmp_path, snr=-16)
    audio = read_wav(str(wav))
    max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)
    cand = find_candidates(
        audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD
    )[0]
    _, dt, freq = cand
    bb, dt_f, freq_f = fine_sync_candidate(audio, freq, dt)
    llrs = soft_demod(bb)
    decoded_bits = naive_hard_decode(llrs)
    expected_bits = ft8code_bits(msg)
    mismatches = sum(a != b for a, b in zip(decoded_bits, expected_bits))
    assert mismatches > 0
    assert not check_crc(decoded_bits)


def test_candidate_peak_filter(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    wav = generate_ft8_wav(msg, tmp_path, snr=10)
    audio = read_wav(str(wav))
    max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)

    scores, dts, freqs = candidate_score_map(audio, max_freq_bin, max_dt_symbols)
    thresh = DEFAULT_SEARCH_THRESHOLD

    # Several cells should exceed ``thresh`` and the filtering step should
    # reduce the count of candidates.
    num_above = np.count_nonzero(scores >= thresh)
    assert num_above > 1

    peaks = peak_candidates(scores, dts, freqs, threshold=thresh)

    assert 0 < len(peaks) < num_above
    score, dt, freq = peaks[0]
    assert abs(freq - 1500) < DEFAULT_FREQ_EPS
    assert abs(dt - 0.0) < DEFAULT_DT_EPS
    assert score > DEFAULT_SEARCH_THRESHOLD

    # Peak filtering tested directly via peak_candidates above. find_candidates
    # uses the budgeted selection path unconditionally.
