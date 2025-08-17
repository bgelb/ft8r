import numpy as np
from demod import ldpc_decode, soft_demod, naive_hard_decode, fine_sync_candidate
from search import find_candidates
from utils import read_wav, check_crc
from tests.utils import (
    generate_ft8_wav,
    default_search_params,
    DEFAULT_SEARCH_THRESHOLD,
    ft8code_bits,
)


def test_ldpc_decode_runs(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    wav = generate_ft8_wav(msg, tmp_path, snr=-15)
    audio = read_wav(str(wav))
    max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)
    cand = find_candidates(
        audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD
    )[0]
    _, dt, freq = cand
    bb, dt_f, freq_f = fine_sync_candidate(audio, freq, dt)
    llrs = soft_demod(bb)
    naive_bits = naive_hard_decode(llrs)
    decoded = ldpc_decode(llrs)
    expected = ft8code_bits(msg)
    mismatches = sum(a != b for a, b in zip(naive_bits, expected))
    assert mismatches > 0
    assert not check_crc(naive_bits)
    assert decoded == expected
    assert check_crc(decoded)


def test_hard_crc_then_ldpc_keeps_crc(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    # Strong SNR so hard-decision CRC passes
    wav = generate_ft8_wav(msg, tmp_path, snr=0)
    audio = read_wav(str(wav))
    max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)
    cand = find_candidates(
        audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD
    )[0]
    _, dt, freq = cand
    bb, dt_f, freq_f = fine_sync_candidate(audio, freq, dt)
    llrs = soft_demod(bb)
    hard_bits = naive_hard_decode(llrs)
    assert check_crc(hard_bits)
    decoded = ldpc_decode(llrs)
    # CRC on LDPC output is also valid
    assert check_crc(decoded)
