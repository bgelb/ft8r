import numpy as np
from demod import ldpc_decode, soft_demod, naive_hard_decode, downsample_to_baseband
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
    wav = generate_ft8_wav(msg, tmp_path, snr=-13)
    audio = read_wav(str(wav))
    max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)
    cand = find_candidates(
        audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD
    )[0]
    _, dt, freq = cand
    bb = downsample_to_baseband(audio, freq)
    llrs = soft_demod(bb, 0.0, dt)
    naive_bits = naive_hard_decode(llrs)
    decoded = ldpc_decode(llrs)
    expected = ft8code_bits(msg)
    mismatches = sum(a != b for a, b in zip(naive_bits, expected))
    assert mismatches > 0
    assert not check_crc(naive_bits)
    assert decoded == expected
    assert check_crc(decoded)
