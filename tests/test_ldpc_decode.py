import numpy as np
from demod import ldpc_decode, soft_demod, naive_hard_decode
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
    wav = generate_ft8_wav(msg, tmp_path)
    audio = read_wav(str(wav))
    max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)
    cand = find_candidates(
        audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD
    )[0]
    _, dt, freq = cand
    llrs = soft_demod(audio, freq, dt)
    naive_bits = naive_hard_decode(llrs)
    decoded = ldpc_decode(llrs)
    expected = ft8code_bits(msg)
    assert naive_bits == expected
    assert decoded == expected
    assert check_crc(decoded)
