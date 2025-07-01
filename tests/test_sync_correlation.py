import numpy as np

from search import find_candidates
from utils import (
    COSTAS_SEQUENCE,
    TONE_SPACING_IN_HZ,
    COSTAS_START_OFFSET_SEC,
    RealSamples,
)
from tests.utils import default_search_params


def test_perfect_sync_detection():
    sample_rate = 12000
    base_freq = 1500
    sym_len = int(sample_rate / TONE_SPACING_IN_HZ)
    start = int(sample_rate * COSTAS_START_OFFSET_SEC)
    total_len = start + sym_len * len(COSTAS_SEQUENCE)
    samples = np.zeros(total_len)

    for idx, tone in enumerate(COSTAS_SEQUENCE):
        freq = base_freq + tone * TONE_SPACING_IN_HZ
        n = np.arange(sym_len)
        wave = 2 * np.cos(2 * np.pi * freq * n / sample_rate)
        sstart = start + idx * sym_len
        samples[sstart : sstart + sym_len] = wave

    audio = RealSamples(samples, sample_rate)
    max_freq_bin, max_dt_symbols = default_search_params(sample_rate)
    candidates = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=0.1)
    assert candidates
    score, dt, freq = candidates[0]
    expected_dt = (start // sym_len) * sym_len / sample_rate - COSTAS_START_OFFSET_SEC
    assert abs(freq - base_freq) < 1.0
    assert abs(dt - expected_dt) < 1e-6
    assert score > 0

