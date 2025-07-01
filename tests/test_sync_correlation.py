import math
import numpy as np

from search import dft_mag
from utils import COSTAS_SEQUENCE, TONE_SPACING_IN_HZ, RealSamples


def test_perfect_sync_correlation():
    sample_rate = 12000
    base_freq = 1500
    sym_len = int(round(sample_rate / TONE_SPACING_IN_HZ))
    start = int(sample_rate * 0.5)
    total_len = start + sym_len * len(COSTAS_SEQUENCE)
    samples = np.zeros(total_len)

    for idx, tone in enumerate(COSTAS_SEQUENCE):
        freq = base_freq + tone * TONE_SPACING_IN_HZ
        n = np.arange(sym_len)
        wave = 2 * np.cos(2 * math.pi * freq * n / sample_rate)
        sstart = start + idx * sym_len
        samples[sstart : sstart + sym_len] = wave

    audio = RealSamples(samples, sample_rate)

    score = 0.0
    for idx, tone in enumerate(COSTAS_SEQUENCE):
        freq = base_freq + tone * TONE_SPACING_IN_HZ
        sstart = start + idx * sym_len
        score += dft_mag(audio.samples, audio.sample_rate_in_hz, sstart, freq, sym_len)

    correlation = score / len(COSTAS_SEQUENCE)
    assert abs(correlation - 1.0) < 1e-6
