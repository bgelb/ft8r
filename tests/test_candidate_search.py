from search import find_candidates
from utils import read_wav, RealSamples, TONE_SPACING_IN_HZ
import random

from tests.utils import generate_ft8_wav


def test_candidate_search(tmp_path):
    msg = "K1ABC W9XYZ EN37"
    wav = generate_ft8_wav(msg, tmp_path)
    audio = read_wav(str(wav))
    freq_range = [1000 + i * TONE_SPACING_IN_HZ for i in range(int((2000 - 1000) / TONE_SPACING_IN_HZ) + 1)]
    dt_step = int(audio.sample_rate_in_hz / TONE_SPACING_IN_HZ)
    dt_range = list(range(0, int(audio.sample_rate_in_hz * 2), dt_step))
    candidates = find_candidates(
        audio,
        freq_range,
        dt_range,
        threshold=4.0,
    )
    assert candidates, "no candidates found"
    score, dt, freq = candidates[0]
    assert abs(freq - 1500) < 1.0
    assert abs(dt - 0.0) < 0.2
    assert score > 4.0
    for _, _, f in candidates:
        assert abs(f - 1500) <= 32


def test_candidate_search_noise():
    sample_rate_in_hz = 12000
    freq_range = [1000 + i * TONE_SPACING_IN_HZ for i in range(int((2000 - 1000) / TONE_SPACING_IN_HZ) + 1)]
    dt_step = int(sample_rate_in_hz / TONE_SPACING_IN_HZ)
    dt_range = list(range(0, int(sample_rate_in_hz * 2), dt_step))
    random.seed(123)
    sym_len = dt_step
    num_samples = dt_range[-1] + sym_len * 7
    noise = [random.uniform(-1e-2, 1e-2) for _ in range(num_samples)]
    noise_audio = RealSamples(noise, sample_rate_in_hz)
    cands = find_candidates(noise_audio, freq_range, dt_range, threshold=4.0)
    assert not cands
