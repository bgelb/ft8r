import pytest

from search import find_candidates
from utils import read_wav
from demod import fine_sync_candidate
from tests.utils import generate_ft8_wav, default_search_params, DEFAULT_SEARCH_THRESHOLD


def run_sync(tmp_path, freq_offset=0.0, dt_offset=0.0):
    msg = "K1ABC W9XYZ EN37"
    wav = generate_ft8_wav(
        msg,
        tmp_path,
        freq=1500 + freq_offset,
        snr=0,
        dt=dt_offset,
        fdop=0.0,
    )
    audio = read_wav(str(wav))
    max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)
    cand = find_candidates(audio, max_freq_bin, max_dt_symbols, threshold=DEFAULT_SEARCH_THRESHOLD)[0]
    _, dt, freq = cand
    bb, dt_f, freq_f = fine_sync_candidate(audio, freq, dt)
    return dt_f, freq_f


def test_fine_sync_dt(tmp_path):
    dt_off = 0.073
    dt_f, freq_f = run_sync(tmp_path, dt_offset=dt_off)
    assert abs(dt_f - dt_off) < 0.005
    assert abs(freq_f - 1500) < 0.5


def test_fine_sync_freq(tmp_path):
    freq_off = 1.7
    dt_f, freq_f = run_sync(tmp_path, freq_offset=freq_off)
    # ``fine_sync_candidate`` should recover the true transmit frequency.
    # The expected value includes the intentional ``freq_off`` from the
    # generated test signal.
    assert abs(freq_f - (1500 + freq_off)) < 0.5
    assert dt_f == pytest.approx(0.0, abs=0.006)
