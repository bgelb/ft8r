import os
import time
from pathlib import Path

import pytest

from demod import decode_full_period
from utils import read_wav
from tests.test_sample_wavs import DATA_DIR


pytestmark = pytest.mark.skipif(
    os.getenv("FT8R_PERF", "0") in ("0", "", "false", "False"),
    reason="performance test disabled by default",
)


@pytest.mark.performance
def test_decode_full_period_runtime_smoke():
    # Choose a moderately busy sample to reflect typical workload
    stem = os.getenv("FT8R_PERF_STEM", "websdr_test6")
    wav_path = DATA_DIR / f"{stem}.wav"
    assert wav_path.exists(), f"Missing test WAV: {wav_path}"

    audio = read_wav(str(wav_path))

    # Warm-up to populate caches and FFT plans
    _ = decode_full_period(audio)

    repeats = int(os.getenv("FT8R_PERF_REPEATS", "1"))
    target_s = float(os.getenv("FT8R_TARGET_S", "5.0"))

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        _ = decode_full_period(audio)
        best = min(best, time.perf_counter() - start)

    # Allow override to avoid failing in constrained CI
    allow_fail = os.getenv("FT8R_PERF_ALLOW_FAIL", "0") not in ("0", "", "false", "False")

    print(f"decode_full_period best {best:.3f}s (target {target_s:.3f}s)")
    assert allow_fail or best <= target_s


