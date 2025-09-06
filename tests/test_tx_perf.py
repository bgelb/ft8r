import time
import pytest

from utils import generate_ft8_waveform
from tests.utils import ft8code_bits, resolve_wsjt_binary


@pytest.mark.skipif(resolve_wsjt_binary('ft8code') is None, reason='requires ft8code')
def test_tx_generator_speed():
    # Simple message
    msg = "CQ K1ABC FN31"
    bits = ft8code_bits(msg)

    # Warm up caches
    _ = generate_ft8_waveform(bits, sample_rate=12000, base_freq_hz=1500.0)

    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = generate_ft8_waveform(bits, sample_rate=12000, base_freq_hz=1500.0)
    dt = time.perf_counter() - t0
    avg_ms = (dt / iters) * 1000.0

    # Expect well under 1 second per generation on typical hardware
    assert avg_ms < 200.0, f"TX gen too slow: avg {avg_ms:.1f} ms over {iters} runs"
