import time
import pytest

from utils import generate_ft8_waveform
from tests.utils import ft8code_bits, resolve_wsjt_binary


@pytest.mark.skipif(resolve_wsjt_binary('ft8code') is None, reason='requires ft8code')
def test_tx_generator_speed():
    # Simple message
    msg = "CQ K1ABC FN31"
    bits = ft8code_bits(msg)

    # Single deterministic run timing
    t0 = time.perf_counter()
    _ = generate_ft8_waveform(bits, sample_rate=12000, base_freq_hz=1500.0)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Expect well under 1 second per generation on typical hardware
    assert dt_ms < 200.0, f"TX gen too slow: {dt_ms:.1f} ms for single run"
