import pytest

from utils import generate_ft8_waveform
from demod import decode_full_period
from tests.utils import ft8code_bits, DEFAULT_DT_EPS, DEFAULT_FREQ_EPS, resolve_wsjt_binary


@pytest.mark.skipif(resolve_wsjt_binary('ft8code') is None, reason='requires ft8code')
def test_tx_roundtrip_simple():
    # Choose a simple, valid message encodable by ft8code
    msg = "CQ K1ABC FN31"
    bits = ft8code_bits(msg)
    assert len(bits) == 174
    # Synthesize clean waveform at 12000 Hz, 1500 Hz base
    audio = generate_ft8_waveform(bits, sample_rate=12000, base_freq_hz=1500.0)
    # Run through our decode pipeline
    results = decode_full_period(audio, include_bits=True)
    # Expect exactly one valid decode with matching text
    assert any(r.get('message') == msg for r in results), f"missing {msg}: {results}"
    rec = next(r for r in results if r.get('message') == msg)
    # Verify dt/df are close to 0/1500 within test epsilons
    assert abs(float(rec.get('dt', 0.0)) - 0.0) < DEFAULT_DT_EPS
    assert abs(float(rec.get('freq', 0.0)) - 1500.0) < DEFAULT_FREQ_EPS
