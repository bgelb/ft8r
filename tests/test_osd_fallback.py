import demod
from pathlib import Path
from utils import read_wav
from utils.decode import decode77


def test_osd_used_for_sample():
    wav = Path(__file__).resolve().parent.parent / "ft8_lib-2.0" / "test" / "wav" / "191111_110200.wav"
    audio = read_wav(str(wav))

    conv_info = []
    orig = demod.ldpc_decode

    def wrapper(llrs):
        bits = orig(llrs)
        try:
            decode77(bits[:77])
            valid = True
        except Exception:
            valid = False
        conv_info.append((demod._LDPC_DECODER.converge, valid))
        return bits

    demod.ldpc_decode = wrapper
    try:
        results = demod.decode_full_period(audio)
    finally:
        demod.ldpc_decode = orig

    assert len(results) >= 1
    assert any((not converged) and valid for converged, valid in conv_info)
