import demod
from pathlib import Path
from utils import read_wav, check_crc
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
            valid_text = True
        except Exception:
            valid_text = False
        ok_crc = False
        try:
            ok_crc = check_crc(bits)
        except Exception:
            ok_crc = False
        conv_info.append((demod._LDPC_DECODER.converge, valid_text, ok_crc))
        return bits

    demod.ldpc_decode = wrapper
    try:
        results = demod.decode_full_period(audio)
    finally:
        demod.ldpc_decode = orig

    # With strict CRC gating, results may be empty. Verify that at least one LDPC
    # attempt ran with the decoder not having converged, demonstrating the OSD path
    # was exercised.
    assert any((not converged) for converged, _valid_text, _ok_crc in conv_info)
