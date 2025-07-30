import pytest
from utils import read_wav
from demod import decode_full_period
from tests.test_sample_wavs import DATA_DIR


def unique(records):
    seen = []
    for rec in records:
        dup = False
        for s in seen:
            if (
                rec["message"] == s["message"]
                and abs(rec["dt"] - s["dt"]) < 0.2
                and abs(rec["freq"] - s["freq"]) < 1.0
            ):
                dup = True
                break
        if not dup:
            seen.append(rec)
    return seen


@pytest.mark.parametrize(
    "wav_rel",
    ["191111_110130.wav", "20m_busy/test_02.wav"],
)
def test_multipass_recovers_additional(wav_rel):
    wav_path = DATA_DIR / wav_rel
    audio = read_wav(str(wav_path))
    single_pass = unique(decode_full_period(audio, max_passes=1))
    multi_pass = unique(decode_full_period(audio, max_passes=3))

    # multi_pass should not contain duplicates
    assert len(multi_pass) == len(set((r["message"], round(r["dt"], 1), round(r["freq"])) for r in multi_pass))

    assert len(multi_pass) > len(single_pass)
