import time

from demod import decode_full_period_multipass
from utils import read_wav
from tests.test_sample_wavs import (
    DATA_DIR,
    parse_expected,
    list_all_stems,
)


# Reuse the short-suite baseline as pass 1 ratchet
PASS1_MIN_RATIO = 0.83


def _short_sample_stems() -> list[str]:
    stems = list_all_stems()
    return [s for i, s in enumerate(sorted(stems)) if i % 5 == 0]


def test_decode_sample_wavs_short_multipass_sic():
    t0 = time.monotonic()
    stems = _short_sample_stems()

    # Per-pass unique decodes accumulated across all files (by payload bits)
    decoded_per_pass: list[dict[str, str]] = [
        {},
        {},
        {},
    ]
    expected_set: set[str] = set()

    for stem in stems:
        wav_path = DATA_DIR / f"{stem}.wav"
        txt_path = DATA_DIR / f"{stem}.txt"

        audio = read_wav(str(wav_path))
        out = decode_full_period_multipass(
            audio, passes=3, sic_scale=0.7, return_pass_records=True
        )
        per_pass = out["per_pass"]
        # Dedup within each pass by payload bits
        for p in range(3):
            for r in per_pass[p]:
                key = r.get("bits") or r.get("message")
                if key is None:
                    continue
                decoded_per_pass[p].setdefault(key, r.get("message", ""))

        # Expected messages for this file
        for (msg, _dt, _freq) in parse_expected(txt_path):
            expected_set.add(msg)

    assert len(expected_set) > 0, "No sample records found"

    # Compute cumulative per-pass decode rates (monotonic by construction)
    cumulative_maps = []
    acc: dict[str, str] = {}
    for p in range(3):
        acc.update(decoded_per_pass[p])
        cumulative_maps.append(dict(acc))

    def _rate(dct: dict[str, str]) -> float:
        correct = sum(1 for txt in dct.values() if txt in expected_set)
        return (correct / len(expected_set)) if expected_set else 0.0

    pass1 = _rate(cumulative_maps[0])
    pass2 = _rate(cumulative_maps[1])
    pass3 = _rate(cumulative_maps[2])

    print(
        f"Short SIC metrics: pass1={pass1:.3f} pass2={pass2:.3f} pass3={pass3:.3f}"
    )

    # Ratchets: pass 1 matches current short baseline threshold; later passes
    # must be non-decreasing (cumulative).
    assert pass1 >= PASS1_MIN_RATIO, (
        f"Pass1 decode rate {pass1:.3f} < {PASS1_MIN_RATIO:.3f}"
    )
    assert pass2 >= pass1, (
        f"Pass2 decode rate {pass2:.3f} < pass1 {pass1:.3f}"
    )
    assert pass3 >= pass2, (
        f"Pass3 decode rate {pass3:.3f} < pass2 {pass2:.3f}"
    )

    # Emit metrics for CI visibility
    try:
        import json, os
        os.makedirs(".tmp", exist_ok=True)
        duration_sec = time.monotonic() - t0
        with open(".tmp/ft8r_short_sic_metrics.json", "w") as f:
            json.dump(
                {
                    "pass1": float(pass1),
                    "pass2": float(pass2),
                    "pass3": float(pass3),
                    "duration_sec": float(duration_sec),
                },
                f,
                indent=2,
            )
    except Exception:
        pass
