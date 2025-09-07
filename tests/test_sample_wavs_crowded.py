import time

from demod import decode_full_period, decode_full_period_multipass
from utils import read_wav
from tests.test_sample_wavs import (
    DATA_DIR,
    parse_expected,
    list_all_stems,
)


def _crowded_sample_stems(top_n: int = 20) -> list[str]:
    stems = list_all_stems()
    sized: list[tuple[int, str]] = []
    for stem in stems:
        txt_path = DATA_DIR / f"{stem}.txt"
        exp = parse_expected(txt_path)
        sized.append((len(exp), stem))
    sized.sort(key=lambda x: (-x[0], x[1]))
    return [s for _n, s in sized[:top_n]]


def test_decode_sample_wavs_crowded_sic():
    t0 = time.monotonic()
    stems = _crowded_sample_stems(20)

    # Per-file-normalized aggregates to keep rates bounded
    base_correct = 0
    base_total = 0
    p1_correct = 0
    p2_correct = 0
    p3_correct = 0
    total = 0

    for stem in stems:
        wav_path = DATA_DIR / f"{stem}.wav"
        txt_path = DATA_DIR / f"{stem}.txt"
        audio = read_wav(str(wav_path))
        exp_texts = {m for (m, _dt, _fq) in parse_expected(txt_path)}

        # Baseline single-pass per file
        res_base = decode_full_period(audio, include_bits=True)
        dedup_base: dict[str, str] = {}
        for r in res_base:
            key = r.get("bits") or r.get("message")
            if key:
                dedup_base.setdefault(key, r.get("message", ""))
        base_correct += len(set(dedup_base.values()) & exp_texts)
        base_total += len(exp_texts)

        # SIC multipass cumulative per file
        out = decode_full_period_multipass(audio, passes=3, sic_scale=0.7, return_pass_records=True)
        per_pass = out["per_pass"]
        texts = [set(), set(), set()]
        for i in range(3):
            if i > 0:
                texts[i].update(texts[i - 1])
            texts[i].update(r.get("message", "") for r in per_pass[i])
        p1_correct += len(texts[0] & exp_texts)
        p2_correct += len(texts[1] & exp_texts)
        p3_correct += len(texts[2] & exp_texts)
        total += len(exp_texts)

    # Compute aggregated, per-file normalized rates
    base_rate = (base_correct / base_total) if base_total else 0.0
    p1_rate = (p1_correct / total) if total else 0.0
    p2_rate = (p2_correct / total) if total else 0.0
    p3_rate = (p3_correct / total) if total else 0.0

    print(
        f"Crowded (top 20) per-file: base={base_rate:.3f} p1={p1_rate:.3f} p2={p2_rate:.3f} p3={p3_rate:.3f}"
    )

    # Persist metrics for CI visibility
    try:
        import json, os
        os.makedirs(".tmp", exist_ok=True)
        duration_sec = time.monotonic() - t0
        with open(".tmp/ft8r_crowded_sic_metrics.json", "w") as f:
            json.dump(
                {
                    "base": float(base_rate),
                    "pass1": float(p1_rate),
                    "pass2": float(p2_rate),
                    "pass3": float(p3_rate),
                    "duration_sec": float(duration_sec),
                    "num_files": int(len(stems)),
                },
                f,
                indent=2,
            )
    except Exception:
        pass

    # Do not assert improvements here; this suite is exploratory and feeds CI
    assert 0.0 <= base_rate <= 1.0
    assert 0.0 <= p1_rate <= 1.0
    assert 0.0 <= p2_rate <= 1.0
    assert 0.0 <= p3_rate <= 1.0

