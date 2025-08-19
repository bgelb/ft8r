#!/usr/bin/env python3
import argparse
import json
import os
import platform
import time
from statistics import mean, median

from utils import read_wav
from demod import decode_full_period
from tests.test_sample_wavs import DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=os.getenv("FT8R_PERF_STEM", "websdr_test6"))
    ap.add_argument("--repeats", type=int, default=int(os.getenv("FT8R_PERF_REPEATS", "3")))
    ap.add_argument("--warmup", type=int, default=int(os.getenv("FT8R_PERF_WARMUP", "1")))
    args = ap.parse_args()

    wav = DATA_DIR / f"{args.stem}.wav"
    audio = read_wav(str(wav))

    for _ in range(max(0, args.warmup)):
        _ = decode_full_period(audio)

    times = []
    last_results = None
    for _ in range(max(1, args.repeats)):
        t0 = time.perf_counter()
        last_results = decode_full_period(audio)
        times.append(time.perf_counter() - t0)

    out = {
        "stem": args.stem,
        "repeats": args.repeats,
        "best_s": min(times) if times else None,
        "avg_s": mean(times) if times else None,
        "median_s": median(times) if times else None,
        "decoded_count": len(last_results or []),
        "python": platform.python_version(),
        "machine": platform.platform(),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()

