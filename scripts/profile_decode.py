#!/usr/bin/env python3
import argparse
import cProfile
import os
import pstats
from pathlib import Path

from demod import decode_full_period
from utils import read_wav


def resolve_sample(stem: str) -> Path:
    tests_data = Path(__file__).resolve().parents[1] / "ft8_lib-2.0" / "test" / "wav"
    p = Path(stem)
    if p.suffix.lower() == ".wav" and p.exists():
        return p
    # Try resolve by known test stem
    candidate = tests_data / f"{stem}.wav"
    if candidate.exists():
        return candidate
    raise SystemExit(f"Could not resolve WAV path for '{stem}'. Provide full path or known stem.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile FT8 full-period decode")
    parser.add_argument("input", help="WAV file path or known test stem (e.g. websdr_test6)")
    parser.add_argument("--repeat", type=int, default=1, help="Number of runs (best reported)")
    parser.add_argument("--cprofile", action="store_true", help="Run with cProfile and print top stats")
    parser.add_argument("--stats", type=str, default="", help="Optional path to write cProfile .prof file")
    parser.add_argument("--json", type=str, default="", help="Optional path to write section timing JSON")
    args = parser.parse_args()

    # Enable lightweight section profiler
    os.environ["FT8R_PROFILE"] = "1"

    wav_path = resolve_sample(args.input)
    audio = read_wav(str(wav_path))

    best = None
    last_prof = None
    for _ in range(max(1, args.repeat)):
        if args.cprofile:
            prof = cProfile.Profile()
            prof.enable()
        else:
            prof = None
        import time

        t0 = time.perf_counter()
        _ = decode_full_period(audio)
        dt = time.perf_counter() - t0

        if args.cprofile:
            prof.disable()
            last_prof = prof

        if best is None or dt < best:
            best = dt

    print(f"Best wall time: {best:.3f} s")

    # Optionally dump cProfile stats
    if args.cprofile and last_prof is not None:
        stats = pstats.Stats(last_prof).sort_stats("cumtime")
        stats.print_stats(30)
        if args.stats:
            last_prof.dump_stats(args.stats)
            print(f"Saved cProfile stats to {args.stats}")

    # Optionally write section-level timings
    if args.json:
        from utils.prof import PROFILER

        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        PROFILER.dump_json(args.json)
        print(f"Saved section timing JSON to {args.json}")


if __name__ == "__main__":
    main()


