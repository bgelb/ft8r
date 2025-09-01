#!/usr/bin/env python3
"""
Filter and list cases from the experiment JSON where the default pipeline failed
but a brute-force seeded decode recovered the message.

Usage:
  PYTHONPATH=. python scripts/list_seed_recovered_cases.py \
    [.tmp/ft8r_wav_failures_experiment.json]

Outputs a tab-separated table: stem, dt, freq, message
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".tmp/ft8r_wav_failures_experiment.json")
    if not path.exists():
        print(f"Missing report: {path}. Run: PYTHONPATH=. python scripts/experiment_wav_failures.py", file=sys.stderr)
        return 2
    data = json.loads(path.read_text())
    details = data.get("details", [])

    # Select cases that were not recovered by lowering threshold, but were
    # recovered by brute-force around known dt/freq.
    focus = [
        d for d in details
        if d.get("threshold_recovered") is None and bool(d.get("bruteforce_recovered"))
    ]

    print("stem\tdt\tfreq\tmessage")
    for d in focus:
        stem = d.get("stem", "")
        exp = d.get("expected", {})
        msg = exp.get("message", "")
        dt = exp.get("dt", 0.0)
        freq = exp.get("freq", 0.0)
        print(f"{stem}\t{dt}\t{freq}\t{msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

