#!/usr/bin/env python3
"""Compare decoded payload bits to ft8code-generated bits for the same text.

Usage:
  python scripts/compare_bits_with_ft8code.py [--short]

Requires WSJT-X tools (ft8code) available via WSJTX_BIN_DIR or system PATH.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

from utils import read_wav
from demod import decode_full_period
from tests.utils import ft8code_bits, resolve_wsjt_binary
from tests.test_sample_wavs import DATA_DIR, list_all_stems
from tests.test_sample_wavs_short import _short_sample_stems


def main(short: bool) -> int:
    if resolve_wsjt_binary("ft8code") is None:
        print("ft8code not available; set WSJTX_BIN_DIR or install WSJT-X.", file=sys.stderr)
        return 2
    stems = _short_sample_stems() if short else list_all_stems()
    total = 0
    matched = 0
    mismatches = []
    for stem in stems:
        wav = DATA_DIR / f"{stem}.wav"
        audio = read_wav(str(wav))
        recs = decode_full_period(audio, include_bits=True)
        for r in recs:
            total += 1
            msg = r["message"]
            got_bits = r.get("bits")
            try:
                want_bits = ft8code_bits(msg)
            except Exception as e:
                continue
            if want_bits == got_bits:
                matched += 1
            else:
                mismatches.append((stem, msg, got_bits, want_bits))
    print(f"Decoded records: {total} matched bits: {matched} mismatches: {len(mismatches)}")
    if mismatches:
        for stem, msg, got, want in mismatches[:10]:
            print(f"Mismatch: stem={stem} msg='{msg}'\n  got={got}\n want={want}")
    return 0


if __name__ == "__main__":
    short = "--short" in sys.argv
    raise SystemExit(main(short))

