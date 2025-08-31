import json
import os
from pathlib import Path

import pytest


pytestmark = []


def test_coarse_recall_ratchet():
    """Ratchet: coarse_recall must not regress below threshold.

    This runs the full coarse characterization (uncapped) over the sample WAVs
    and asserts that coarse_recall (fraction of golden signals with at least one
    coarse candidate within tolerance) remains above a baseline target.
    """
    from benchmarks.coarse.cli import cmd_golden, cmd_run, cmd_eval

    root = Path(__file__).resolve().parents[1]
    wav_root = root / "ft8_lib-2.0" / "test" / "wav"
    assert wav_root.exists(), f"Missing WAV root: {wav_root}"

    work = root / ".tmp" / "coarse_ratchet"
    (work / "strategies").mkdir(parents=True, exist_ok=True)
    (work / "runs").mkdir(parents=True, exist_ok=True)

    # Prepare golden table
    golden_path = work / "golden.csv"
    rc = cmd_golden(wav_root=wav_root, out_path=golden_path, limit=0)
    assert rc == 0 and golden_path.exists(), "Failed to generate golden table"

    # Uncapped strategy JSON written inline
    strategy_path = work / "strategies" / "uncapped.json"
    strategy_path.write_text(
        json.dumps(
            {
                "name": "uncapped",
                "threshold": 1.0,
                "nms_radius_dt": 1,
                "nms_radius_df": 1,
                "candidate_cap": 0,
            }
        )
    )

    # Run coarse stage across the full set
    runs_dir = work / "runs"
    rc = cmd_run(wav_root=wav_root, out_dir=runs_dir, strategy_path=strategy_path, limit=0)
    assert rc == 0, "Coarse run failed"

    # Locate output candidates
    runs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("uncapped__")])
    assert runs, "No run directory produced"
    run_dir = max(runs, key=lambda p: p.stat().st_mtime)
    cand_path = run_dir / "candidates__uncapped.csv"
    assert cand_path.exists(), f"Missing candidates CSV: {cand_path}"

    # Eval
    eval_dir = work / "eval"
    rc = cmd_eval(golden_path=golden_path, candidates_path=cand_path, out_dir=eval_dir)
    assert rc == 0, "Eval failed"
    metrics_path = eval_dir / "metrics.json"
    assert metrics_path.exists(), "Missing eval metrics"

    m = json.loads(metrics_path.read_text())
    coarse_recall = float(m.get("coarse_recall", -1.0))
    assert 0.0 <= coarse_recall <= 1.0, "Invalid coarse_recall"

    # Baseline (measured): ~0.692; allow small slack to avoid flakiness
    target = float(os.getenv("FT8R_COARSE_RATCHET_TARGET", "0.68"))
    assert coarse_recall >= target, (
        f"coarse_recall regression: {coarse_recall:.4f} < target {target:.4f}"
    )


