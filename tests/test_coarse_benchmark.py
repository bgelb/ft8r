import json
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("FT8R_COARSE_FULL", "0") in ("0", "", "false", "False"),
    reason="full coarse characterization disabled by default; set FT8R_COARSE_FULL=1 to enable",
)


def test_coarse_recall_curve_full():
    from benchmarks.coarse.cli import cmd_golden, cmd_run, cmd_eval, cmd_report

    root = Path(__file__).resolve().parents[1]
    wav_root = root / "ft8_lib-2.0" / "test" / "wav"
    assert wav_root.exists(), f"Missing WAV root: {wav_root}"

    work = root / ".tmp" / "coarse_benchmark_full"
    (work / "strategies").mkdir(parents=True, exist_ok=True)
    (work / "runs").mkdir(parents=True, exist_ok=True)

    golden_path = work / "golden.csv"
    strategy_path = work / "strategies" / "uncapped.json"
    strategy_path.write_text(
        json.dumps(
            {
                "name": "uncapped",
                "threshold": 1.0,
                "nms_radius_dt": 1,
                "nms_radius_df": 1,
                "candidate_cap": 0,  # keep all candidates
            }
        )
    )

    # 1) Golden over full set
    rc = cmd_golden(wav_root=wav_root, out_path=golden_path, limit=0)
    assert rc == 0 and golden_path.exists(), "Failed to generate golden table"

    # 2) Run coarse stage
    runs_dir = work / "runs"
    rc = cmd_run(wav_root=wav_root, out_dir=runs_dir, strategy_path=strategy_path, limit=0)
    assert rc == 0, "Coarse run failed"

    # Locate most recent run directory for the strategy
    runs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("uncapped__")])
    assert runs, "No run directory produced"
    run_dir = max(runs, key=lambda p: p.stat().st_mtime)
    cand_path = run_dir / "candidates__uncapped.csv"
    assert cand_path.exists(), f"Missing candidates CSV: {cand_path}"

    # 3) Eval and compute recall curve to K=200
    eval_dir = work / "eval"
    rc = cmd_eval(golden_path=golden_path, candidates_path=cand_path, out_dir=eval_dir)
    assert rc == 0, "Eval failed"

    metrics_path = eval_dir / "metrics.json"
    tables_path = eval_dir / "tables.tsv"
    assert metrics_path.exists() and tables_path.exists(), "Missing eval outputs"

    m = json.loads(metrics_path.read_text())
    assert 0.0 <= m.get("coarse_recall", -1.0) <= 1.0, "Invalid coarse_recall"

    # Load near-rank list
    lines = [ln.strip() for ln in tables_path.read_text().splitlines() if ln.strip()]
    assert lines and lines[0] == "near_rank", "Invalid tables.tsv header"
    near_ranks = [int(x) for x in lines[1:]]
    assert near_ranks, "No near-rank data found"

    # Build recall@K curve up to 200 (among matched)
    def _recall_at_k(k: int) -> float:
        return sum(1 for r in near_ranks if r <= k) / len(near_ranks)

    ks = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250]
    curve = {k: _recall_at_k(k) for k in ks}

    # Sanity: monotonic non-decreasing
    last = 0.0
    for k in ks:
        assert curve[k] + 1e-12 >= last, f"recall@{k} decreased vs previous K"
        last = curve[k]

    # Persist curve (for local inspection / CI artifact)
    out_curve = work / "recall_curve.json"
    out_curve.write_text(json.dumps({"coarse_recall": m.get("coarse_recall"), "recall_at": curve}, indent=2))

    # 4) Produce a simple report as well
    report_dir = work / "report"
    rc = cmd_report(metrics_path=metrics_path, out_dir=report_dir)
    assert rc == 0 and (report_dir / "report.md").exists(), "Report generation failed"


