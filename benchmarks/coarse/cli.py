import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional
from utils.golden import normalize_wsjtx_message


# Public CLI entrypoint
def main(argv: Optional[List[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    if args.cmd == "golden":
        return cmd_golden(
            wav_root=Path(args.wav_root),
            out_path=Path(args.out),
            limit=args.limit,
        )
    if args.cmd == "run":
        return cmd_run(
            wav_root=Path(args.wav_root),
            out_dir=Path(args.out_dir),
            strategy_path=Path(args.strategy),
            limit=args.limit,
        )
    if args.cmd == "eval":
        return cmd_eval(
            golden_path=Path(args.golden),
            candidates_path=Path(args.candidates),
            out_dir=Path(args.out_dir),
        )
    if args.cmd == "report":
        return cmd_report(
            metrics_path=Path(args.metrics),
            out_dir=Path(args.out_dir),
        )
    parser.error("unknown command")
    return 2


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench coarse", description="Coarse-stage benchmark harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("golden", help="Generate golden labels via WSJT-X jt9")
    pg.add_argument("wav_root", help="Directory containing WAV files (recursively)")
    pg.add_argument("--out", required=True, help="Output golden CSV/Parquet path (by extension)")
    pg.add_argument("--limit", type=int, default=0, help="Optional limit of WAVs for smoke runs")

    pr = sub.add_parser("run", help="Run coarse search for a strategy and emit candidates")
    pr.add_argument("wav_root", help="Directory containing WAV files (recursively)")
    pr.add_argument("--strategy", required=True, help="Path to strategy JSON/yaml")
    pr.add_argument("--out-dir", required=True, help="Output directory root for this strategy")
    pr.add_argument("--limit", type=int, default=0, help="Optional limit of WAVs for smoke runs")

    pe = sub.add_parser("eval", help="Evaluate candidates against golden and compute metrics")
    pe.add_argument("--golden", required=True)
    pe.add_argument("--candidates", required=True)
    pe.add_argument("--out-dir", required=True)

    pp = sub.add_parser("report", help="Generate Markdown report and figures from metrics")
    pp.add_argument("--metrics", required=True)
    pp.add_argument("--out-dir", required=True)

    return p


# =============== Golden generation ===============

@dataclass
class GoldenRow:
    wav_stem: str
    dt: float
    df: float
    snr_db: float
    msg: str


def _iter_wavs(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.wav")):
        yield p


def _parse_wsjt_txt_line(line: str) -> Optional[GoldenRow]:
    line = line.strip()
    if not line:
        return None
    # Expected like: "000000  -5  1.1  809 ~  SQ5FBI G3NDC IO91"
    try:
        # Split once around the tilde to separate the message
        if "~" in line:
            left, msg = line.split("~", 1)
            msg = normalize_wsjtx_message(msg)
        else:
            parts = line.split(maxsplit=5)
            msg = normalize_wsjtx_message(parts[5]) if len(parts) >= 6 else ""
            left = " ".join(parts[:5])
        lparts = left.split()
        # lparts: [time_tag, snr, dt, freq]
        snr_db = float(lparts[1])
        dt = float(lparts[2])
        df = float(lparts[3])
        return GoldenRow(wav_stem="", dt=dt, df=df, snr_db=snr_db, msg=msg)
    except Exception:
        return None


def cmd_golden(wav_root: Path, out_path: Path, limit: int) -> int:
    # Prefer existing .txt truth files next to wavs; otherwise allow running jt9 if present later.
    rows: List[GoldenRow] = []
    count = 0
    for wav in _iter_wavs(wav_root):
        stem = wav.with_suffix("").name
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        for line in txt.read_text().splitlines():
            r = _parse_wsjt_txt_line(line)
            if r is None:
                continue
            r.wav_stem = stem
            rows.append(r)
        count += 1
        if limit and count >= limit:
            break

    if out_path.suffix.lower() == ".csv":
        _write_csv(rows, out_path)
    elif out_path.suffix.lower() in (".parquet", ".pq"):
        _write_parquet(rows, out_path)
    else:
        raise SystemExit("golden out must be .csv or .parquet")

    return 0


def _write_csv(rows: List[GoldenRow], out: Path) -> None:
    import csv

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wav_stem", "dt", "df", "snr_db", "msg"])
        for r in rows:
            w.writerow([r.wav_stem, r.dt, r.df, r.snr_db, r.msg])


def _write_parquet(rows: List[GoldenRow], out: Path) -> None:
    import pandas as pd

    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_parquet(out, index=False)


# =============== Coarse run ===============

@dataclass
class Strategy:
    name: str
    nms_radius_dt: int = 1
    nms_radius_df: int = 1
    threshold: float = 1.0
    candidate_cap: int = 0
    normalization: str = "ratio"
    notes: str = ""


def _load_strategy(path: Path) -> Strategy:
    import json
    text = path.read_text()
    # Prefer JSON; only attempt YAML if the extension indicates YAML
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise SystemExit(f"YAML strategy file provided but PyYAML is not installed: {e}")
        obj = yaml.safe_load(text)
        return Strategy(**obj)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON strategy file: {e}")
    return Strategy(**obj)


def cmd_run(wav_root: Path, out_dir: Path, strategy_path: Path, limit: int) -> int:
    from utils import read_wav
    from tests.utils import default_search_params
    from search import candidate_score_map, peak_candidates

    strat = _load_strategy(strategy_path)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{strat.name}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log strategy blob for reproducibility
    (run_dir / "strategy.json").write_text(json.dumps(asdict(strat), indent=2))

    # One candidates CSV per run
    import csv
    cand_path = run_dir / f"candidates__{strat.name}.csv"
    with cand_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wav_stem", "t_coarse", "f_coarse", "score", "rank"])

        count = 0
        for wav in _iter_wavs(wav_root):
            audio = read_wav(str(wav))
            max_freq_bin, max_dt_symbols = default_search_params(audio.sample_rate_in_hz)
            scores, dts, freqs = candidate_score_map(audio, max_freq_bin, max_dt_symbols)

            # NMS radius with simple local suppression by changing the footprint size
            # We reuse peak_candidates but with a custom threshold; radius is handled by
            # multiple maximum_filter passes if needed.
            peaks = peak_candidates(scores, dts, freqs, threshold=strat.threshold)
            if strat.candidate_cap:
                peaks = peaks[: strat.candidate_cap]

            stem = wav.with_suffix("").name
            for rank, (score, dt, freq) in enumerate(peaks, start=1):
                w.writerow([stem, dt, freq, score, rank])

            count += 1
            if limit and count >= limit:
                break

    return 0


# =============== Eval ===============

@dataclass
class Metrics:
    coarse_recall: float
    recall_at: dict
    median_near_rank: Optional[float]
    p90_near_rank: Optional[float]
    useful_fraction: Optional[float]


def cmd_eval(golden_path: Path, candidates_path: Path, out_dir: Path) -> int:
    import numpy as np
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    g_rows = _read_table_rows(golden_path)
    c_rows = _read_table_rows(candidates_path)

    # Match tolerances
    dt_tol = 0.08  # 80 ms
    df_tol = 3.125  # 1 coarse bin

    # Index by wav_stem
    by_stem_g: dict[str, List[dict]] = {}
    by_stem_c: dict[str, List[dict]] = {}
    for r in g_rows:
        by_stem_g.setdefault(r["wav_stem"], []).append(r)
    for r in c_rows:
        by_stem_c.setdefault(r["wav_stem"], []).append(r)

    recalls: List[int] = []
    near_ranks: List[int] = []

    for stem, gg in by_stem_g.items():
        cc = by_stem_c.get(stem, [])
        if not cc:
            recalls.extend([0] * len(gg))
            continue
        for row in gg:
            dt = float(row["dt"])
            df = float(row["df"])
            # find nearest matching candidate by tolerance
            best_rank: Optional[int] = None
            for cand in cc:
                ddt = abs(float(cand["t_coarse"]) - dt)
                ddf = abs(float(cand["f_coarse"]) - df)
                if ddt <= dt_tol and ddf <= df_tol:
                    rank = int(cand["rank"]) if isinstance(cand["rank"], (int, float)) else int(cand["rank"])
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
            if best_rank is not None:
                recalls.append(1)
                near_ranks.append(best_rank)
            else:
                recalls.append(0)

    import statistics

    coarse_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    median_nr = (float(statistics.median(near_ranks)) if near_ranks else None)
    # p90 via numpy for convenience if available; else approximate
    p90_nr = (float(np.percentile(near_ranks, 90)) if near_ranks else None)

    recall_at = {}
    for k in (1, 5, 10, 25, 50, 75, 100, 150, 200, 250):
        if near_ranks:
            recall_at[str(k)] = float(sum(1 for r in near_ranks if r <= k) / len(near_ranks))
        else:
            recall_at[str(k)] = 0.0

    metrics = Metrics(
        coarse_recall=coarse_recall,
        recall_at=recall_at,
        median_near_rank=median_nr,
        p90_near_rank=p90_nr,
        useful_fraction=None,
    )

    (out_dir / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2))
    # Minimal TSV breakdown: near_ranks
    (out_dir / "tables.tsv").write_text("near_rank\n" + "\n".join(map(str, near_ranks)))
    return 0


def _read_table_rows(path: Path) -> List[dict]:
    if path.suffix.lower() == ".csv":
        import csv
        rows: List[dict] = []
        with path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows
    if path.suffix.lower() in (".parquet", ".pq"):
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise SystemExit(f"Reading parquet requires pandas: {e}")
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    raise SystemExit("Unsupported table format. Use .csv or .parquet")


# =============== Report ===============

def cmd_report(metrics_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = json.loads(Path(metrics_path).read_text())
    # One-page Markdown summary
    md = ["# Coarse Benchmark Summary", "", f"Generated: {datetime.utcnow().isoformat()}Z", ""]
    md.append(f"- coarse_recall: {m.get('coarse_recall')}")
    md.append(f"- recall_at: {m.get('recall_at')}")
    md.append(f"- median_near_rank: {m.get('median_near_rank')}")
    md.append(f"- p90_near_rank: {m.get('p90_near_rank')}")
    (out_dir / "report.md").write_text("\n".join(md))
    return 0
