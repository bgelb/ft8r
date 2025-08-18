import json
from pathlib import Path

import pytest


def pytest_configure(config):
    # Metrics for short sample_wavs regression
    config._ft8r_metrics = {"matched": 0, "total": 0}


@pytest.fixture(scope="session")
def ft8r_metrics(request):
    return request.config._ft8r_metrics


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    metrics = getattr(config, "_ft8r_metrics", None)
    if not metrics:
        return
    total = int(metrics.get("total") or 0)
    matched = int(metrics.get("matched") or 0)
    if total <= 0:
        return
    percent = 100.0 * matched / total if total else 0.0
    line = f"FT8R short sample_wavs: {matched}/{total} ({percent:.1f}%)"
    terminalreporter.write_sep("=", line)

    # Persist for CI to parse
    out_dir = Path(".tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    # If detailed metrics were written by the test, don't overwrite them.
    out_file = out_dir / "ft8r_short_metrics.json"
    if not out_file.exists():
        out_file.write_text(
            json.dumps({"matched": matched, "total": total, "percent": percent}, indent=2)
        )
