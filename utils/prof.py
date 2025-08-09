import json
import os
import time
from contextlib import contextmanager
from typing import Dict


class Profiler:
    """Minimal section profiler with near-zero overhead when disabled.

    Enable by setting environment variable ``FT8R_PROFILE=1``. Use via:

        from utils.prof import PROFILER
        with PROFILER.section("candidate_search"):
            ...
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled: bool = enabled
        self._sections: Dict[str, Dict[str, float | int]] = {}

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            rec = self._sections.get(name)
            if rec is None:
                rec = {"total": 0.0, "count": 0}
                self._sections[name] = rec
            rec["total"] = float(rec["total"]) + float(elapsed)
            rec["count"] = int(rec["count"]) + 1

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, rec in self._sections.items():
            total = float(rec.get("total", 0.0))
            count = int(rec.get("count", 0))
            avg = (total / count) if count else 0.0
            out[name] = {
                "total_s": total,
                "count": float(count),
                "avg_ms": avg * 1000.0,
            }
        return out

    def dump_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2, sort_keys=True)


PROFILER = Profiler(enabled=(os.getenv("FT8R_PROFILE", "0") not in ("0", "", "false", "False")))


