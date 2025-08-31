"""Coarse-stage benchmark harness.

Implements a minimal CLI with four subcommands:
  - golden: generate golden labels from WSJT-X for a set of WAVs
  - run: produce coarse candidates using the existing search API
  - eval: match candidates to golden and compute metrics
  - report: create a Markdown summary and figures

All outputs are written under a timestamped folder per strategy in a bench
output root that is ignored by version control.
"""


