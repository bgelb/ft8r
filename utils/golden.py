import re

def normalize_wsjtx_message(msg: str) -> str:
    """Normalize a WSJT-X message string for comparison.

    - Removes appended metadata after runs of 2+ spaces (e.g., country/region hints)
    - Collapses remaining internal whitespace to single spaces
    - Trims leading/trailing spaces
    """
    s = (msg or "").strip()
    s = re.split(r"\s{2,}", s)[0]
    s = re.sub(r"\s+", " ", s)
    return s

