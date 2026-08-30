"""Time formatting utilities.

Ported verbatim from PA_Agent ``pa_agent/util/timefmt.py`` (baseline 1090a5b).
"""
from __future__ import annotations

import time


def now_local_ms() -> int:
    """Return current local time as milliseconds since epoch."""
    return int(time.time() * 1000)
