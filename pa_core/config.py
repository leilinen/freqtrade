"""Centralised path constants for pa_core.

Adapted from PA_Agent ``pa_agent/config/paths.py`` (baseline 1090a5b).

Differences vs. upstream: assets live inside the package (imports then work
from any cwd), and the runtime write directories are dropped — analysis
records are persisted to PostgreSQL by ``pa_core.records`` (later porting
step), not to a local ``records/pending`` directory.
"""
from __future__ import annotations

from pathlib import Path

# ── Root ──────────────────────────────────────────────────────────────────────
# pa_core/ package root (this file is pa_core/config.py)
PACKAGE_ROOT: Path = Path(__file__).resolve().parent

# ── Prompt engineering assets (read-only at runtime) ─────────────────────────
# 29 strategy/knowledge .txt files + _reference/ markdown docs
PROMPT_DIR: Path = PACKAGE_ROOT / "prompts"

# ── Experience library (success/failure cases per cycle_position) ─────────────
# Directory structure only for now (upstream ships it empty)
EXPERIENCE_DIR: Path = PACKAGE_ROOT / "experience"

# ── Analysis record JSON output (transitional) ────────────────────────────────
# PendingWriter drops AnalysisRecord JSON here until layer 6 replaces it with
# PostgreSQL persistence. Writable inside the dev checkout.
RECORDS_PENDING_DIR: Path = PACKAGE_ROOT / "records_pending"
