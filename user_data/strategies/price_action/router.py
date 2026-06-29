"""Deterministic routing from market diagnosis to strategy templates."""
from __future__ import annotations

from typing import Any

from .strategy_templates import StrategyTemplate, get_template


TREND_CYCLES = {"spike", "micro_channel", "tight_channel", "normal_channel"}
RANGE_CYCLES = {"broad_channel", "trending_tr", "trading_range"}


def route_strategies(diagnosis: dict[str, Any]) -> list[StrategyTemplate]:
    """Select one or two strategy templates from market diagnosis."""
    bar_analysis = diagnosis.get("bar_analysis") or {}

    cycle = str(diagnosis.get("cycle_position", "unknown")).lower()
    direction = _normalize_direction(diagnosis.get("direction", "neutral"))
    patterns = {
        str(p).lower()
        for p in (diagnosis.get("detected_patterns", []) or [])
        if p
    }
    setup = str(
        diagnosis.get("entry_setup")
        or bar_analysis.get("entry_setup_type")
        or ""
    ).lower()

    selected: list[str] = []
    failed_up = (
        ("failure" in setup and "up" in setup)
        or ("breakout_failure" in patterns and direction == "short")
    )
    failed_down = (
        ("failure" in setup and "down" in setup)
        or ("breakout_failure" in patterns and direction == "long")
    )
    breakout_up = "breakout_up" in patterns or setup in ("breakout", "breakout_pullback")
    breakout_down = "breakout_down" in patterns or setup in ("breakout", "breakout_pullback")

    if failed_down:
        selected.append("breakout_failure_long")
    elif failed_up:
        selected.append("breakout_failure_short")
    elif breakout_up and direction == "long":
        selected.append("breakout_continuation_long")
    elif breakout_down and direction == "short":
        selected.append("breakout_continuation_short")
    elif cycle in TREND_CYCLES and direction == "long":
        selected.append("trend_pullback_long")
    elif cycle in TREND_CYCLES and direction == "short":
        selected.append("trend_pullback_short")
    elif cycle in RANGE_CYCLES:
        if direction == "long":
            selected.append("range_reversal_long")
        elif direction == "short":
            selected.append("range_reversal_short")

    if not selected:
        if {"ii", "iii", "inside", "barbwire"} & patterns:
            selected.append("barbwire_wait")
        else:
            selected.append("ema20_magnet_wait")

    if len(selected) == 1 and selected[0] not in ("barbwire_wait", "ema20_magnet_wait"):
        if {"ii", "iii", "inside", "barbwire"} & patterns:
            selected.append("barbwire_wait")
        else:
            selected.append("ema20_magnet_wait")

    return [get_template(template_id) for template_id in selected[:2]]


def _normalize_direction(direction: object) -> str:
    value = str(direction or "neutral").lower()
    if value in ("long", "bull", "bullish", "up"):
        return "long"
    if value in ("short", "bear", "bearish", "down"):
        return "short"
    return "neutral"
