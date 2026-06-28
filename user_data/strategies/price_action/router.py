"""Deterministic L3 routing from L2 diagnosis to strategy templates."""
from __future__ import annotations

from typing import Any

from .strategy_templates import StrategyTemplate, get_template


def route_strategies(diagnosis: dict[str, Any]) -> list[StrategyTemplate]:
    """Select one or two strategy templates from L2 market diagnosis."""
    market_state = diagnosis.get("market_state") or {}
    signal_chain = diagnosis.get("signal_chain") or {}
    gate = diagnosis.get("gate") or {}

    cycle = str(market_state.get("cycle", "unknown")).lower()
    direction = _normalize_direction(
        signal_chain.get("direction") or market_state.get("direction") or "neutral"
    )
    gate_break = str(gate.get("breakout", "none")).lower()
    gate_position = str(gate.get("position", "unknown")).lower()
    patterns = {str(p).lower() for p in signal_chain.get("patterns", []) if p}
    setup = str(signal_chain.get("setup", "")).lower()

    selected: list[str] = []
    failed_up = gate_break == "failed_up" or ("failure" in setup and "up" in setup)
    failed_down = gate_break == "failed_down" or ("failure" in setup and "down" in setup)

    if failed_down:
        selected.append("breakout_failure_long")
    elif failed_up:
        selected.append("breakout_failure_short")
    elif gate_break in ("up", "both") and direction == "long":
        selected.append("breakout_continuation_long")
    elif gate_break in ("down", "both") and direction == "short":
        selected.append("breakout_continuation_short")
    elif cycle in ("trend", "always_in") and direction == "long":
        selected.append("trend_pullback_long")
    elif cycle in ("trend", "always_in") and direction == "short":
        selected.append("trend_pullback_short")
    elif cycle in ("trading_range", "range", "reversal"):
        if direction == "long" or gate_position in ("below", "testing_lower"):
            selected.append("range_reversal_long")
        elif direction == "short" or gate_position in ("above", "testing_upper"):
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
