"""Breakout order price normalization (ported from PA_Agent).

Forces breakout ``entry_price`` to basis extreme +/- 1 tick, mirroring upstream
``PA_Agent/pa_agent/util/price_tick.py`` behavior. Called from
:func:`normalize_trade_decision` before :class:`DecisionValidator` so the
validator can use strict ``<=`` / ``>=`` without rejecting legitimate
trigger-at-extreme cases.

Unlike upstream, this port operates on ``feature_rows: list[dict[str, Any]]``
(with keys ``k``/``high``/``low``/...) instead of a ``kline_frame`` object with
``.bars[].seq/.high/.low`` attributes.
"""
from __future__ import annotations

import re
from typing import Any


def infer_price_tick_from_rows(
    feature_rows: list[dict[str, Any]] | None,
) -> float | None:
    """Guess one tick from decimal places in OHLC values.

    Mirrors upstream ``infer_price_tick_from_frame``. Returns ``None`` when no
    rows are available, ``1.0`` for integer prices, otherwise ``10**-d`` where
    ``d`` is the maximum decimal places observed (capped at 6).
    """
    if not feature_rows:
        return None
    max_decimals = 0
    for row in feature_rows:
        for attr in ("open", "high", "low", "close"):
            raw = row.get(attr)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            text = f"{value:.12f}".rstrip("0")
            if "." in text:
                max_decimals = max(max_decimals, len(text.split(".")[1]))
    if max_decimals <= 0:
        return 1.0
    return 10 ** (-min(max_decimals, 6))


def round_to_tick(price: float, tick: float) -> float:
    """Round ``price`` to the nearest multiple of ``tick``."""
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 10)


_K_SEQ_RE = re.compile(r"K\s*(\d+)", re.IGNORECASE)


def _parse_k_seq(value: Any) -> int | None:
    if value is None:
        return None
    m = _K_SEQ_RE.search(str(value))
    return int(m.group(1)) if m else None


def _feature_row_by_seq(
    feature_rows: list[dict[str, Any]] | None,
    seq: int,
) -> dict[str, Any] | None:
    for row in feature_rows or []:
        if _parse_k_seq(row.get("k")) == seq:
            return row
    return None


def canonical_breakout_extreme(order_direction: str) -> str | None:
    """Return schema-correct ``entry_basis_extreme`` for a breakout order."""
    direction = str(order_direction or "").strip()
    if direction == "做多":
        return "high"
    if direction == "做空":
        return "low"
    return None


def normalize_breakout_basis_extreme(decision: dict[str, Any]) -> bool:
    """Align ``entry_basis_extreme`` with ``order_direction`` (做空→low, 做多→high).

    Returns ``True`` when the field was changed.
    """
    if decision.get("order_type") != "突破单":
        return False
    want = canonical_breakout_extreme(str(decision.get("order_direction", "") or ""))
    if not want:
        return False
    have = str(decision.get("entry_basis_extreme", "") or "").strip().lower()
    if have == want:
        return False
    decision["entry_basis_extreme"] = want
    return True


def normalize_breakout_entry_price(
    decision: dict[str, Any],
    *,
    feature_rows: list[dict[str, Any]] | None = None,
    tick: float | None = None,
) -> bool:
    """Force ``entry_price`` to basis extreme +/- 1 tick for breakout orders.

    Recomputes ``entry_price`` from the cited ``entry_basis_bar``'s extreme
    regardless of what the AI provided. Returns ``True`` when adjusted.
    """
    if decision.get("order_type") != "突破单":
        return False
    if not feature_rows:
        return False

    basis_seq = _parse_k_seq(decision.get("entry_basis_bar"))
    if basis_seq is None:
        return False
    row = _feature_row_by_seq(feature_rows, basis_seq)
    if row is None:
        return False

    basis_high_raw = row.get("high")
    basis_low_raw = row.get("low")
    if basis_high_raw is None or basis_low_raw is None:
        return False
    try:
        basis_high = float(basis_high_raw)
        basis_low = float(basis_low_raw)
    except (TypeError, ValueError):
        return False

    direction = str(decision.get("order_direction", "") or "")
    extreme = str(decision.get("entry_basis_extreme", "") or "")
    step = (
        tick
        if tick and tick > 0
        else infer_price_tick_from_rows(feature_rows) or 0.01
    )

    target: float | None = None
    if direction == "做多" and extreme == "high":
        target = round_to_tick(basis_high + step, step)
    elif direction == "做空" and extreme == "low":
        target = round_to_tick(basis_low - step, step)
    if target is None:
        return False

    entry_raw = decision.get("entry_price")
    try:
        current = float(entry_raw) if entry_raw is not None else None
    except (TypeError, ValueError):
        current = None

    if current == target:
        return False

    decision["entry_price"] = target
    return True
