"""Schema normalize layer: repair common LLM JSON quirks before validation.

This module mirrors upstream PA_Agent's normalize stage (single pass, runs
between ``parse_json_object`` and ``validate_*``). All functions are
idempotent: running them twice yields the same output as running them once.

Public entry points:

* :func:`normalize_market_diagnosis` -- stage1 (gate_trace answer aliases)
* :func:`normalize_trade_decision` -- stage2 (next_cycle_prediction
  probabilities + decision_trace answer aliases)

PR1 scope: probability rescale/clamp/argmax + answer alias mapping.
PR2 will add bar_by_bar pad + bar_type/role/context_effect enum repair.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any

from .price_tick import (
    normalize_breakout_basis_extreme,
    normalize_breakout_entry_price,
)
from .pattern_routing import ensure_detected_patterns_coherent
from .coherence_checks import auto_fix_bar_by_bar_types
from .trace_normalize import (
    normalize_trace_list_bar_range,
    repair_stage1_gate_trace,
    repair_stage1_gate_trace_questions,
    repair_stage2_decision_trace_questions,
    repair_stage2_terminal,
    strip_ai_gate_14,
)

logger = logging.getLogger(__name__)

# ── Closed-enum normalization (ported from PA_Agent stage2_normalizer) ──
#
# LLMs often append annotations to closed enums (e.g. ``doji（十字星）``)
# or use synonyms (``strong`` → ``high``). ``_normalize_closed_enum`` strips
# the suffix and maps synonyms back to schema tokens before the validator
# rejects them. Applied to stage2 ``bar_analysis.{bar_type, entry_bar,
# signal_bar}`` enums.

# Separators that mark the start of an annotation: CJK + ASCII brackets,
# em/en dash, colon. Anything after the first separator is dropped.
_ENUM_SUFFIX_SEPARATORS: tuple[str, ...] = (
    "（", "(", "【", "[", "—", "–", " - ", "：", ":",
)


def _strip_enum_suffix(raw: str) -> str:
    """Drop trailing annotations models append to closed enums.

    Mirrors upstream ``_strip_enum_suffix``. Splits on the first CJK/ASCII
    bracket / dash / colon and returns the trimmed head.
    """
    text = raw.strip()
    for sep in _ENUM_SUFFIX_SEPARATORS:
        if sep in text:
            head = text.split(sep, 1)[0].strip()
            if head:
                return head
    return text


def _normalize_closed_enum(
    raw: object,
    allowed: frozenset[str],
    *,
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Map messy model enum text to a schema token, or None if unrecognized.

    Mirrors upstream ``_normalize_closed_enum``. Order: strip suffix →
    lowercase + underscore → alias map → membership check →
    longest-prefix match fallback.
    """
    if not isinstance(raw, str):
        return None
    text = _strip_enum_suffix(raw)
    key = text.strip().lower().replace(" ", "_")
    if aliases:
        key = aliases.get(key, key)
    if key in allowed:
        return key
    for token in sorted(allowed, key=len, reverse=True):
        if key.startswith(token):
            return token
    return None


_BAR_TYPE_ENUM = frozenset({
    "trend_bull", "trend_bear", "doji", "inside",
    "outside_bull", "outside_bear", "flat", "other",
})
_BAR_TYPE_ALIASES: dict[str, str] = {
    "ine": "inside",
    "ins": "inside",
    "insid": "inside",
    "doj": "doji",
    "trendbull": "trend_bull",
    "trendbear": "trend_bear",
    "outsidebull": "outside_bull",
    "outsidebear": "outside_bear",
}
_ENTRY_BAR_FRESHNESS_ENUM = frozenset({"fresh", "pending", "stale", "invalid"})
_ENTRY_BAR_FRESHNESS_ALIASES: dict[str, str] = {
    "expired": "stale",
    "old": "stale",
    "aged": "stale",
    "too_old": "stale",
    "active": "fresh",
    "ready": "fresh",
    "new": "fresh",
    "waiting": "pending",
    "trigger": "pending",
    "k0_trigger": "pending",
    "limit_order_pending": "pending",
    "limit_pending": "pending",
    "order_pending": "pending",
    "awaiting_fill": "pending",
    "awaiting_trigger": "pending",
}
_ENTRY_BAR_STRENGTH_ENUM = frozenset({"strong", "weak", "not_triggered"})
_ENTRY_BAR_STRENGTH_ALIASES: dict[str, str] = {
    "pending": "not_triggered",
    "waiting": "not_triggered",
    "triggered": "strong",
    "not_triggered": "not_triggered",
    "strong": "strong",
    "weak": "weak",
}
_SIGNAL_BAR_QUALITY_ENUM = frozenset({"strong", "medium", "weak", "invalid"})
_SIGNAL_BAR_QUALITY_ALIASES: dict[str, str] = {
    "low": "weak",
    "high": "strong",
    "moderate": "medium",
    "poor": "weak",
    "good": "strong",
    "bad": "invalid",
    "弱": "weak",
    "中": "medium",
    "强": "strong",
    "无效": "invalid",
}


def _stage1_bar_analysis_bar_type(stage1_json: dict[str, Any] | None) -> str | None:
    """Return the canonical bar_type from stage1 bar_analysis, if any."""
    if not isinstance(stage1_json, dict):
        return None
    bar_analysis = stage1_json.get("bar_analysis")
    if not isinstance(bar_analysis, dict):
        return None
    return _normalize_closed_enum(
        bar_analysis.get("bar_type"), _BAR_TYPE_ENUM, aliases=_BAR_TYPE_ALIASES
    )


def _normalize_entry_bar_freshness(entry_bar: dict[str, Any]) -> bool:
    raw = entry_bar.get("freshness")
    mapped = _normalize_closed_enum(
        raw,
        _ENTRY_BAR_FRESHNESS_ENUM,
        aliases=_ENTRY_BAR_FRESHNESS_ALIASES,
    )
    if mapped and mapped != raw:
        entry_bar["freshness"] = mapped
        return True
    return False


def _normalize_second_entry(second_entry: dict[str, Any]) -> bool:
    """``type`` must be a string; models often emit null when not a second entry."""
    raw_type = second_entry.get("type")
    if raw_type is not None and (
        isinstance(raw_type, str) and str(raw_type).strip()
    ):
        return False
    second_entry["type"] = "none"
    return True


def normalize_stage2_bar_analysis_enums(
    out: dict[str, Any],
    *,
    stage1_json: dict[str, Any] | None = None,
) -> bool:
    """Strip enum annotations and sync bar_type from stage1 when available.

    Mirrors upstream ``_normalize_stage2_bar_analysis_enums``. Repairs
    bar_type / entry_bar.{freshness, strength} / signal_bar.{quality,
    pattern, reason} / second_entry.type before validation.
    """
    changed = False
    bar_analysis = out.get("bar_analysis")
    if not isinstance(bar_analysis, dict):
        return False

    stage1_bt = _stage1_bar_analysis_bar_type(stage1_json)
    raw_bt = bar_analysis.get("bar_type")
    norm_bt = stage1_bt or _normalize_closed_enum(
        raw_bt, _BAR_TYPE_ENUM, aliases=_BAR_TYPE_ALIASES
    )
    if norm_bt and norm_bt != raw_bt:
        bar_analysis["bar_type"] = norm_bt
        changed = True

    entry_bar = bar_analysis.get("entry_bar")
    if isinstance(entry_bar, dict):
        if _normalize_entry_bar_freshness(entry_bar):
            changed = True
        raw_strength = entry_bar.get("strength")
        norm_strength = _normalize_closed_enum(
            raw_strength,
            _ENTRY_BAR_STRENGTH_ENUM,
            aliases=_ENTRY_BAR_STRENGTH_ALIASES,
        )
        if norm_strength and norm_strength != raw_strength:
            entry_bar["strength"] = norm_strength
            changed = True

    signal_bar = bar_analysis.get("signal_bar")
    if isinstance(signal_bar, dict):
        raw_q = signal_bar.get("quality")
        norm_q = _normalize_closed_enum(
            raw_q,
            _SIGNAL_BAR_QUALITY_ENUM,
            aliases=_SIGNAL_BAR_QUALITY_ALIASES,
        )
        if norm_q and norm_q != raw_q:
            signal_bar["quality"] = norm_q
            changed = True
        raw_pat = str(signal_bar.get("pattern", "") or "").strip().lower()
        if raw_pat in ("no_signal", "no-signal", "nosignal", "not_triggered"):
            signal_bar["pattern"] = "none"
            changed = True
        if not str(signal_bar.get("reason") or "").strip():
            signal_bar["reason"] = "无独立信号棒（quality=invalid 或计划型观望）"
            changed = True

    second_entry = bar_analysis.get("second_entry")
    if isinstance(second_entry, dict) and _normalize_second_entry(second_entry):
        changed = True

    return changed


# ── Decision no-order coercion (ported from PA_Agent stage2_normalizer) ──
#
# When the decision_trace or terminal indicates the trade was rejected
# (node 10.3=否, terminal.outcome in {wait, reject}, or §14 violation),
# the model often still emits a full trade decision with prices — a common
# slip that wastes a retry cycle. These helpers detect the rejection signals
# and clear the decision to 不下单 before validation, mirroring upstream
# behavior.

_TRADE_ORDER_TYPES = frozenset({"限价单", "突破单", "市价单"})

_ORDER_TYPE_ALIASES: dict[str, str] = {
    "no_order": "不下单",
    "notrade": "不下单",
    "no_trade": "不下单",
    "hold": "不下单",
    "skip": "不下单",
    "none": "不下单",
    "wait": "不下单",
    "limit": "限价单",
    "limit_order": "限价单",
    "breakout": "突破单",
    "breakout_order": "突破单",
    "market": "市价单",
    "market_order": "市价单",
}

_NO_ORDER_PRICE_FIELDS: tuple[str, ...] = (
    "order_direction",
    "entry_price",
    "take_profit_price",
    "take_profit_price_2",
    "stop_loss_price",
    "entry_basis_bar",
    "entry_basis_extreme",
    "entry_rule",
)

# Denial phrases that contradict answer=是 on §14 nodes. Some models write
# answer=是 to mean "I completed the scan" — we cross-check the reason text
# before treating it as a real violation.
_SECTION14_DENIAL_PHRASES: tuple[str, ...] = (
    "未触犯",
    "未违反",
    "无触犯",
    "无违规",
    "通过扫描",
    "扫描通过",
    "无禁止",
    "未触发",
)


def _normalize_order_type_aliases(decision: dict[str, Any]) -> bool:
    """Map English order_type slips (no_order, limit, …) to schema enums.

    Mirrors upstream ``_normalize_order_type_aliases``. Returns ``True`` when
    the field was changed.
    """
    raw = str(decision.get("order_type", "") or "").strip()
    if not raw:
        return False
    key = raw.lower().replace(" ", "_").replace("-", "_")
    mapped = _ORDER_TYPE_ALIASES.get(key) or _ORDER_TYPE_ALIASES.get(raw.lower())
    if mapped and mapped != raw:
        decision["order_type"] = mapped
        logger.debug("order_type %r -> %r", raw, mapped)
        return True
    return False


def _trace_node_answer(trace: Any, node_id: str) -> str | None:
    """Return the trimmed ``answer`` for the first trace item with ``node_id``."""
    if not isinstance(trace, list):
        return None
    for item in trace:
        if not isinstance(item, dict):
            continue
        if str(item.get("node_id", "")).strip() == node_id:
            return str(item.get("answer", "") or "").strip()
    return None


def _section14_violated(trace: Any) -> bool:
    """Return True only when §14 answer is 是 AND reason confirms violation.

    Cross-checks reason text against denial phrases because some models write
    answer=是 to mean "scan completed" rather than "violation found".
    """
    if not isinstance(trace, list):
        return False
    for item in trace:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("node_id", "") or "").strip()
        if not nid.startswith("14"):
            continue
        if str(item.get("answer", "") or "").strip() != "是":
            continue
        reason = str(item.get("reason", "") or "")
        if any(phrase in reason for phrase in _SECTION14_DENIAL_PHRASES):
            logger.debug(
                "_section14_violated: node %s answer=是 but reason contains "
                "denial phrase; treating as NOT violated",
                nid,
            )
            continue
        return True
    return False


def _clear_decision_to_no_order(decision: dict[str, Any]) -> None:
    """Force ``order_type`` to 不下单 and null out price/rule fields.

    Provides valid defaults for ``trade_confidence`` and
    ``trade_confidence_reasoning`` (schema-required non-null).
    """
    decision["order_type"] = "不下单"
    for field in _NO_ORDER_PRICE_FIELDS:
        decision[field] = None
    decision["estimated_win_rate"] = None
    decision["estimated_win_rate_reasoning"] = None
    if decision.get("trade_confidence") is None:
        decision["trade_confidence"] = 0
    existing_reasoning = decision.get("trade_confidence_reasoning")
    if not isinstance(existing_reasoning, str) or not existing_reasoning:
        decision["trade_confidence_reasoning"] = "无入场计划，不存在交易信心"


def _set_trace_node_answer(
    trace: Any,
    node_id: str,
    answer: str,
    *,
    reason_suffix: str = "",
) -> None:
    """Set ``answer`` (and optionally append a reason suffix) on a trace node.

    Ported from PA_Agent ``stage2_normalizer._set_trace_node_answer``. If the
    node is absent the call is a no-op (the validator/normalizer owns trace
    structure; we do not fabricate nodes here).
    """
    if not isinstance(trace, list):
        return
    for item in trace:
        if not isinstance(item, dict):
            continue
        if str(item.get("node_id", "")).strip() != node_id:
            continue
        item["answer"] = answer
        if reason_suffix:
            base = str(item.get("reason", "") or "").strip()
            item["reason"] = f"{base}{reason_suffix}".strip()
        return


# ── Planned-limit trace fix (ported from PA_Agent stage2_normalizer:1434-1516) ──
#
# When the model writes a valid planned-limit decision but leaves §9.0=否/等待,
# two fixes are required:
#   1. ``_fix_9_0_for_planned_limit``: upgrade §9.0 answer to 是 (limit plan
#      accepts weak/invalid signal bars or no signal bar at all).
#   2. ``_fix_background_limit_trace``: ensure §9.0P=是 is present in trace,
#      recording the background-driven limit path.
# Order matters: 9.0 first, then 9.0P (9.0P only fires when 9.0 ∈ {否,等待},
# so it is safe to run after 9.0 is upgraded — in that case it is a no-op).


def _fix_background_limit_trace(out: dict[str, Any]) -> bool:
    """Ensure §9.0P=是 when a planned limit order follows §9.0=否/等待.

    Direct port of upstream ``stage2_normalizer._fix_background_limit_trace``.
    """
    from .decision_nodes import is_planned_limit_order

    if not is_planned_limit_order(out):
        return False
    trace = out.get("decision_trace")
    if not isinstance(trace, list):
        return False

    node_90: dict[str, Any] | None = None
    node_90p: dict[str, Any] | None = None
    for item in trace:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("node_id", "")).strip()
        if nid == "9.0":
            node_90 = item
        elif nid == "9.0P":
            node_90p = item

    changed = False
    if node_90 is not None:
        ans = str(node_90.get("answer", "") or "").strip()
        if ans in ("否", "等待"):
            if node_90p is None:
                trace.insert(
                    trace.index(node_90) + 1,
                    {
                        "node_id": "9.0P",
                        "section": "入场信号",
                        "question": "背景驱动限价单评估（§9.0=否 时必须评估）",
                        "answer": "是",
                        "reason": (
                            "程序校正：计划型限价单，周期/结构位支持挂限价，"
                            "继续 §10 定三价。"
                        ),
                        "skipped": False,
                        "bar_range": "K10-K1",
                    },
                )
                changed = True
            elif str(node_90p.get("answer", "") or "").strip() in ("否", "等待"):
                node_90p["answer"] = "是"
                base = str(node_90p.get("reason", "") or "").strip()
                suffix = "（程序校正：背景限价路径，非信号棒路径。）"
                node_90p["reason"] = f"{base}{suffix}".strip() if base else suffix.strip()
                changed = True
    return changed


def _fix_9_0_for_planned_limit(out: dict[str, Any]) -> bool:
    """When model outputs a valid planned limit but §9.0=否/等待, upgrade to 是.

    Direct port of upstream ``stage2_normalizer._fix_9_0_for_planned_limit``.
    """
    from .decision_nodes import is_planned_limit_order

    if not is_planned_limit_order(out):
        return False
    trace = out.get("decision_trace")
    if not isinstance(trace, list):
        return False
    changed = False
    for item in trace:
        if not isinstance(item, dict):
            continue
        if str(item.get("node_id", "")).strip() != "9.0":
            continue
        ans = str(item.get("answer", "") or "").strip()
        if ans not in ("否", "等待"):
            return False
        item["answer"] = "是"
        base = str(item.get("reason", "") or "").strip()
        suffix = (
            "（程序校正：计划型限价单，接受 weak/invalid 或无信号棒，"
            "等待回撤/反弹到位入场，非等下一根确认棒后放弃。）"
        )
        item["reason"] = f"{base}{suffix}".strip() if base else suffix.strip()
        changed = True
        break
    return changed


# ── Trade-metrics veto (ported from PA_Agent stage2_normalizer:882-925) ──
#
# After the breakout entry snap and stop-widening, an order can still fail
# the risk/reward / trader-equation / K1-freshness / TP2-geometry checks
# (e.g. a malformed stop the LLM refuses to fix, or a stale limit order).
# Upstream forces such orders to 不下单 so a failed cycle still persists a
# valid "no-order" decision rather than an invalid one. See the plan's
# safety-net #3 (metrics 失败强制不下单).

def _coerce_decision_when_trade_metrics_fail(
    out: dict[str, Any],
    *,
    feature_rows: list[dict[str, Any]] | None = None,
    decision_stance: str | None = None,
) -> bool:
    """After breakout entry snap + stop widening, reject orders that still fail RR / trader equation / K1 freshness.

    Returns ``True`` when the decision was coerced to 不下单. Ported from
    upstream ``stage2_normalizer._coerce_decision_when_trade_metrics_fail``;
    the ``kline_frame`` argument is replaced by ``feature_rows``.
    """
    decision = out.get("decision")
    if not isinstance(decision, dict) or decision.get("order_type") not in _TRADE_ORDER_TYPES:
        return False
    # Require a complete entry/tp/sl triple. Incomplete price plans are left
    # for the validator (``actionable_decision_requires_full_price_plan``);
    # coercing a half-formed decision here would mask the real "missing
    # field" error and make unit tests of other normalize sub-features
    # (entry-snap, field-hoist) depend on metrics geometry they don't set.
    if any(
        decision.get(field) is None
        for field in ("entry_price", "take_profit_price", "stop_loss_price")
    ):
        return False

    # Planned-limit orders (background-driven pending limits) are exempt from
    # the trade-metrics veto: a pending limit may legitimately omit TP2 or
    # carry §9.0=否, and its geometry is finalised only when the limit fills.
    # Without this guard the coerce would flip a valid planned limit to
    # 不下单 before ``_fix_9_0_for_planned_limit`` / ``_fix_background_limit_trace``
    # can reconcile the trace.
    try:
        from .decision_nodes import is_planned_limit_order

        if is_planned_limit_order(out):
            return False
    except ImportError:
        pass

    from .trade_metrics import validate_order_trade_metrics

    bar_analysis = out.get("bar_analysis")
    metric_errors = validate_order_trade_metrics(
        decision,
        decision_stance=decision_stance,
        feature_rows=feature_rows,
        bar_analysis=bar_analysis if isinstance(bar_analysis, dict) else None,
    )
    if not metric_errors:
        return False

    summary = metric_errors[0]
    _clear_decision_to_no_order(decision)
    _set_trace_node_answer(
        out.get("decision_trace"),
        "10.3",
        "否",
        reason_suffix=f"（程序按 decision 三价校验未通过：{summary}，已改为不下单。）",
    )
    terminal = out.get("terminal")
    if isinstance(terminal, dict):
        terminal["outcome"] = "reject"
        terminal["node_id"] = "10.3"
        terminal.setdefault(
            "label",
            "交易者方程/盈亏比未达标，不下单",
        )
    logger.debug("Coerced decision to 不下单 (trade metrics: %s)", summary)
    return True


# ── Stage2 unwrap / required-fields / truncate (Batch D) ──
#
# When the model puts trade-decision fields at the JSON root instead of
# nested under `decision`, or writes decision as a scalar ("wait"/"reject"),
# `_unwrap_flat_stage2_decision` rebuilds the canonical structure before
# schema validation. `_ensure_decision_required_fields` fills missing
# non-null schema fields with sensible defaults derived from stage1.
# `_truncate_decision_reasoning` caps reasoning length to avoid verbose
# JSON overflowing the model's response budget.

# Cap decision.reasoning to keep JSON payload bounded.
DECISION_REASONING_MAX_LEN = 280

_DECISION_SUBFIELD_KEYS: frozenset[str] = frozenset({
    "order_direction",
    "order_type",
    "entry_price",
    "entry_basis_bar",
    "entry_basis_extreme",
    "entry_rule",
    "take_profit_price",
    "take_profit_price_2",
    "stop_loss_price",
    "reasoning",
    "diagnosis_confidence",
    "diagnosis_confidence_reasoning",
    "trade_confidence",
    "trade_confidence_reasoning",
    "estimated_win_rate",
    "estimated_win_rate_reasoning",
    "key_factors",
    "watch_points",
    "risk_assessment",
    "invalidation_condition",
})

# Maps scalar decision tokens and terminal-outcome-like strings to the
# canonical outcome. Used by `_order_type_from_decision_scalar` to translate
# "wait"/"reject" (a terminal-outcome hint) into 不下单.
_TERMINAL_OUTCOME_ALIASES: dict[str, str] = {
    "action": "trade",
    "execute": "trade",
    "execution": "trade",
    "place_order": "trade",
    "breakout_entry": "trade",
    "breakout": "trade",
    "limit_entry": "trade",
    "market_entry": "trade",
    "entry": "trade",
    "trade_entry": "trade",
    "no_trade": "wait",
    "no_order": "wait",
    "wait": "wait",
    "reject": "reject",
    "trade": "trade",
    "proceed": "proceed",
}


def _order_type_from_decision_scalar(value: str) -> str | None:
    """Map a scalar decision token (wait/reject/limit/…) to order_type.

    Mirrors upstream. Returns ``None`` for unrecognized tokens so the caller
    can fall back to 不下单 as the safe default.
    """
    token = str(value or "").strip().lower()
    if not token:
        return None
    if token in _ORDER_TYPE_ALIASES:
        return _ORDER_TYPE_ALIASES[token]
    normalized = token.replace(" ", "_").replace("-", "_")
    if normalized in _ORDER_TYPE_ALIASES:
        return _ORDER_TYPE_ALIASES[normalized]
    outcome = _TERMINAL_OUTCOME_ALIASES.get(token) or _TERMINAL_OUTCOME_ALIASES.get(
        normalized
    )
    if outcome in ("wait", "reject"):
        return "不下单"
    return None


# ── §stage2 enum alias normalization (mirror stage2_normalizer:36-47, 316-426) ─
# Maps order_direction synonyms (long/buy/bullish/…) to the canonical 做多/做空.
_ORDER_DIRECTION_ALIASES: dict[str, str] = {
    "bearish": "做空",
    "bullish": "做多",
    "short": "做空",
    "long": "做多",
    "sell": "做空",
    "buy": "做多",
    "空头": "做空",
    "多头": "做多",
    "做空": "做空",
    "做多": "做多",
}


def _normalize_order_direction_value(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text in ("做多", "做空"):
        return text
    return _ORDER_DIRECTION_ALIASES.get(text.lower())


def _normalize_always_in_value(
    raw: object,
    *,
    diagnosis_direction: str | None = None,
) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    key = text.lower().replace(" ", "")
    if key in ("long", "short", "neutral"):
        return key
    if "失效" in text or "invalid" in key or key in ("none", "n/a", "na"):
        return "neutral"
    if "ais" in key or "空头" in text:
        return "short"
    if "ail" in key or "多头" in text:
        return "long"
    if "bear" in key:
        return "short"
    if "bull" in key:
        return "long"
    if "中性" in text or key == "neutral":
        return "neutral"
    if diagnosis_direction == "bearish":
        return "short"
    if diagnosis_direction == "bullish":
        return "long"
    return None


def _normalize_terminal_outcome_value(
    raw: object,
    *,
    order_type: str | None = None,
) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    key = text.lower().replace(" ", "_")
    mapped = _TERMINAL_OUTCOME_ALIASES.get(key)
    if mapped:
        if order_type == "不下单" and mapped == "trade":
            return "wait"
        return mapped
    if key in ("wait", "reject", "trade", "proceed"):
        return key
    return None


def _normalize_stage2_enum_aliases(out: dict[str, Any]) -> bool:
    """Map common enum slips before schema validation.

    Normalizes ``decision.order_direction`` (bullish/short/buy → 做多/做空),
    ``bar_analysis.always_in`` (AIS/空头/bear → short), and
    ``terminal.outcome`` (action/execute → trade, with 不下单→wait downgrade).
    Mirrors upstream ``_normalize_stage2_enum_aliases``.
    """
    changed = False
    diag = out.get("diagnosis_summary")
    diag_direction = (
        str(diag.get("direction", "")).strip()
        if isinstance(diag, dict)
        else ""
    ) or None

    decision = out.get("decision")
    order_type = (
        str(decision.get("order_type", "")).strip()
        if isinstance(decision, dict)
        else None
    ) or None
    if isinstance(decision, dict):
        raw_dir = decision.get("order_direction")
        mapped_dir = _normalize_order_direction_value(raw_dir)
        if mapped_dir and mapped_dir != raw_dir:
            decision["order_direction"] = mapped_dir
            logger.debug("order_direction %r -> %r", raw_dir, mapped_dir)
            changed = True

    bar_analysis = out.get("bar_analysis")
    if isinstance(bar_analysis, dict):
        raw_ai = bar_analysis.get("always_in")
        mapped_ai = _normalize_always_in_value(
            raw_ai, diagnosis_direction=diag_direction
        )
        if mapped_ai and mapped_ai != raw_ai:
            bar_analysis["always_in"] = mapped_ai
            logger.debug("always_in %r -> %r", raw_ai, mapped_ai)
            changed = True

    terminal = out.get("terminal")
    if isinstance(terminal, dict):
        raw_outcome = terminal.get("outcome")
        mapped_outcome = _normalize_terminal_outcome_value(
            raw_outcome, order_type=order_type
        )
        if mapped_outcome and mapped_outcome != raw_outcome:
            terminal["outcome"] = mapped_outcome
            logger.debug("terminal.outcome %r -> %r", raw_outcome, mapped_outcome)
            changed = True

    return changed


def _unwrap_flat_stage2_decision(out: dict[str, Any]) -> bool:
    """Repair models that put decision fields at root or use decision=scalar.

    Mirrors upstream ``_unwrap_flat_stage2_decision``. Hoists any
    ``_DECISION_SUBFIELD_KEYS`` found at the top level into ``out['decision']``.
    When ``decision`` itself is a scalar string, builds a fresh dict with
    ``order_type`` derived from the scalar (falling back to 不下单).
    """
    changed = False
    hoisted: dict[str, Any] = {}
    for key in _DECISION_SUBFIELD_KEYS:
        if key in out:
            hoisted[key] = out.pop(key)
            changed = True

    raw = out.get("decision")
    if isinstance(raw, str):
        order_type = _order_type_from_decision_scalar(raw) or "不下单"
        decision: dict[str, Any] = {"order_type": order_type}
        decision.update(hoisted)
        out["decision"] = decision
        logger.debug(
            "Unwrapped scalar decision %r -> order_type=%s with %d hoisted fields",
            raw,
            order_type,
            len(hoisted),
        )
        return True

    if isinstance(raw, dict):
        for key, val in hoisted.items():
            existing = raw.get(key)
            if key not in raw or existing is None or existing == "" or existing == []:
                raw[key] = val
                changed = True
        return changed

    if hoisted:
        out["decision"] = hoisted
        logger.debug("Built decision object from %d hoisted root fields", len(hoisted))
        return True
    return changed


def _hoist_terminal_from_decision(out: dict[str, Any]) -> bool:
    """Move ``terminal`` nested under ``decision`` to the top level.

    Some models nest terminal inside the decision object. Schema expects
    terminal at root. Mirrors upstream ``_hoist_terminal_from_decision``.
    """
    if isinstance(out.get("terminal"), dict):
        return False
    decision = out.get("decision")
    if not isinstance(decision, dict):
        return False
    nested = decision.pop("terminal", None)
    if not isinstance(nested, dict):
        return False
    out["terminal"] = nested
    logger.debug("Hoisted terminal from decision to top level")
    return True


def _ensure_decision_required_fields(
    out: dict[str, Any],
    *,
    stage1_json: dict[str, Any] | None = None,
) -> bool:
    """Fill missing decision sub-fields that commonly trigger schema retries.

    Mirrors upstream. Adds empty lists for key_factors/watch_points,
    sensible text defaults for reasoning/diagnosis_confidence_reasoning/
    risk_assessment, numeric defaults for diagnosis_confidence and
    trade_confidence, and a label for terminal when missing.
    """
    decision = out.get("decision")
    if not isinstance(decision, dict):
        return False
    s1 = stage1_json or {}
    changed = _normalize_order_type_aliases(decision)
    if not isinstance(decision.get("key_factors"), list):
        decision["key_factors"] = []
        changed = True
    if not isinstance(decision.get("watch_points"), list):
        decision["watch_points"] = []
        changed = True
    text_defaults = {
        "reasoning": "基于阶段一诊断与当前K线结构的阶段二决策说明",
        "diagnosis_confidence_reasoning": (
            str(s1.get("htf_context") or "").strip()[:500]
            or "依据阶段一诊断与闸门结论"
        ),
        "risk_assessment": "见 watch_points 与 invalidation_condition",
    }
    for key, default in text_defaults.items():
        if not isinstance(decision.get(key), str) or not str(decision.get(key)).strip():
            decision[key] = default
            changed = True
    if decision.get("diagnosis_confidence") is None:
        try:
            decision["diagnosis_confidence"] = int(s1.get("diagnosis_confidence") or 50)
        except (TypeError, ValueError):
            decision["diagnosis_confidence"] = 50
        changed = True
    if decision.get("trade_confidence") is None:
        decision["trade_confidence"] = (
            0 if decision.get("order_type") == "不下单" else 50
        )
        changed = True
    if (
        not isinstance(decision.get("trade_confidence_reasoning"), str)
        or not decision["trade_confidence_reasoning"].strip()
    ):
        decision["trade_confidence_reasoning"] = (
            "无入场计划，不存在交易信心"
            if decision.get("order_type") == "不下单"
            else "基于结构与入场方案的综合评估"
        )
        changed = True
    if decision.get("order_type") == "不下单":
        if "estimated_win_rate" not in decision:
            decision["estimated_win_rate"] = None
            changed = True
    elif decision.get("estimated_win_rate") is None:
        decision["estimated_win_rate"] = 50
        changed = True
    if decision.get("estimated_win_rate_reasoning") is not None and not isinstance(
        decision.get("estimated_win_rate_reasoning"), str
    ):
        decision["estimated_win_rate_reasoning"] = None
        changed = True
    elif "estimated_win_rate_reasoning" not in decision:
        decision["estimated_win_rate_reasoning"] = (
            None
            if decision.get("order_type") == "不下单"
            else "基于入场/止损/目标三价与结构背景的胜率估算"
        )
        changed = True
    elif (
        decision.get("order_type") != "不下单"
        and isinstance(decision.get("estimated_win_rate_reasoning"), str)
        and not str(decision.get("estimated_win_rate_reasoning")).strip()
    ):
        decision["estimated_win_rate_reasoning"] = "基于入场/止损/目标三价与结构背景的胜率估算"
        changed = True
    terminal = out.get("terminal")
    if isinstance(terminal, dict) and not str(terminal.get("label") or "").strip():
        outcome = str(terminal.get("outcome") or "wait")
        terminal["label"] = {
            "trade": "执行下单方案",
            "reject": "交易者方程未通过",
            "wait": "等待更好 setup",
            "proceed": "继续评估",
        }.get(outcome, "阶段二终局")
        changed = True
    return changed


def _truncate_decision_reasoning(decision: dict[str, Any]) -> bool:
    """Cap ``decision.reasoning`` length to avoid verbose JSON.

    Mirrors upstream ``_truncate_decision_reasoning``. Returns True when
    the field was modified.
    """
    reasoning = decision.get("reasoning")
    if not isinstance(reasoning, str):
        return False
    text = reasoning.strip()
    if len(text) <= DECISION_REASONING_MAX_LEN:
        if text != reasoning:
            decision["reasoning"] = text
            return True
        return False
    decision["reasoning"] = text[: DECISION_REASONING_MAX_LEN - 1] + "…"
    return True


# Decision fields models sometimes nest under diagnosis_summary by mistake.
_DECISION_FIELDS_FROM_DIAG_SUMMARY: tuple[str, ...] = (
    "estimated_win_rate_reasoning",
    "estimated_win_rate",
    "trade_confidence_reasoning",
    "diagnosis_confidence_reasoning",
    "diagnosis_confidence",
    "trade_confidence",
)


def repair_diagnosis_summary_and_decision(
    out: dict[str, Any],
    *,
    stage1_json: dict[str, Any] | None = None,
) -> bool:
    """Hoist misplaced decision fields out of diagnosis_summary, fill defaults.

    Mirrors upstream ``_repair_diagnosis_summary_and_decision``. The model
    sometimes writes trade_confidence / estimated_win_rate / etc. under
    ``diagnosis_summary`` instead of ``decision`` — this hoists them back
    to the canonical location and fills missing diagnosis_summary schema
    fields (cycle_position, direction, key_signals) from stage1.
    """
    decision = out.get("decision")
    if not isinstance(decision, dict):
        return False
    s1 = stage1_json or {}
    dsum = out.get("diagnosis_summary")
    if not isinstance(dsum, dict):
        return False

    changed = False
    for key in _DECISION_FIELDS_FROM_DIAG_SUMMARY:
        if key not in dsum:
            continue
        val = dsum.get(key)
        existing = decision.get(key)
        if existing not in (None, "", []) and key in decision:
            dsum.pop(key, None)
            changed = True
            continue
        if val in (None, "", []):
            dsum.pop(key, None)
            continue
        decision[key] = val
        dsum.pop(key, None)
        logger.debug("Hoisted diagnosis_summary.%s -> decision.%s", key, key)
        changed = True

    if not str(dsum.get("cycle_position") or "").strip():
        dsum["cycle_position"] = str(s1.get("cycle_position") or "unknown")
        changed = True
    if not str(dsum.get("direction") or "").strip():
        dsum["direction"] = str(s1.get("direction") or "neutral")
        changed = True
    if not isinstance(dsum.get("key_signals"), list):
        key_signals = dsum.get("key_signals")
        if not isinstance(key_signals, list):
            key_signals = list(s1.get("key_signals") or [])
        dsum["key_signals"] = key_signals
        changed = True
    return changed


def _coerce_decision_no_order(out: dict[str, Any]) -> bool:
    """When trace/terminal reject a trade, clear decision prices (common model slip).

    Returns ``True`` when the decision was coerced to 不下单. Mirrors upstream
    ``_coerce_decision_no_order``. Triggers:

    * ``terminal.outcome`` in {wait, reject}
    * ``decision_trace`` node 10.3 answer = 否
    * §14 node answer = 是 (with no denial phrase in reason)
    """
    decision = out.get("decision")
    if not isinstance(decision, dict):
        return False
    _normalize_order_type_aliases(decision)

    terminal = out.get("terminal")
    outcome = (
        str(terminal.get("outcome", "") or "").strip()
        if isinstance(terminal, dict)
        else ""
    )
    order_type = decision.get("order_type")

    if order_type not in _TRADE_ORDER_TYPES:
        # Already 不下单 (or unmapped): only react to terminal/outcome mismatch.
        if outcome in ("wait", "reject") and order_type != "不下单":
            _clear_decision_to_no_order(decision)
            logger.debug(
                "Coerced %r + terminal=%s to 不下单", order_type, outcome
            )
            return True
        return False

    # Planned-limit orders (background-driven pending limits) are legitimate
    # and must not be coerced to 不下单 by trace/terminal rejection triggers —
    # they often carry §9.0=否 (no closed signal bar) yet still warrant a limit
    # plan. Let the dedicated ``_fix_9_0_for_planned_limit`` /
    # ``_fix_background_limit_trace`` normalizers (run later in the pipeline)
    # reconcile the trace instead. (Mirrors the guard intent of upstream
    # ``_coerce_decision_no_order`` for the planned-limit path.)
    if order_type in _TRADE_ORDER_TYPES:
        try:
            from .decision_nodes import is_planned_limit_order

            if is_planned_limit_order(out):
                return False
        except ImportError:
            pass

    trace = out.get("decision_trace")
    triggers: list[str] = []
    if _trace_node_answer(trace, "10.3") == "否":
        triggers.append("10.3=否")
    if outcome in ("wait", "reject"):
        triggers.append(f"terminal.outcome={outcome}")
    if _section14_violated(trace):
        triggers.append("§14触犯")

    if not triggers:
        return False

    _clear_decision_to_no_order(decision)
    logger.debug("Coerced decision to 不下单 (%s)", ", ".join(triggers))
    return True


# ── Cycle ordering (must stay in sync with validation.TRADE_DECISION_CYCLE_ORDER) ──

_CYCLE_ORDER: tuple[str, ...] = (
    "spike",
    "micro_channel",
    "tight_channel",
    "normal_channel",
    "broad_channel",
    "trending_tr",
    "trading_range",
    "extreme_tr",
)

_VALID_FEATURES_USED: frozenset[str] = frozenset({
    "stage1_diagnosis",
    "kline_features",
    "analysis_history",
    "experience_library",
    "stage2_decision",
    "previous_prediction_summary",
})

# ── Trace answer alias tables (ported from upstream trace_normalize.py:36-156) ──

_GATE_RESULT_ANSWER_ALIASES: dict[str, str] = {
    "proceed": "是",
    "wait": "等待",
    "unknown": "中性",
}

# node_id -> {raw answer -> (canonical answer, branch)}
_NODE_ANSWER_BY_ID: dict[str, dict[str, tuple[str, str]]] = {
    "2.3": {
        "多头": ("是", "bullish"),
        "空头": ("是", "bearish"),
        "做多": ("是", "bullish"),
        "做空": ("是", "bearish"),
        "bullish": ("是", "bullish"),
        "bearish": ("是", "bearish"),
        "bull": ("是", "bullish"),
        "bear": ("是", "bearish"),
        "中性": ("中性", "neutral"),
        "neutral": ("中性", "neutral"),
    },
    "4.2": {
        "上涨": ("是", "bullish"),
        "下跌": ("是", "bearish"),
        "上涨通道": ("是", "bullish"),
        "下跌通道": ("是", "bearish"),
        "多头": ("是", "bullish"),
        "空头": ("是", "bearish"),
        "bullish": ("是", "bullish"),
        "bearish": ("是", "bearish"),
    },
    "6.2": {
        "普通交易区间": ("是", "trading_range"),
        "普通区间": ("是", "trading_range"),
        "普通": ("是", "trading_range"),
        "趋势型交易区间": ("是", "trending_tr"),
        "趋势型区间": ("是", "trending_tr"),
        "趋势型": ("是", "trending_tr"),
        "trading_range": ("是", "trading_range"),
        "trending_tr": ("是", "trending_tr"),
    },
    "6.3": {
        "下边缘": ("是", "lower"),
        "上边缘": ("是", "upper"),
        "在下边缘": ("是", "lower"),
        "在上边缘": ("是", "upper"),
        "区间下边缘": ("是", "lower"),
        "区间上边缘": ("是", "upper"),
        "下边缘附近": ("是", "lower"),
        "上边缘附近": ("是", "upper"),
        "中间": ("否", "middle"),
        "中间1/3": ("否", "middle"),
        "在中间": ("否", "middle"),
        "中间区域": ("否", "middle"),
        "不在边缘": ("否", "middle"),
        "lower": ("是", "lower"),
        "upper": ("是", "upper"),
        "middle": ("否", "middle"),
    },
    "8.2": {
        "楔形回撤": ("是", "pullback"),
        "楔形反转": ("是", "reversal"),
        "回撤": ("是", "pullback"),
        "反转": ("是", "reversal"),
        "pullback": ("是", "pullback"),
        "reversal": ("是", "reversal"),
    },
    "3.5": {
        "路径A": ("是", "path_a"),
        "路径B": ("是", "path_b"),
        "路径C": ("是", "path_c"),
        "A": ("是", "path_a"),
        "B": ("是", "path_b"),
        "C": ("是", "path_c"),
    },
    "2.2": {
        "冲突": ("是", "conflict"),
        "背景冲突": ("是", "conflict"),
        "方向冲突": ("是", "conflict"),
        "新旧冲突": ("是", "conflict"),
        "conflict": ("是", "conflict"),
        "同向": ("是", "aligned"),
        "共振": ("是", "aligned"),
        "方向一致": ("是", "aligned"),
        "aligned": ("是", "aligned"),
        "中性背景": ("中性", "neutral_background"),
        "背景中性": ("中性", "neutral_background"),
        "neutral_background": ("中性", "neutral_background"),
        "mixed": ("中性", "mixed"),
    },
}

_GENERIC_ANSWER: dict[str, str] = {
    "通过": "是",
    "未通过": "否",
    "不通过": "否",
    "违反": "否",
    "触犯": "否",
    "无交易计划，不存在触犯": "否",
    "未触犯": "否",
    "不存在触犯": "否",
    "pass": "是",
    "fail": "否",
    "yes": "是",
    "no": "否",
    "not_applicable": "不适用",
    "n/a": "不适用",
    "na": "不适用",
    # Common AI synonyms outside the strict enum (map before schema validation).
    "部分": "中性",
    "部分一致": "中性",
    "部分通过": "中性",
    "部分符合": "中性",
    "部分是": "中性",
    "部分否": "否",
    "待确认": "等待",
    "待定": "等待",
    "需确认": "等待",
    "尚未确认": "等待",
    "未确认": "等待",
    "不确定": "中性",
    # Terminal decision node (10.x) business synonyms.
    "不下单": "否",
    "不入场": "否",
    "放弃": "否",
    "不交易": "否",
    "下单": "是",
    "入场": "是",
    "交易": "是",
    "做多": "是",
    "做空": "是",
    "买入": "是",
    "卖出": "是",
    "等待观察": "等待",
    "继续观察": "等待",
    "观察": "等待",
}

_COMPOSITE_ANSWER_RE = re.compile(
    r"^(是|否|中性|等待|不适用)\s*[（(](.+?)[）)]\s*$"
)


# ── Probability default synthesis (ported from stage2_normalizer.py:1344-1364) ──

def _default_cycle_probs(cycle: str) -> dict[str, int]:
    """Build a default probability distribution centered on ``cycle``."""
    c = (cycle or "unknown").strip().lower()
    base: dict[str, int] = {k: 0 for k in _CYCLE_ORDER}
    if c in base:
        base[c] = 55
        rest = 45 // max(len(_CYCLE_ORDER) - 1, 1)
        for k in _CYCLE_ORDER:
            if k != c:
                base[k] = rest
        diff = 100 - sum(base.values())
        base[c] = max(0, base[c] + diff)
    else:
        base["broad_channel"] = 30
        base["trading_range"] = 25
        base["normal_channel"] = 20
        base["trending_tr"] = 15
        base["spike"] = 10
    return base


def _default_bar_probs(direction: str) -> dict[str, int]:
    """Build a default bar-direction probability distribution (mirror SOURCE:1335)."""
    d = (direction or "neutral").strip().lower()
    if d == "bullish":
        return {"bullish": 45, "bearish": 30, "neutral": 25}
    if d == "bearish":
        return {"bearish": 45, "bullish": 30, "neutral": 25}
    return {"neutral": 40, "bearish": 30, "bullish": 30}


# ── terminal / entry_bar / predictions repair (mirror SOURCE:773-879,1367-1423) ─
# Local K-seq parser (same regex as price_tick._parse_k_seq / decision_nodes._seq_from_k).
_K_SEQ_RE = re.compile(r"K\s*(\d+)", re.IGNORECASE)


def _parse_k_seq_local(value: Any) -> int | None:
    m = _K_SEQ_RE.search(str(value or ""))
    return int(m.group(1)) if m else None


def _repair_terminal_trade_node(out: dict[str, Any]) -> bool:
    """A successful trade should not terminate at §14 (prohibition scan)."""
    decision = out.get("decision")
    terminal = out.get("terminal")
    trace = out.get("decision_trace")
    if not isinstance(decision, dict) or not isinstance(terminal, dict):
        return False
    if decision.get("order_type") not in _TRADE_ORDER_TYPES:
        return False
    if terminal.get("outcome") != "trade":
        return False

    node_id = str(terminal.get("node_id", "") or "").strip()
    if not node_id.startswith("14"):
        return False

    replacement: str | None = None
    if isinstance(trace, list):
        for item in reversed(trace):
            if not isinstance(item, dict):
                continue
            nid = str(item.get("node_id", "") or "").strip()
            if nid.startswith("11."):
                replacement = nid
                break
        if replacement is None:
            for item in reversed(trace):
                if not isinstance(item, dict):
                    continue
                if str(item.get("node_id", "") or "").strip() == "10.3":
                    replacement = "10.3"
                    break

    if replacement is None:
        replacement = "10.3"
    terminal["node_id"] = replacement
    logger.debug("terminal.node_id %r -> %r (trade cannot terminate at §14)", node_id, replacement)
    return True


def _normalize_market_order_entry_bar(
    bar_analysis: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    """Market orders need a concrete entry_bar; borrow signal_bar when model left it pending."""
    if decision.get("order_type") != "市价单":
        return False
    entry_bar = bar_analysis.get("entry_bar")
    signal_bar = bar_analysis.get("signal_bar")
    if not isinstance(entry_bar, dict) or not isinstance(signal_bar, dict):
        return False
    if entry_bar.get("bar") is not None:
        return False
    sig_bar = signal_bar.get("bar")
    if not sig_bar:
        return False
    # Market order fills on the latest closed bar; signal_bar stays older (K2+).
    entry_bar["bar"] = str(bar_analysis.get("last_closed_bar") or "K1").strip() or "K1"
    raw_strength = str(entry_bar.get("strength") or signal_bar.get("quality") or "weak").strip().lower()
    strength_map = {"strong": "strong", "medium": "weak", "weak": "weak", "low": "weak", "high": "strong"}
    entry_bar["strength"] = strength_map.get(raw_strength, "weak")
    entry_bar["freshness"] = "fresh"
    entry_bar["follow_through"] = True
    entry_bar["still_valid"] = entry_bar.get("still_valid", True)
    logger.debug("market order: entry_bar.bar set from signal_bar %s", sig_bar)
    return True


def _normalize_signal_entry_bar_chain(bar_analysis: dict[str, Any], decision: dict[str, Any]) -> bool:
    """Signal K must be strictly older than entry K (larger seq); pending entry exempt."""
    if decision.get("order_type") not in _TRADE_ORDER_TYPES:
        return False
    signal_bar = bar_analysis.get("signal_bar")
    entry_bar = bar_analysis.get("entry_bar")
    if not isinstance(signal_bar, dict) or not isinstance(entry_bar, dict):
        return False

    strength = str(entry_bar.get("strength", "") or "").strip().lower()
    freshness = str(entry_bar.get("freshness", "") or "").strip().lower()
    pending = (
        strength == "not_triggered"
        or not entry_bar.get("bar")
        or freshness in ("pending", "stale", "invalid")
    )
    if pending:
        entry_bar["bar"] = None
        entry_bar["strength"] = "not_triggered"
        entry_bar.setdefault("freshness", "pending")
        if entry_bar.get("follow_through") in (None, "", False):
            entry_bar["follow_through"] = "pending"
        return False

    signal_seq = _parse_k_seq_local(signal_bar.get("bar"))
    entry_seq = _parse_k_seq_local(entry_bar.get("bar"))
    if signal_seq is None or entry_seq is None:
        return False
    if signal_seq > entry_seq:
        return False

    signal_bar["bar"] = f"K{entry_seq + 1}"
    logger.debug(
        "signal_bar K%s -> K%s (must be older than entry K%s)",
        signal_seq,
        entry_seq + 1,
        entry_seq,
    )
    return True


def ensure_stage2_predictions(
    out: dict[str, Any],
    *,
    stage1_json: dict[str, Any] | None = None,
    skip_next_bar: bool = False,
) -> bool:
    """Inject next_bar/next_cycle prediction stubs when the model omitted them."""
    changed = False
    diag = out.get("diagnosis_summary") if isinstance(out.get("diagnosis_summary"), dict) else {}
    s1 = stage1_json or {}
    direction = str(diag.get("direction") or s1.get("direction") or "neutral")
    cycle = str(diag.get("cycle_position") or s1.get("cycle_position") or "unknown")

    decision = out.get("decision") if isinstance(out.get("decision"), dict) else {}
    reasoning = str(decision.get("reasoning") or "").strip()
    synth_note = "（程序根据阶段二诊断摘要补全，原模型未输出预测字段）"

    if not skip_next_bar and not isinstance(out.get("next_bar_prediction"), dict):
        probs = _default_bar_probs(direction)
        dom = max(probs, key=probs.get)  # type: ignore[arg-type]
        out["next_bar_prediction"] = {
            "direction": dom,
            "probabilities": probs,
            "unpredictable": False,
            "reasoning": (
                (reasoning[:400] + "…") if len(reasoning) > 400 else reasoning
            ) or f"基于当前方向 {direction} 的参考预测{synth_note}",
            "features_used": ["stage1_diagnosis", "stage2_decision"],
        }
        changed = True

    if not isinstance(out.get("next_cycle_prediction"), dict):
        c_probs = _default_cycle_probs(cycle)
        dom_c = max(c_probs, key=c_probs.get)  # type: ignore[arg-type]
        out["next_cycle_prediction"] = {
            "cycle": dom_c,
            "direction": direction if direction in ("bullish", "bearish", "neutral") else "neutral",
            "probabilities": c_probs,
            "unpredictable": False,
            "reasoning": (
                f"当前周期 {cycle}，方向 {direction}。"
                f"下一周期概率为程序参考分布{synth_note}"
            ),
            "features_used": ["stage1_diagnosis", "stage2_decision"],
        }
        changed = True

    return changed


# ── next_cycle_prediction normalize (ported from stage2_normalizer.py:928-1063) ──

def _normalize_next_cycle_prediction(
    prediction: dict[str, Any],
    *,
    stage1_json: dict[str, Any] | None = None,
) -> None:
    """In-place normalize next_cycle_prediction common model quirks. Idempotent.

    Covers: stray-key migration, unpredictable fallback, features_used
    filtering, reasoning truncation, probability int/clamp/rescale/argmax.
    """
    if not isinstance(prediction, dict):
        return

    # Migrate alias keys into canonical ``cycle`` field.
    for alt_key in ("predicted_next_cycle", "next_cycle"):
        alt_val = prediction.get(alt_key)
        if alt_val and not prediction.get("cycle"):
            prediction["cycle"] = str(alt_val).strip().lower()
    for stray in (
        "current_cycle",
        "predicted_next_cycle",
        "next_cycle",
        "confidence",
    ):
        prediction.pop(stray, None)

    # primary/secondary shorthand → cycle
    primary = prediction.pop("primary", None)
    prediction.pop("primary_probability", None)
    prediction.pop("secondary", None)
    prediction.pop("secondary_probability", None)
    if primary and not prediction.get("cycle"):
        prediction["cycle"] = str(primary).strip().lower()

    prediction.setdefault("unpredictable", False)
    unpredictable = bool(prediction.get("unpredictable", False))
    prediction["unpredictable"] = unpredictable

    # features_used: filter to schema enum + ensure stage1_diagnosis present.
    feats = prediction.get("features_used")
    if not isinstance(feats, list):
        feats = []
    feats = [f for f in feats if isinstance(f, str)]
    feats = [f for f in feats if f in _VALID_FEATURES_USED]
    if "stage1_diagnosis" not in feats:
        feats.insert(0, "stage1_diagnosis")
    seen: set[str] = set()
    deduped: list[str] = []
    for f in feats:
        if f not in seen:
            deduped.append(f)
            seen.add(f)
    prediction["features_used"] = deduped

    # reasoning truncation
    reasoning = prediction.get("reasoning")
    if isinstance(reasoning, str) and len(reasoning) > 1500:
        prediction["reasoning"] = reasoning[:1499] + "…"
    elif not isinstance(reasoning, str):
        prediction["reasoning"] = ""

    if unpredictable:
        prediction["cycle"] = None
        prediction["direction"] = None
        prediction["probabilities"] = None
        return

    # probabilities: int round + clamp + rescale sum=100 + cycle=argmax
    probs = prediction.get("probabilities")
    if not isinstance(probs, dict):
        cycle_guess = str(
            prediction.get("cycle")
            or (stage1_json or {}).get("cycle_position")
            or "trading_range"
        ).strip().lower()
        prediction["probabilities"] = _default_cycle_probs(cycle_guess)
        probs = prediction["probabilities"]

    if isinstance(probs, dict):
        normalized: dict[str, int] = {}
        for key in _CYCLE_ORDER:
            raw = probs.get(key)
            try:
                value = int(round(float(raw))) if raw is not None else 0
            except (TypeError, ValueError):
                value = 0
            normalized[key] = max(0, min(100, value))

        total = sum(normalized[k] for k in _CYCLE_ORDER)
        if total > 0 and not (99 <= total <= 101):
            scale = 100.0 / total
            rescaled = {k: int(round(normalized[k] * scale)) for k in _CYCLE_ORDER}
            diff = 100 - sum(rescaled[k] for k in _CYCLE_ORDER)
            if diff != 0:
                biggest = max(_CYCLE_ORDER, key=lambda k: rescaled[k])
                rescaled[biggest] = max(0, rescaled[biggest] + diff)
            normalized = rescaled
        elif total == 0:
            # All-zero input: fall back to default distribution.
            cycle_guess = str(
                prediction.get("cycle")
                or (stage1_json or {}).get("cycle_position")
                or "trading_range"
            ).strip().lower()
            normalized = _default_cycle_probs(cycle_guess)

        prediction["probabilities"] = normalized

        # cycle = argmax (tie-break by _CYCLE_ORDER literal order).
        max_value = max(normalized[k] for k in _CYCLE_ORDER)
        argmax_cycle = next(k for k in _CYCLE_ORDER if normalized[k] == max_value)
        model_cycle = str(prediction.get("cycle") or "").strip().lower()
        if model_cycle != argmax_cycle:
            prediction["cycle"] = argmax_cycle

    # direction: keep model value; coerce non-string to None.
    direction = prediction.get("direction")
    if direction is not None and not isinstance(direction, str):
        prediction["direction"] = None


# ── Trace answer resolution (ported from trace_normalize.py:326-365) ──

def _resolve_trace_answer(
    node_id: str,
    answer: str,
) -> tuple[str, str | None] | None:
    """Map raw AI answer to (canonical answer, optional branch).

    Returns ``None`` if no mapping applies (answer left unchanged).
    """
    ans = (answer or "").strip()
    if not ans:
        return None

    # gate_result tokens mistakenly written as answers.
    if ans.lower() in _GATE_RESULT_ANSWER_ALIASES:
        return _GATE_RESULT_ANSWER_ALIASES[ans.lower()], None

    per_node = _NODE_ANSWER_BY_ID.get(node_id, {})

    mapped = per_node.get(ans) or per_node.get(ans.lower())
    if mapped:
        return mapped

    # Composite answers like "是（多头）" / "否（上涨）".
    m = _COMPOSITE_ANSWER_RE.match(ans)
    if m:
        base = m.group(1)
        tail = m.group(2).strip()
        branch: str | None = None
        for key in sorted(per_node.keys(), key=len, reverse=True):
            if key in tail or key.lower() in tail.lower():
                branch = per_node[key][1]
                break
        return base, branch

    # Substring match against per-node aliases.
    for key in sorted(per_node.keys(), key=len, reverse=True):
        if key in ans or key.lower() in ans.lower():
            return per_node[key]

    # Generic answer table.
    if ans in _GENERIC_ANSWER:
        return _GENERIC_ANSWER[ans], None
    if ans.lower() in _GENERIC_ANSWER:
        return _GENERIC_ANSWER[ans.lower()], None

    # Qualified partial answers (e.g. "部分符合" / "部分是").
    if ans.startswith("部分"):
        return "中性", None

    return None


def _resolve_trace_answers(trace: list[Any]) -> None:
    """Apply :func:`_resolve_trace_answer` to each trace item in-place."""
    if not isinstance(trace, list):
        return
    for item in trace:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id", ""))
        answer = item.get("answer")
        if not isinstance(answer, str):
            continue
        resolved = _resolve_trace_answer(node_id, answer)
        if resolved is None:
            continue
        canonical, branch = resolved
        item["answer"] = canonical
        if branch and isinstance(item.get("branch"), str):
            # Only override an existing branch slot; never inject one.
            item["branch"] = branch


# ── Public entry points ──

def _max_seq_from_feature_rows(feature_rows: list[dict[str, Any]] | None) -> int | None:
    """Infer the max K index from feature_rows length (K1..K{N})."""
    if not feature_rows:
        return None
    n = len(feature_rows)
    return n if n >= 1 else None


def normalize_market_diagnosis(
    diagnosis: dict[str, Any],
    *,
    feature_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage1 normalize, called before :func:`validate_market_diagnosis`.

    PR1 scope: gate_trace answer alias mapping only.
    PR2 will add bar_by_bar pad + bar_type/role/context_effect repair.
    Batch A: gate_trace bar_range canonicalization (ported from upstream
    ``normalize_stage1_traces``).
    """
    out = copy.deepcopy(diagnosis)
    gate = out.get("gate_trace")
    if isinstance(gate, list):
        strip_ai_gate_14(gate)
    repair_stage1_gate_trace(out)
    if isinstance(gate, list) and repair_stage1_gate_trace_questions(gate):
        logger.debug("gate_trace questions aligned with decision tree spec")
    _resolve_trace_answers(gate or [])
    normalize_trace_list_bar_range(
        gate,
        default_max_seq=_max_seq_from_feature_rows(feature_rows),
    )

    # Program-computed §1.1 (data), §2.3 (direction), and §2.4 (Always-In):
    # override whatever the LLM emitted. Mirrors upstream PA_Agent
    # DecisionNodeEngine apply_stage1. See issue #37: GLM-5.2 systematically mis-fills the
    # 2.3 ``branch`` field (null / position words like "middle"), causing
    # ``direction_branch_conflict`` rejections.
    if feature_rows:
        try:
            from .decision_nodes import apply_stage1_nodes

            if apply_stage1_nodes(out, feature_rows):
                logger.debug("stage1 nodes 1.1/2.3/2.4 computed by DecisionNodeEngine")
                gate = out.get("gate_trace") or []
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning("DecisionNodeEngine.apply_stage1_nodes failed: %s", exc)

    for msg in auto_fix_bar_by_bar_types(out, feature_rows=feature_rows):
        logger.info("stage1 %s", msg)

    if ensure_detected_patterns_coherent(out):
        logger.debug("detected_patterns synced with key_signals/entry_setup_type")
    return out


def normalize_trade_decision(
    decision_json: dict[str, Any],
    *,
    diagnosis: dict[str, Any] | None = None,
    feature_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage2 normalize, called before :class:`DecisionValidator`.

    Fixes ``next_cycle_prediction.probabilities`` (float→int, clamp,
    rescale sum=100, cycle=argmax), normalizes stage2 ``bar_analysis``
    closed enums (bar_type / entry_bar / signal_bar / second_entry),
    maps decision_trace answer aliases (e.g. "不下单" → "否"),
    canonicalizes bar_range strings (Batch A port from upstream
    ``normalize_stage2_traces``), hoists decision fields misplaced under
    diagnosis_summary (Batch E), repairs flat/scalar decision payloads
    (Batch D: unwrap → hoist terminal → ensure required fields →
    truncate reasoning), coerces decision to 不下单 when trace/terminal
    reject the trade (ported from upstream ``_coerce_decision_no_order``),
    and normalizes breakout entry_price to basis extreme +/- 1 tick
    (ported from PA_Agent price_tick).
    """
    out = copy.deepcopy(decision_json)
    if normalize_stage2_bar_analysis_enums(out, stage1_json=diagnosis):
        logger.debug("stage2 bar_analysis enums normalized")
    prediction = out.get("next_cycle_prediction")
    if isinstance(prediction, dict):
        _normalize_next_cycle_prediction(prediction, stage1_json=diagnosis)
    decision_trace = out.get("decision_trace")
    if isinstance(decision_trace, list) and repair_stage2_decision_trace_questions(
        decision_trace
    ):
        logger.debug("decision_trace questions aligned with decision tree spec")
    _resolve_trace_answers(decision_trace or [])
    normalize_trace_list_bar_range(
        out.get("decision_trace"),
        default_max_seq=_max_seq_from_feature_rows(feature_rows),
    )

    if repair_diagnosis_summary_and_decision(out, stage1_json=diagnosis):
        logger.debug("diagnosis_summary decision fields hoisted / defaults filled")
    if _unwrap_flat_stage2_decision(out):
        logger.debug("flat stage2 decision hoisted into decision object")
    if _hoist_terminal_from_decision(out):
        logger.debug("terminal hoisted from decision to top level")
    if _ensure_decision_required_fields(out, stage1_json=diagnosis):
        logger.debug("decision required fields filled with defaults")
    decision_obj = out.get("decision")
    if isinstance(decision_obj, dict) and _truncate_decision_reasoning(decision_obj):
        logger.debug("decision.reasoning truncated to %d chars", DECISION_REASONING_MAX_LEN)

    # Normalize enum aliases (order_direction / always_in / terminal.outcome)
    # before coercion and validation. Mirrors SOURCE step 9.
    if _normalize_stage2_enum_aliases(out):
        logger.debug("stage2 enum aliases normalized")

    if _coerce_decision_no_order(out):
        logger.debug("decision coerced to 不下单 (trace/terminal rejection)")

    # Sync stage2 diagnosis_summary.direction with program-overwritten
    # stage1 direction. The DecisionNodeEngine (§2.3) rewrites
    # diagnosis.direction during stage1 normalize; AI's stage2 reply still
    # carries the older value it saw in the stage2 user prompt. Without
    # this sync the validator trips ``diagnosis_summary_direction_mismatch``.
    if isinstance(diagnosis, dict) and isinstance(out.get("diagnosis_summary"), dict):
        stage1_dir = diagnosis.get("direction")
        if stage1_dir and out["diagnosis_summary"].get("direction") != stage1_dir:
            out["diagnosis_summary"]["direction"] = stage1_dir

    if repair_stage2_terminal(out):
        logger.debug("terminal.node_id aligned to 10.3 (no-order rejection)")

    decision = out.get("decision")
    if isinstance(decision, dict) and normalize_breakout_basis_extreme(decision):
        logger.debug(
            "breakout entry_basis_extreme aligned to %s for %s",
            decision.get("entry_basis_extreme"),
            decision.get("order_direction"),
        )
    if isinstance(
        decision, dict
    ) and normalize_breakout_entry_price(decision, feature_rows=feature_rows):
        logger.debug(
            "breakout entry_price adjusted to basis extreme +/- 1 tick (basis=%s)",
            decision.get("entry_basis_bar"),
        )

    # ── TP1 RR cap (widen stop) + trade-metrics veto ────────────────────────
    # Mirrors PA_Agent stage2_normalizer.py:1561-1570. First widen the stop so
    # TP1 reward:risk falls within the 1.5 program cap (entry/TP unchanged),
    # then reject the whole order to 不下单 if it still fails RR / trader
    # equation / K1 freshness / TP2 geometry. This is the bottom-line safety
    # net (#1 widen_stop + #3 metrics 失败强制不下单) — without it a malformed
    # stop (e.g. 0.94-pt stop → TP1 R/R=16.81) passes through after retries.
    if isinstance(decision, dict):
        from .trade_metrics import adjust_decision_stop_for_tp1_rr_cap

        if adjust_decision_stop_for_tp1_rr_cap(decision, feature_rows=feature_rows):
            logger.debug("stop_loss widened to bring TP1 RR within program cap")
    if _coerce_decision_when_trade_metrics_fail(out, feature_rows=feature_rows):
        logger.debug("decision coerced to 不下单 (trade metrics failed)")

    # ── Planned-limit trace fixes (ported from stage2_normalizer:1571-1573) ──
    # Order per SOURCE: ensure §9.0P=是 background node first, then upgrade
    # §9.0 answer. Both gate on is_planned_limit_order (requires order_type
    # still being 限价单), so they must run before any coercion flips it.
    if _fix_background_limit_trace(out):
        logger.debug("§9.0P=是 background-limit trace node ensured")
    if _fix_9_0_for_planned_limit(out):
        logger.debug("§9.0 upgraded to 是 for planned-limit order")

    # ── Stage2 decision-node engine (ported from stage2_normalizer:1576-1582) ──
    # Program-compute §9.1/§9.2/§9.3/§9.5 signal-bar judges + §11 order-method
    # routing, then apply node_overrides and merge into decision_trace. Runs
    # after the planned-limit trace fixes so the §9.0/§9.0P background path is
    # already reconciled before the §9 judges decide whether to skip.
    if feature_rows is not None:
        try:
            from .decision_nodes import DecisionNodeEngine

            DecisionNodeEngine.apply_stage2(out, feature_rows, diagnosis)
            logger.debug("stage2 §9/§11 nodes computed by DecisionNodeEngine")
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning("DecisionNodeEngine.apply_stage2 failed: %s", exc)

    # ── Post-engine cleanup (mirror SOURCE normalize_stage2:1589-1650) ────────
    # No-order re-null: engine/trace may have changed order_type; re-assert the
    # schema "then" branch (all price fields + direction null for 不下单).
    decision = out.get("decision")
    if isinstance(decision, dict) and decision.get("order_type") == "不下单":
        for field in _NO_ORDER_PRICE_FIELDS:
            decision[field] = None
        decision["estimated_win_rate"] = None
        if decision.get("trade_confidence") is None:
            decision["trade_confidence"] = 0
        if not isinstance(decision.get("trade_confidence_reasoning"), str) or not decision["trade_confidence_reasoning"]:
            decision["trade_confidence_reasoning"] = "无入场计划，不存在交易信心"

    # Terminal trade-node repair (trade cannot terminate at §14).
    if _repair_terminal_trade_node(out):
        logger.debug("terminal.node_id repaired (was §14)")

    # Entry-bar / signal-bar chain normalization.
    bar_analysis = out.get("bar_analysis")
    if isinstance(bar_analysis, dict) and isinstance(decision, dict):
        if _normalize_market_order_entry_bar(bar_analysis, decision):
            logger.debug("market order entry_bar borrowed from signal_bar")
        _normalize_signal_entry_bar_chain(bar_analysis, decision)

    # Pending entry_bar state normalization.
    if isinstance(bar_analysis, dict):
        signal_bar = bar_analysis.get("signal_bar")
        if isinstance(signal_bar, dict):
            if not signal_bar.get("bar"):
                signal_bar["bar"] = None
                signal_bar.setdefault("quality", "invalid")
                signal_bar.setdefault("pattern", "none")

        entry_bar = bar_analysis.get("entry_bar")
        if isinstance(entry_bar, dict):
            strength = str(entry_bar.get("strength", "") or "").strip().lower()
            # Only treat as pending when the model explicitly said not_triggered
            # or the entry bar is None with a pending-ish freshness. Aged/stale
            # freshness is left to _normalize_entry_bar_freshness (run earlier).
            if strength == "not_triggered":
                entry_bar.setdefault("bar", None)
                fresh = str(entry_bar.get("freshness") or "").strip().lower()
                if fresh not in ("stale", "aged", "expired"):
                    entry_bar["freshness"] = "pending"
                if entry_bar.get("follow_through") in (None, "", "pending"):
                    entry_bar["follow_through"] = "pending"

    # diagnosis_summary injection-if-missing.
    if not isinstance(out.get("diagnosis_summary"), dict):
        s1 = diagnosis or {}
        out["diagnosis_summary"] = {
            "cycle_position": s1.get("cycle_position", "unknown"),
            "direction": s1.get("direction", "neutral"),
            "key_signals": [],
        }
        logger.debug("Injected missing diagnosis_summary from stage1")

    # Prediction stub injection.
    if ensure_stage2_predictions(out, stage1_json=diagnosis):
        logger.debug("stage2 prediction stubs injected")

    return out
