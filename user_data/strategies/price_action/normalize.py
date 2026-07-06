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
from .trace_normalize import (
    normalize_trace_list_bar_range,
    repair_stage2_terminal,
    strip_ai_gate_14,
)

logger = logging.getLogger(__name__)

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
    _resolve_trace_answers(gate or [])
    normalize_trace_list_bar_range(
        gate,
        default_max_seq=_max_seq_from_feature_rows(feature_rows),
    )
    return out


def normalize_trade_decision(
    decision_json: dict[str, Any],
    *,
    diagnosis: dict[str, Any] | None = None,
    feature_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage2 normalize, called before :class:`DecisionValidator`.

    Fixes ``next_cycle_prediction.probabilities`` (float→int, clamp,
    rescale sum=100, cycle=argmax), maps decision_trace answer aliases
    (e.g. "不下单" → "否"), canonicalizes bar_range strings (Batch A port
    from upstream ``normalize_stage2_traces``), coerces decision to 不下单
    when trace/terminal reject the trade (ported from upstream
    ``_coerce_decision_no_order``), and normalizes breakout entry_price
    to basis extreme +/- 1 tick (ported from PA_Agent price_tick).
    """
    out = copy.deepcopy(decision_json)
    prediction = out.get("next_cycle_prediction")
    if isinstance(prediction, dict):
        _normalize_next_cycle_prediction(prediction, stage1_json=diagnosis)
    _resolve_trace_answers(out.get("decision_trace") or [])
    normalize_trace_list_bar_range(
        out.get("decision_trace"),
        default_max_seq=_max_seq_from_feature_rows(feature_rows),
    )

    if _coerce_decision_no_order(out):
        logger.debug("decision coerced to 不下单 (trace/terminal rejection)")

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
    return out
