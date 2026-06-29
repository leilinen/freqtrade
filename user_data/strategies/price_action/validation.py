"""Four-stage L4 decision validation."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any


CHECK_JSON_SYNTAX = "json_syntax"
CHECK_STAGE_CONSISTENCY = "stage_consistency"
CHECK_SEMANTIC_REASONABLENESS = "semantic_reasonableness"
CHECK_NUMERIC_RANGE = "numeric_range"

STAGE1_REQUIRED_FIELDS = (
    "cycle_position",
    "direction",
    "diagnosis_confidence",
    "market_phase",
    "detected_patterns",
    "key_signals",
    "htf_context",
    "entry_setup",
    "strategy_files_needed",
    "bar_by_bar_summary",
    "gate_trace",
    "gate_result",
)
STAGE1_CYCLE_POSITIONS = {
    "spike",
    "micro_channel",
    "tight_channel",
    "normal_channel",
    "broad_channel",
    "trending_tr",
    "trading_range",
    "extreme_tr",
    "unknown",
}
STAGE1_DIRECTIONS = {"bullish", "bearish", "neutral"}
STAGE1_MARKET_PHASES = {"stable", "transitioning"}
STAGE1_GATE_RESULTS = {"proceed", "wait", "unknown"}
STAGE1_PROCEED_TRACE_NODES = {"1.2", "1.3", "2.1", "2.2", "2.5"}


@dataclass
class ValidationResult:
    """Result for ordered L4 validation."""

    valid: bool
    checks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "valid" if self.valid else "invalid"

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "checks": self.checks,
            "errors": self.errors,
        }


class DecisionValidator:
    """Validate L4 output in the required PA_Agent order."""

    def validate(
        self,
        raw_response: str,
        *,
        diagnosis: dict[str, Any],
        l1_features: dict[str, Any],
        strategies: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, ValidationResult]:
        checks: list[str] = []

        checks.append(CHECK_JSON_SYNTAX)
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            return None, ValidationResult(False, checks, [f"invalid_json:{exc.msg}"])
        if not isinstance(parsed, dict):
            return None, ValidationResult(False, checks, ["json_root_must_be_object"])

        checks.append(CHECK_STAGE_CONSISTENCY)
        errors = self._stage_consistency_errors(parsed, diagnosis)
        if errors:
            return parsed, ValidationResult(False, checks, errors)

        checks.append(CHECK_SEMANTIC_REASONABLENESS)
        errors = self._semantic_errors(parsed, diagnosis, l1_features, strategies or [])
        if errors:
            return parsed, ValidationResult(False, checks, errors)

        checks.append(CHECK_NUMERIC_RANGE)
        errors = self._numeric_errors(parsed, l1_features)
        if errors:
            return parsed, ValidationResult(False, checks, errors)

        return parsed, ValidationResult(True, checks, [])

    def _stage_consistency_errors(
        self,
        decision_json: dict[str, Any],
        diagnosis: dict[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        if decision_json.get("stage") != "trade_decision":
            errors.append("stage_must_be_trade_decision")
        if not _looks_like_stage1_diagnosis(diagnosis):
            errors.append("diagnosis_must_be_stage1")

        decision = decision_json.get("decision")
        if not isinstance(decision, dict):
            errors.append("decision_must_be_object")
            return errors

        decision_type = str(decision.get("type", "")).lower()
        direction = str(decision.get("direction", "")).lower()
        valid_types = {"enter_long", "enter_short", "wait", "avoid"}
        if decision_type not in valid_types:
            errors.append("decision_type_invalid")
        if decision_type == "enter_long" and direction != "long":
            errors.append("enter_long_requires_long_direction")
        if decision_type == "enter_short" and direction != "short":
            errors.append("enter_short_requires_short_direction")
        return errors

    def _semantic_errors(
        self,
        decision_json: dict[str, Any],
        diagnosis: dict[str, Any],
        l1_features: dict[str, Any],
        strategies: list[dict[str, Any]],
    ) -> list[str]:
        del diagnosis, strategies
        decision = decision_json["decision"]
        decision_type = str(decision.get("type", "")).lower()
        entry = _number(decision.get("entry"))
        stop = _number(decision.get("stop_loss"))
        tp1 = _number(decision.get("take_profit_1"))
        tp2 = _number(decision.get("take_profit_2"))

        if decision_type in ("wait", "avoid"):
            if entry is not None and stop is not None and (tp1 is not None or tp2 is not None):
                return ["wait_or_avoid_should_not_include_full_order_plan"]
            return []

        if entry is None or stop is None:
            return ["actionable_decision_requires_entry_and_stop"]

        errors: list[str] = []
        if decision_type == "enter_long":
            if not stop < entry:
                errors.append("long_stop_must_be_below_entry")
            if tp1 is not None and not tp1 > entry:
                errors.append("long_tp1_must_be_above_entry")
            if tp2 is not None and not tp2 > entry:
                errors.append("long_tp2_must_be_above_entry")
        elif decision_type == "enter_short":
            if not stop > entry:
                errors.append("short_stop_must_be_above_entry")
            if tp1 is not None and not tp1 < entry:
                errors.append("short_tp1_must_be_below_entry")
            if tp2 is not None and not tp2 < entry:
                errors.append("short_tp2_must_be_below_entry")

        atr_expand = _number(l1_features.get("atr_expand_ratio"))
        gate_break = str(l1_features.get("gate_break", "none")).lower()
        if atr_expand is not None and atr_expand > 2.0 and gate_break != "none":
            errors.append("atr_expansion_over_2x_vetoes_breakout_entry")
        return errors

    def _numeric_errors(
        self,
        decision_json: dict[str, Any],
        l1_features: dict[str, Any],
    ) -> list[str]:
        decision = decision_json["decision"]
        decision_type = str(decision.get("type", "")).lower()
        confidence = _number(decision.get("confidence"))
        rr = _number(decision.get("risk_reward"))
        errors: list[str] = []

        if confidence is None or not 0.0 <= confidence <= 1.0:
            errors.append("confidence_must_be_0_to_1")
        if rr is not None and not 0.0 <= rr <= 20.0:
            errors.append("risk_reward_out_of_range")

        if decision_type not in ("enter_long", "enter_short"):
            return errors

        prices = {
            "entry": _number(decision.get("entry")),
            "stop_loss": _number(decision.get("stop_loss")),
            "take_profit_1": _number(decision.get("take_profit_1")),
            "take_profit_2": _number(decision.get("take_profit_2")),
        }
        for name, value in prices.items():
            if value is not None and value <= 0:
                errors.append(f"{name}_must_be_positive")

        entry = prices["entry"]
        latest_high = _number(l1_features.get("high"))
        latest_low = _number(l1_features.get("low"))
        atr = _number(l1_features.get("atr14")) or 0.0
        if entry is not None and latest_high is not None and latest_low is not None:
            tolerance = max(atr * 5.0, abs(latest_high - latest_low) * 3.0)
            lower_bound = max(0.0, latest_low - tolerance)
            upper_bound = latest_high + tolerance
            if not lower_bound <= entry <= upper_bound:
                errors.append("entry_too_far_from_latest_candle")
        return errors


def parse_json_object(raw_response: str) -> dict[str, Any]:
    """Parse a required JSON object response."""
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response root must be a JSON object")
    return parsed


def validate_stage1_diagnosis(
    diagnosis: dict[str, Any],
    *,
    l1_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Validate the PA_Agent Stage 1 diagnosis contract used before routing."""
    errors: list[str] = []
    if not isinstance(diagnosis, dict):
        return ["stage1_root_must_be_object"]

    for field in STAGE1_REQUIRED_FIELDS:
        if field not in diagnosis:
            errors.append(f"stage1_missing_{field}")

    cycle = str(diagnosis.get("cycle_position", "")).lower()
    if cycle not in STAGE1_CYCLE_POSITIONS:
        errors.append("stage1_cycle_position_invalid")

    direction = str(diagnosis.get("direction", "")).lower()
    if direction not in STAGE1_DIRECTIONS:
        errors.append("stage1_direction_invalid")

    confidence = diagnosis.get("diagnosis_confidence")
    confidence_valid = (
        not isinstance(confidence, bool)
        and isinstance(confidence, int)
        and 0 <= confidence <= 100
    )
    if not confidence_valid:
        errors.append("stage1_diagnosis_confidence_must_be_0_to_100_int")

    market_phase = str(diagnosis.get("market_phase", "")).lower()
    if market_phase not in STAGE1_MARKET_PHASES:
        errors.append("stage1_market_phase_invalid")

    for field in ("detected_patterns", "key_signals", "strategy_files_needed"):
        if field in diagnosis and not isinstance(diagnosis.get(field), list):
            errors.append(f"stage1_{field}_must_be_array")

    rows_by_k = {str(row.get("k")): row for row in (l1_rows or [])}
    latest = rows_by_k.get("K1")
    bar_analysis = diagnosis.get("bar_analysis")
    if isinstance(bar_analysis, dict) and latest:
        if bar_analysis.get("bar_type") != latest.get("bar_type"):
            errors.append("stage1_bar_analysis_bar_type_mismatch")

    summary = diagnosis.get("bar_by_bar_summary")
    if not isinstance(summary, list) or not summary:
        errors.append("stage1_bar_by_bar_summary_required")
    else:
        expected_count = min(5, len(l1_rows or summary))
        if len(l1_rows or []) >= 5 and len(summary) != 5:
            errors.append("stage1_bar_by_bar_summary_must_cover_k5_to_k1")
        expected_bars = {f"K{i}" for i in range(1, expected_count + 1)}
        seen_bars = {str(item.get("bar")) for item in summary if isinstance(item, dict)}
        if expected_bars and seen_bars and seen_bars != expected_bars:
            errors.append("stage1_bar_by_bar_summary_bars_invalid")
        for item in summary:
            if not isinstance(item, dict):
                errors.append("stage1_bar_by_bar_summary_item_must_be_object")
                continue
            bar = str(item.get("bar", ""))
            row = rows_by_k.get(bar)
            if row and item.get("bar_type") != row.get("bar_type"):
                errors.append(f"stage1_bar_by_bar_{bar}_bar_type_mismatch")

    gate_trace = diagnosis.get("gate_trace")
    gate_result = str(diagnosis.get("gate_result", "")).lower()
    if gate_result not in STAGE1_GATE_RESULTS:
        errors.append("stage1_gate_result_invalid")
    if not isinstance(gate_trace, list) or not gate_trace:
        errors.append("stage1_gate_trace_required")
    else:
        node_ids = {
            str(item.get("node_id"))
            for item in gate_trace
            if isinstance(item, dict) and item.get("node_id") is not None
        }
        if gate_result == "proceed" and not STAGE1_PROCEED_TRACE_NODES <= node_ids:
            errors.append("stage1_gate_trace_missing_proceed_nodes")
        if gate_result in ("wait", "unknown"):
            last = gate_trace[-1] if isinstance(gate_trace[-1], dict) else {}
            if last.get("answer") not in ("否", "等待"):
                errors.append("stage1_gate_wait_requires_negative_or_waiting_final_answer")
        for item in gate_trace:
            if not isinstance(item, dict):
                errors.append("stage1_gate_trace_item_must_be_object")
                continue
            if not item.get("bar_range"):
                errors.append("stage1_gate_trace_bar_range_required")
            elif "K0" in str(item.get("bar_range")):
                errors.append("stage1_gate_trace_bar_range_must_not_reference_k0")

    return errors


def _looks_like_stage1_diagnosis(diagnosis: dict[str, Any]) -> bool:
    return isinstance(diagnosis, dict) and (
        "cycle_position" in diagnosis
        and "direction" in diagnosis
        and "gate_result" in diagnosis
    )


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number
