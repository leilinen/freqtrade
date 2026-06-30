"""Trade-decision validation."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any


CHECK_JSON_SYNTAX = "json_syntax"
CHECK_STAGE_CONSISTENCY = "stage_consistency"
CHECK_SEMANTIC_REASONABLENESS = "semantic_reasonableness"
CHECK_NUMERIC_RANGE = "numeric_range"

MARKET_DIAGNOSIS_REQUIRED_FIELDS = (
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
MARKET_DIAGNOSIS_CYCLE_POSITIONS = {
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
MARKET_DIAGNOSIS_DIRECTIONS = {"bullish", "bearish", "neutral"}
MARKET_DIAGNOSIS_MARKET_PHASES = {"stable", "transitioning"}
MARKET_DIAGNOSIS_GATE_RESULTS = {"proceed", "wait", "unknown"}
MARKET_DIAGNOSIS_TRACE_ANSWERS = {"是", "否", "中性", "等待", "不适用"}
MARKET_DIAGNOSIS_FORBIDDEN_GATE_NODES = {"0.3"}
MARKET_DIAGNOSIS_PROCEED_TRACE_NODES = {
    "1.1",
    "1.2",
    "1.3",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
}
MARKET_DIAGNOSIS_SPIKE_STAGES = {"active", "ending", "transitioning"}
MARKET_DIAGNOSIS_CLIMAX_RISKS = {"none", "warning", "triggered"}
MARKET_DIAGNOSIS_TRANSITION_RISKS = {"high", "medium", "low"}
MARKET_DIAGNOSIS_BAR_TYPES = {
    "trend_bull",
    "trend_bear",
    "doji",
    "inside",
    "outside_bull",
    "outside_bear",
    "flat",
    "other",
}
MARKET_DIAGNOSIS_ALWAYS_IN = {"long", "short", "neutral"}
MARKET_DIAGNOSIS_SIGNAL_QUALITIES = {"strong", "medium", "weak", "invalid"}
MARKET_DIAGNOSIS_BAR_ROLES = {
    "structure",
    "signal",
    "entry",
    "confirmation",
    "noise",
    "trap",
    "climax",
    "test",
}
MARKET_DIAGNOSIS_CONTEXT_EFFECTS = {
    "strengthens_bull",
    "weakens_bull",
    "strengthens_bear",
    "weakens_bear",
    "neutral",
    "transition",
    "weakened_bull",
    "weakened_bear",
}
MARKET_DIAGNOSIS_FOLLOW_THROUGH = {"yes", "no", "pending", "failed"}
MARKET_DIAGNOSIS_TRAPPED_SIDES = {"bulls", "bears", "both", "none", "unknown"}
MARKET_DIAGNOSIS_RANGE_CYCLES = {
    "trading_range",
    "extreme_tr",
    "trending_tr",
    "broad_channel",
}

_K_RANGE_RE = re.compile(r"^K(\d+)-K(\d+)$", re.IGNORECASE)
_K_SINGLE_RE = re.compile(r"^K(\d+)$", re.IGNORECASE)
_K_ANY_RE = re.compile(r"K\s*(\d+)", re.IGNORECASE)
_CYCLE_BRANCH_ALIASES = {
    "tr": "trading_range",
    "交易区间": "trading_range",
    "普通交易区间": "trading_range",
    "趋势型交易区间": "trending_tr",
    "极端交易区间": "extreme_tr",
    "尖峰": "spike",
    "微型通道": "micro_channel",
    "窄通道": "tight_channel",
    "正常通道": "normal_channel",
    "宽通道": "broad_channel",
}
_DIRECTION_BRANCH_ALIASES = {
    "bull": "bullish",
    "多头": "bullish",
    "上涨": "bullish",
    "bear": "bearish",
    "空头": "bearish",
    "下跌": "bearish",
    "中性": "neutral",
    "震荡": "neutral",
}


@dataclass
class ValidationResult:
    """Result for ordered trade-decision validation."""

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
    """Validate trade-decision output in the required PA_Agent order."""

    def validate(
        self,
        raw_response: str,
        *,
        diagnosis: dict[str, Any],
        price_action_features: dict[str, Any],
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
        errors = self._semantic_errors(parsed, diagnosis, price_action_features, strategies or [])
        if errors:
            return parsed, ValidationResult(False, checks, errors)

        checks.append(CHECK_NUMERIC_RANGE)
        errors = self._numeric_errors(parsed, price_action_features)
        if errors:
            return parsed, ValidationResult(False, checks, errors)

        return parsed, ValidationResult(True, checks, [])

    def _stage_consistency_errors(
        self,
        decision_json: dict[str, Any],
        diagnosis: dict[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        if not _looks_like_market_diagnosis(diagnosis):
            errors.append("diagnosis_must_be_market_diagnosis")

        decision = decision_json.get("decision")
        if not isinstance(decision, dict):
            errors.append("decision_must_be_object")
            return errors

        order_type = decision.get("order_type")
        if order_type not in ("限价单", "突破单", "市价单", "不下单"):
            errors.append("order_type_invalid")

        order_direction = decision.get("order_direction")
        if order_type in ("限价单", "突破单", "市价单"):
            if order_direction not in ("做多", "做空"):
                errors.append("actionable_decision_requires_order_direction")
        elif order_direction is not None:
            errors.append("no_trade_order_direction_must_be_null")

        summary = decision_json.get("diagnosis_summary")
        if not isinstance(summary, dict):
            errors.append("diagnosis_summary_must_be_object")
        else:
            if summary.get("cycle_position") != diagnosis.get("cycle_position"):
                errors.append("diagnosis_summary_cycle_position_mismatch")
            if summary.get("direction") != diagnosis.get("direction"):
                errors.append("diagnosis_summary_direction_mismatch")
            if not isinstance(summary.get("key_signals"), list):
                errors.append("diagnosis_summary_key_signals_must_be_array")

        if not isinstance(decision_json.get("decision_trace"), list):
            errors.append("decision_trace_must_be_array")
        terminal = decision_json.get("terminal")
        if not isinstance(terminal, dict):
            errors.append("terminal_must_be_object")
        else:
            if terminal.get("outcome") not in ("wait", "reject", "trade", "proceed"):
                errors.append("terminal_outcome_invalid")
        if not isinstance(decision_json.get("next_cycle_prediction"), dict):
            errors.append("next_cycle_prediction_must_be_object")
        return errors

    def _semantic_errors(
        self,
        decision_json: dict[str, Any],
        diagnosis: dict[str, Any],
        price_action_features: dict[str, Any],
        strategies: list[dict[str, Any]],
    ) -> list[str]:
        del diagnosis, strategies
        decision = decision_json["decision"]
        order_type = str(decision.get("order_type", ""))
        order_direction = decision.get("order_direction")
        entry = _number(decision.get("entry_price"))
        stop = _number(decision.get("stop_loss_price"))
        tp1 = _number(decision.get("take_profit_price"))
        tp2 = _number(decision.get("take_profit_price_2"))

        if order_type == "不下单":
            errors = []
            for field in (
                "entry_price",
                "entry_basis_bar",
                "entry_basis_extreme",
                "entry_rule",
                "take_profit_price",
                "take_profit_price_2",
                "stop_loss_price",
                "order_direction",
                "estimated_win_rate",
            ):
                if decision.get(field) is not None:
                    errors.append("no_trade_fields_must_be_null")
                    break
            terminal = decision_json.get("terminal") or {}
            if terminal.get("outcome") == "trade":
                errors.append("no_trade_terminal_must_not_be_trade")
            return errors

        if order_type == "突破单":
            if decision.get("entry_basis_bar") is None:
                return ["breakout_order_requires_entry_basis_bar"]
            if decision.get("entry_basis_extreme") not in ("high", "low"):
                return ["breakout_order_requires_entry_basis_extreme"]
            if not decision.get("entry_rule"):
                return ["breakout_order_requires_entry_rule"]

        if order_direction not in ("做多", "做空"):
            return ["actionable_decision_requires_order_direction"]
        if entry is None or stop is None or tp1 is None or tp2 is None:
            return ["actionable_decision_requires_full_price_plan"]

        terminal = decision_json.get("terminal") or {}
        if terminal.get("outcome") != "trade":
            return ["actionable_decision_requires_trade_terminal"]

        estimated = decision.get("estimated_win_rate")
        if estimated is None:
            return ["actionable_decision_requires_estimated_win_rate"]

        errors: list[str] = []
        if order_direction == "做多":
            if not stop < entry:
                errors.append("long_stop_must_be_below_entry")
            if not entry < tp1:
                errors.append("long_tp1_must_be_above_entry")
            if not tp1 < tp2:
                errors.append("long_tp2_must_be_above_tp1")
        elif order_direction == "做空":
            if not stop > entry:
                errors.append("short_stop_must_be_above_entry")
            if not entry > tp1:
                errors.append("short_tp1_must_be_below_entry")
            if not tp1 > tp2:
                errors.append("short_tp2_must_be_below_tp1")

        atr_expand = _number(price_action_features.get("atr_expand_ratio"))
        gate_break = str(price_action_features.get("gate_break", "none")).lower()
        if atr_expand is not None and atr_expand > 2.0 and gate_break != "none":
            errors.append("atr_expansion_over_2x_vetoes_breakout_entry")
        return errors

    def _numeric_errors(
        self,
        decision_json: dict[str, Any],
        price_action_features: dict[str, Any],
    ) -> list[str]:
        decision = decision_json["decision"]
        order_type = str(decision.get("order_type", ""))
        trade_confidence = decision.get("trade_confidence")
        diagnosis_confidence = decision.get("diagnosis_confidence")
        estimated_win_rate = decision.get("estimated_win_rate")
        errors: list[str] = []

        for field, value in (
            ("trade_confidence", trade_confidence),
            ("diagnosis_confidence", diagnosis_confidence),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
                errors.append(f"{field}_must_be_0_to_100_int")
        if estimated_win_rate is not None:
            if (
                isinstance(estimated_win_rate, bool)
                or not isinstance(estimated_win_rate, int)
                or not 0 <= estimated_win_rate <= 100
            ):
                errors.append("estimated_win_rate_must_be_0_to_100_int")

        prediction = decision_json.get("next_cycle_prediction")
        if isinstance(prediction, dict):
            if not isinstance(prediction.get("unpredictable"), bool):
                errors.append("next_cycle_prediction_unpredictable_must_be_bool")

        if order_type == "不下单":
            return errors

        prices = {
            "entry_price": _number(decision.get("entry_price")),
            "stop_loss_price": _number(decision.get("stop_loss_price")),
            "take_profit_price": _number(decision.get("take_profit_price")),
            "take_profit_price_2": _number(decision.get("take_profit_price_2")),
        }
        for name, value in prices.items():
            if value is not None and value <= 0:
                errors.append(f"{name}_must_be_positive")

        entry = prices["entry_price"]
        latest_high = _number(price_action_features.get("high"))
        latest_low = _number(price_action_features.get("low"))
        atr = _number(price_action_features.get("atr14")) or 0.0
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


def validate_market_diagnosis(
    diagnosis: dict[str, Any],
    *,
    feature_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Validate the PA_Agent market-diagnosis contract used before routing."""
    errors: list[str] = []
    if not isinstance(diagnosis, dict):
        return ["market_diagnosis_root_must_be_object"]

    for field in MARKET_DIAGNOSIS_REQUIRED_FIELDS:
        if field not in diagnosis:
            errors.append(f"market_diagnosis_missing_{field}")

    cycle = str(diagnosis.get("cycle_position", "")).lower()
    if cycle not in MARKET_DIAGNOSIS_CYCLE_POSITIONS:
        errors.append("market_diagnosis_cycle_position_invalid")

    direction = str(diagnosis.get("direction", "")).lower()
    if direction not in MARKET_DIAGNOSIS_DIRECTIONS:
        errors.append("market_diagnosis_direction_invalid")

    confidence = diagnosis.get("diagnosis_confidence")
    confidence_valid = (
        not isinstance(confidence, bool)
        and isinstance(confidence, int)
        and 0 <= confidence <= 100
    )
    if not confidence_valid:
        errors.append("market_diagnosis_confidence_must_be_0_to_100_int")

    market_phase = str(diagnosis.get("market_phase", "")).lower()
    if market_phase not in MARKET_DIAGNOSIS_MARKET_PHASES:
        errors.append("market_diagnosis_market_phase_invalid")

    spike_stage = diagnosis.get("spike_stage")
    if spike_stage is not None and spike_stage not in MARKET_DIAGNOSIS_SPIKE_STAGES:
        errors.append("market_diagnosis_spike_stage_invalid")
    if cycle == "spike" and spike_stage not in MARKET_DIAGNOSIS_SPIKE_STAGES:
        errors.append("market_diagnosis_spike_requires_spike_stage")

    climax_risk = diagnosis.get("climax_risk")
    if climax_risk is not None and climax_risk not in MARKET_DIAGNOSIS_CLIMAX_RISKS:
        errors.append("market_diagnosis_climax_risk_invalid")

    transition_risk = diagnosis.get("transition_risk")
    if transition_risk is not None and transition_risk not in MARKET_DIAGNOSIS_TRANSITION_RISKS:
        errors.append("market_diagnosis_transition_risk_invalid")
    if market_phase == "transitioning" and transition_risk not in MARKET_DIAGNOSIS_TRANSITION_RISKS:
        errors.append("market_diagnosis_transitioning_requires_transition_risk")

    for field in (
        "detected_patterns",
        "key_signals",
        "strategy_files_needed",
        "support_levels",
        "resistance_levels",
    ):
        value = diagnosis.get(field)
        if field in diagnosis and not isinstance(value, list):
            errors.append(f"market_diagnosis_{field}_must_be_array")
        elif isinstance(value, list) and not all(isinstance(item, str) for item in value):
            errors.append(f"market_diagnosis_{field}_items_must_be_strings")

    rows_by_k = {str(row.get("k")): row for row in (feature_rows or [])}
    max_k_seq = _max_feature_k_seq(feature_rows)
    latest = rows_by_k.get("K1")
    bar_analysis = diagnosis.get("bar_analysis")
    if not isinstance(bar_analysis, dict):
        if "bar_analysis" in diagnosis:
            errors.append("market_diagnosis_bar_analysis_must_be_object")
    else:
        always_in = bar_analysis.get("always_in")
        if always_in is not None and always_in not in MARKET_DIAGNOSIS_ALWAYS_IN:
            errors.append("market_diagnosis_bar_analysis_always_in_invalid")
        bar_type = bar_analysis.get("bar_type")
        if bar_type is not None and bar_type not in MARKET_DIAGNOSIS_BAR_TYPES:
            errors.append("market_diagnosis_bar_analysis_bar_type_invalid")
        if latest and bar_type != latest.get("bar_type"):
            errors.append("market_diagnosis_bar_analysis_bar_type_mismatch")
        signal_bar = bar_analysis.get("signal_bar")
        if isinstance(signal_bar, dict):
            quality = signal_bar.get("quality")
            if quality is not None and quality not in MARKET_DIAGNOSIS_SIGNAL_QUALITIES:
                errors.append("market_diagnosis_signal_bar_quality_invalid")

    summary = diagnosis.get("bar_by_bar_summary")
    if not isinstance(summary, list) or not summary:
        errors.append("market_diagnosis_bar_by_bar_summary_required")
    else:
        expected_count = min(5, len(feature_rows or summary))
        if len(feature_rows or []) >= 5 and len(summary) != 5:
            errors.append("market_diagnosis_bar_by_bar_summary_must_cover_k5_to_k1")
        expected_bars = {f"K{i}" for i in range(1, expected_count + 1)}
        seen_bars = {str(item.get("bar")) for item in summary if isinstance(item, dict)}
        if expected_bars and seen_bars and seen_bars != expected_bars:
            errors.append("market_diagnosis_bar_by_bar_summary_bars_invalid")
        for item in summary:
            if not isinstance(item, dict):
                errors.append("market_diagnosis_bar_by_bar_summary_item_must_be_object")
                continue
            bar = str(item.get("bar", ""))
            for field in (
                "bar",
                "role",
                "bar_type",
                "context_effect",
                "follow_through",
                "trapped_side",
                "reason",
            ):
                if field not in item:
                    errors.append(f"market_diagnosis_bar_by_bar_missing_{field}")
            if not _bar_label_exists(bar, max_k_seq):
                errors.append("market_diagnosis_bar_by_bar_bar_reference_invalid")
            if item.get("role") not in MARKET_DIAGNOSIS_BAR_ROLES:
                errors.append("market_diagnosis_bar_by_bar_role_invalid")
            if item.get("bar_type") not in MARKET_DIAGNOSIS_BAR_TYPES:
                errors.append("market_diagnosis_bar_by_bar_bar_type_invalid")
            if item.get("context_effect") not in MARKET_DIAGNOSIS_CONTEXT_EFFECTS:
                errors.append("market_diagnosis_bar_by_bar_context_effect_invalid")
            if item.get("follow_through") not in MARKET_DIAGNOSIS_FOLLOW_THROUGH:
                errors.append("market_diagnosis_bar_by_bar_follow_through_invalid")
            if item.get("trapped_side") not in MARKET_DIAGNOSIS_TRAPPED_SIDES:
                errors.append("market_diagnosis_bar_by_bar_trapped_side_invalid")
            row = rows_by_k.get(bar)
            if row and item.get("bar_type") != row.get("bar_type"):
                errors.append(f"market_diagnosis_bar_by_bar_{bar}_bar_type_mismatch")

    gate_trace = diagnosis.get("gate_trace")
    gate_result = str(diagnosis.get("gate_result", "")).lower()
    if gate_result not in MARKET_DIAGNOSIS_GATE_RESULTS:
        errors.append("market_diagnosis_gate_result_invalid")
    if not isinstance(gate_trace, list) or not gate_trace:
        errors.append("market_diagnosis_gate_trace_required")
    else:
        node_ids = {
            str(item.get("node_id"))
            for item in gate_trace
            if isinstance(item, dict) and item.get("node_id") is not None
        }
        if gate_result == "proceed" and not MARKET_DIAGNOSIS_PROCEED_TRACE_NODES <= node_ids:
            errors.append("market_diagnosis_gate_trace_missing_proceed_nodes")
        if gate_result in ("wait", "unknown"):
            last = gate_trace[-1] if isinstance(gate_trace[-1], dict) else {}
            if last.get("answer") not in ("否", "等待"):
                errors.append(
                    "market_diagnosis_gate_wait_requires_negative_or_waiting_final_answer"
                )
        _validate_gate_trace_order(gate_trace, gate_result, errors)
        _validate_gate_trace_branch_consistency(gate_trace, diagnosis, errors)
        _validate_duplicate_bar_ranges(gate_trace, errors)
        for item in gate_trace:
            if not isinstance(item, dict):
                errors.append("market_diagnosis_gate_trace_item_must_be_object")
                continue
            _validate_gate_trace_item(item, max_k_seq, errors)

    return errors


def _max_feature_k_seq(feature_rows: list[dict[str, Any]] | None) -> int | None:
    seqs: list[int] = []
    for row in feature_rows or []:
        match = _K_ANY_RE.fullmatch(str(row.get("k", "")).strip())
        if match:
            seqs.append(int(match.group(1)))
    return max(seqs) if seqs else None


def _bar_label_exists(label: str, max_k_seq: int | None) -> bool:
    match = _K_ANY_RE.fullmatch(str(label or "").strip())
    if not match:
        return False
    seq = int(match.group(1))
    if seq < 1:
        return False
    return max_k_seq is None or seq <= max_k_seq


def _validate_gate_trace_item(
    item: dict[str, Any],
    max_k_seq: int | None,
    errors: list[str],
) -> None:
    node_id = str(item.get("node_id", "") or "").strip()
    if not node_id:
        errors.append("market_diagnosis_gate_trace_node_id_required")
    if node_id in MARKET_DIAGNOSIS_FORBIDDEN_GATE_NODES:
        errors.append("market_diagnosis_gate_trace_forbidden_stage_node")

    answer = item.get("answer")
    if answer not in MARKET_DIAGNOSIS_TRACE_ANSWERS:
        errors.append("market_diagnosis_gate_trace_answer_invalid")

    if item.get("skipped"):
        if answer != "不适用":
            errors.append("market_diagnosis_gate_trace_skipped_answer_invalid")
        return

    for field in ("question", "reason", "bar_range"):
        if field not in item:
            errors.append(f"market_diagnosis_gate_trace_{field}_required")

    bar_range = str(item.get("bar_range", "") or "").strip()
    if not bar_range:
        errors.append("market_diagnosis_gate_trace_bar_range_required")
        return
    if bar_range in ("不适用", "—", "全局", "GLOBAL"):
        return
    if "填写" in bar_range or bar_range.startswith("<") or "由你" in bar_range:
        errors.append("market_diagnosis_gate_trace_bar_range_placeholder")
        return

    seqs = _parse_k_range(bar_range)
    if not seqs:
        errors.append("market_diagnosis_gate_trace_bar_range_format_invalid")
        return
    if any(seq == 0 for seq in seqs):
        errors.append("market_diagnosis_gate_trace_bar_range_must_not_reference_k0")
    if any(seq < 1 for seq in seqs):
        errors.append("market_diagnosis_gate_trace_bar_range_invalid")
    if max_k_seq is not None and any(seq > max_k_seq for seq in seqs):
        errors.append("market_diagnosis_gate_trace_bar_range_out_of_frame")


def _parse_k_range(value: str) -> list[int]:
    text = value.strip().upper().replace(" ", "")
    range_match = _K_RANGE_RE.fullmatch(text)
    if range_match:
        older = int(range_match.group(1))
        newer = int(range_match.group(2))
        if older < newer:
            return [-1]
        return list(range(newer, older + 1))
    single_match = _K_SINGLE_RE.fullmatch(text)
    if single_match:
        return [int(single_match.group(1))]
    return []


def _validate_gate_trace_order(
    gate_trace: list[Any],
    gate_result: str,
    errors: list[str],
) -> None:
    node_ids = [
        str(item.get("node_id", "") or "")
        for item in gate_trace
        if isinstance(item, dict) and item.get("node_id")
    ]
    terminal_exempt = gate_result in ("wait", "unknown") and len(node_ids) > 1
    check_up_to = len(node_ids) - 1 if terminal_exempt else len(node_ids)
    for index in range(1, check_up_to):
        if _gate_trace_sort_key(node_ids[index]) < _gate_trace_sort_key(node_ids[index - 1]):
            errors.append("market_diagnosis_gate_trace_node_order_invalid")
            return


def _gate_trace_sort_key(node_id: str) -> tuple[int, int, str]:
    parts = str(node_id or "").split(".", 1)
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        return (999, 999, node_id)
    if len(parts) == 1:
        return (major, 0, node_id)
    try:
        return (major, int(parts[1]), node_id)
    except ValueError:
        return (major, 999, node_id)


def _validate_gate_trace_branch_consistency(
    gate_trace: list[Any],
    diagnosis: dict[str, Any],
    errors: list[str],
) -> None:
    cycle = str(diagnosis.get("cycle_position", "") or "").strip().lower()
    alt_cycle = str(diagnosis.get("alternative_cycle_position") or "").strip().lower()
    direction = str(diagnosis.get("direction", "") or "").strip().lower()

    item_12 = _find_gate_trace_item(gate_trace, "1.2")
    if item_12 and not item_12.get("skipped"):
        branch_cycle = _normalize_cycle_branch(item_12.get("branch"))
        if branch_cycle and cycle and branch_cycle not in (cycle, alt_cycle):
            errors.append("market_diagnosis_gate_trace_cycle_branch_conflict")

    item_23 = _find_gate_trace_item(gate_trace, "2.3")
    if item_23 and not item_23.get("skipped"):
        branch_direction = _normalize_direction_branch(item_23.get("branch"))
        if branch_direction and direction and branch_direction != direction:
            if not (cycle in MARKET_DIAGNOSIS_RANGE_CYCLES and branch_direction == "neutral"):
                errors.append("market_diagnosis_gate_trace_direction_branch_conflict")
        answer = str(item_23.get("answer", "") or "").strip()
        if answer == "中性" and direction not in ("neutral", ""):
            if cycle not in MARKET_DIAGNOSIS_RANGE_CYCLES:
                errors.append("market_diagnosis_gate_trace_direction_answer_conflict")


def _find_gate_trace_item(gate_trace: list[Any], node_id: str) -> dict[str, Any] | None:
    for item in gate_trace:
        if isinstance(item, dict) and str(item.get("node_id", "")) == node_id:
            return item
    return None


def _normalize_cycle_branch(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return _CYCLE_BRANCH_ALIASES.get(key, key.replace(" ", "_"))


def _normalize_direction_branch(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return _DIRECTION_BRANCH_ALIASES.get(key, key)


def _validate_duplicate_bar_ranges(gate_trace: list[Any], errors: list[str]) -> None:
    ranges: list[str] = []
    for item in gate_trace:
        if not isinstance(item, dict):
            continue
        if item.get("skipped") and item.get("answer") == "不适用":
            continue
        bar_range = str(item.get("bar_range", "") or "").strip()
        if bar_range and bar_range not in ("不适用", "—", "全局", "GLOBAL"):
            ranges.append(bar_range.upper().replace(" ", ""))
    if len(ranges) >= 4 and len(set(ranges)) == 1:
        errors.append("market_diagnosis_gate_trace_duplicate_bar_ranges")


def _looks_like_market_diagnosis(diagnosis: dict[str, Any]) -> bool:
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
