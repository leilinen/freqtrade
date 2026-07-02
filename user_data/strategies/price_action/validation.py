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
TRADE_DECISION_ACTIONABLE_ORDER_TYPES = {"限价单", "突破单", "市价单"}
TRADE_DECISION_TRACE_ANSWERS = {"是", "否", "中性", "等待", "不适用"}
TRADE_DECISION_CYCLE_ORDER = (
    "spike",
    "micro_channel",
    "tight_channel",
    "normal_channel",
    "broad_channel",
    "trending_tr",
    "trading_range",
    "extreme_tr",
)
TRADE_DECISION_DIRECTIONS = {"bullish", "bearish", "neutral"}
TRADE_DECISION_MIN_RISK_REWARD = 1.0

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
        feature_rows: list[dict[str, Any]] | None = None,
        strategies: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, ValidationResult]:
        checks: list[str] = []

        checks.append(CHECK_JSON_SYNTAX)
        try:
            parsed = parse_json_object(raw_response)
        except json.JSONDecodeError as exc:
            return None, ValidationResult(False, checks, [f"invalid_json:{exc.msg}"])
        except ValueError:
            return None, ValidationResult(False, checks, ["json_root_must_be_object"])

        return self._validate_parsed_core(
            parsed,
            checks,
            diagnosis=diagnosis,
            price_action_features=price_action_features,
            feature_rows=feature_rows,
            strategies=strategies or [],
        )

    def validate_parsed(
        self,
        parsed: dict[str, Any],
        *,
        diagnosis: dict[str, Any],
        price_action_features: dict[str, Any],
        feature_rows: list[dict[str, Any]] | None = None,
        strategies: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, ValidationResult]:
        """Validate an already-parsed (and typically normalized) dict.

        Skips JSON parsing — used when the orchestrator runs
        :func:`normalize_trade_decision` between parse and validate.
        """
        checks: list[str] = [CHECK_JSON_SYNTAX]
        return self._validate_parsed_core(
            parsed,
            checks,
            diagnosis=diagnosis,
            price_action_features=price_action_features,
            feature_rows=feature_rows,
            strategies=strategies or [],
        )

    def _validate_parsed_core(
        self,
        parsed: dict[str, Any],
        checks: list[str],
        *,
        diagnosis: dict[str, Any],
        price_action_features: dict[str, Any],
        feature_rows: list[dict[str, Any]] | None,
        strategies: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, ValidationResult]:
        """Shared validation pipeline (stage consistency → semantic → numeric)."""
        checks.append(CHECK_STAGE_CONSISTENCY)
        errors = self._stage_consistency_errors(parsed, diagnosis)
        if errors:
            return parsed, ValidationResult(False, checks, errors)

        checks.append(CHECK_SEMANTIC_REASONABLENESS)
        errors = self._semantic_errors(
            parsed,
            diagnosis,
            price_action_features,
            feature_rows,
            strategies,
        )
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
        feature_rows: list[dict[str, Any]] | None,
        strategies: list[dict[str, Any]],
    ) -> list[str]:
        del strategies
        decision = decision_json["decision"]
        order_type = str(decision.get("order_type", ""))
        order_direction = decision.get("order_direction")
        entry = _number(decision.get("entry_price"))
        stop = _number(decision.get("stop_loss_price"))
        tp1 = _number(decision.get("take_profit_price"))
        tp2 = _number(decision.get("take_profit_price_2"))
        errors: list[str] = []
        errors.extend(_decision_trace_errors(decision_json, feature_rows))

        if order_type == "不下单":
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

        errors.extend(_stage2_direction_errors(decision_json, diagnosis))
        errors.extend(_breakout_basis_errors(decision, feature_rows))

        if errors:
            return errors

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

        errors.extend(_trade_metric_errors(decision))

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
            errors.extend(_next_cycle_prediction_errors(prediction))

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


def _decision_trace_errors(
    decision_json: dict[str, Any],
    feature_rows: list[dict[str, Any]] | None,
) -> list[str]:
    if decision_json.get("gate_shortcircuited"):
        return []

    errors: list[str] = []
    trace = decision_json.get("decision_trace")
    if not isinstance(trace, list) or not trace:
        return ["decision_trace_must_be_non_empty"]

    max_k_seq = _max_feature_k_seq(feature_rows)
    for item in trace:
        if not isinstance(item, dict):
            errors.append("decision_trace_item_must_be_object")
            continue
        node_id = str(item.get("node_id", "") or "").strip()
        if not node_id:
            errors.append("decision_trace_node_id_required")
        if node_id == "0.3":
            errors.append("decision_trace_must_not_include_0_3")
        if item.get("answer") not in TRADE_DECISION_TRACE_ANSWERS:
            errors.append("decision_trace_answer_invalid")
        for field in ("question", "reason", "bar_range"):
            if field not in item:
                errors.append(f"decision_trace_{field}_required")
        if not item.get("skipped"):
            _validate_decision_trace_bar_range(item.get("bar_range"), max_k_seq, errors)

    terminal = decision_json.get("terminal") or {}
    decision = decision_json.get("decision") or {}
    order_type = decision.get("order_type")
    outcome = terminal.get("outcome")
    terminal_node = str(terminal.get("node_id", "") or "").strip()
    node_ids = [
        str(item.get("node_id", "") or "").strip()
        for item in trace
        if isinstance(item, dict) and item.get("node_id")
    ]

    if order_type == "不下单" and outcome == "trade":
        errors.append("no_trade_terminal_must_not_be_trade")
    if order_type in TRADE_DECISION_ACTIONABLE_ORDER_TYPES:
        if outcome != "trade":
            errors.append("actionable_decision_requires_trade_terminal")
        if terminal_node.startswith("14"):
            errors.append("trade_terminal_must_not_be_section_14")
        idx_103 = _index_of_node(node_ids, "10.3")
        if idx_103 < 0:
            errors.append("actionable_decision_requires_trader_equation_node_10_3")
        else:
            item_103 = trace[idx_103]
            if isinstance(item_103, dict) and item_103.get("answer") != "是":
                errors.append("trader_equation_node_10_3_must_be_yes")
        idx_11 = _first_node_index_with_prefix(node_ids, "11.")
        if idx_103 >= 0 and idx_11 >= 0 and idx_11 < idx_103:
            errors.append("order_method_nodes_must_follow_trader_equation")
        idx_9 = _first_node_index_with_prefix(node_ids, "9.")
        idx_101 = _index_of_node(node_ids, "10.1")
        if idx_9 < 0:
            errors.append("actionable_decision_requires_section_9_trace")
        if idx_9 >= 0 and idx_101 >= 0 and idx_9 > idx_101:
            errors.append("entry_signal_section_9_must_precede_stop_section_10_1")

    for earlier, later in (("10.1", "10.2"), ("10.2", "10.3")):
        earlier_idx = _index_of_node(node_ids, earlier)
        later_idx = _index_of_node(node_ids, later)
        if earlier_idx >= 0 and later_idx >= 0 and earlier_idx > later_idx:
            errors.append(f"decision_trace_order_{earlier}_before_{later}_required")

    ranks = [_decision_trace_sort_key(node_id)[0] for node_id in node_ids]
    for index in range(1, len(ranks)):
        if ranks[index] < ranks[index - 1]:
            errors.append("decision_trace_chapter_order_invalid")
            break

    return errors


def _validate_decision_trace_bar_range(
    value: Any,
    max_k_seq: int | None,
    errors: list[str],
) -> None:
    bar_range = str(value or "").strip()
    if not bar_range:
        errors.append("decision_trace_bar_range_required")
        return
    if bar_range in ("不适用", "—", "全局", "GLOBAL"):
        return
    if "填写" in bar_range or bar_range.startswith("<") or "由你" in bar_range:
        errors.append("decision_trace_bar_range_placeholder")
        return
    seqs = _parse_k_range(bar_range)
    if not seqs:
        errors.append("decision_trace_bar_range_format_invalid")
        return
    if any(seq == 0 for seq in seqs):
        errors.append("decision_trace_bar_range_must_not_reference_k0")
    if any(seq < 1 for seq in seqs):
        errors.append("decision_trace_bar_range_invalid")
    if max_k_seq is not None and any(seq > max_k_seq for seq in seqs):
        errors.append("decision_trace_bar_range_out_of_frame")


def _stage2_direction_errors(
    decision_json: dict[str, Any],
    diagnosis: dict[str, Any],
) -> list[str]:
    if decision_json.get("gate_shortcircuited"):
        return []
    decision = decision_json.get("decision") or {}
    order_type = decision.get("order_type")
    order_direction = decision.get("order_direction")
    if order_type not in TRADE_DECISION_ACTIONABLE_ORDER_TYPES:
        return []
    if order_direction not in ("做多", "做空"):
        return []

    stage1_direction = str(diagnosis.get("direction", "") or "").strip().lower()
    if stage1_direction not in ("bullish", "bearish"):
        return []
    needed_direction = "bullish" if order_direction == "做多" else "bearish"
    if stage1_direction == needed_direction:
        return []
    if _decision_trace_documents_direction_override(
        decision_json.get("decision_trace"),
        needed_direction,
    ):
        return []
    return ["order_direction_conflicts_with_stage1_direction_without_node_2_3"]


def _decision_trace_documents_direction_override(trace: Any, direction: str) -> bool:
    if not isinstance(trace, list):
        return False
    for item in trace:
        if not isinstance(item, dict):
            continue
        if str(item.get("node_id", "") or "").strip() != "2.3":
            continue
        branch = _normalize_direction_branch(item.get("branch"))
        reason = str(item.get("reason", "") or "").strip().lower()
        if branch == direction or direction in reason:
            return True
    return False


def _breakout_basis_errors(
    decision: dict[str, Any],
    feature_rows: list[dict[str, Any]] | None,
) -> list[str]:
    if decision.get("order_type") != "突破单":
        return []

    errors: list[str] = []
    direction = decision.get("order_direction")
    extreme = decision.get("entry_basis_extreme")
    basis_bar = decision.get("entry_basis_bar")

    if basis_bar is None:
        errors.append("breakout_order_requires_entry_basis_bar")
    if extreme not in ("high", "low"):
        errors.append("breakout_order_requires_entry_basis_extreme")
    if not decision.get("entry_rule"):
        errors.append("breakout_order_requires_entry_rule")
    if direction == "做多" and extreme == "low":
        errors.append("long_breakout_order_must_use_high_extreme")
    if direction == "做空" and extreme == "high":
        errors.append("short_breakout_order_must_use_low_extreme")
    if errors:
        return errors

    basis_seq = _parse_k_seq(basis_bar)
    if basis_seq is None:
        return ["breakout_entry_basis_bar_invalid"]

    basis_row = _feature_row_by_k(feature_rows, basis_seq)
    if feature_rows is not None and basis_row is None:
        return ["breakout_entry_basis_bar_out_of_frame"]
    if basis_row is None:
        return []

    entry = _number(decision.get("entry_price"))
    if entry is None:
        return []
    if direction == "做多" and extreme == "high":
        high = _number(basis_row.get("high"))
        if high is not None and entry <= high:
            errors.append("long_breakout_entry_must_be_above_basis_high")
    if direction == "做空" and extreme == "low":
        low = _number(basis_row.get("low"))
        if low is not None and entry >= low:
            errors.append("short_breakout_entry_must_be_below_basis_low")
    return errors


def _trade_metric_errors(decision: dict[str, Any]) -> list[str]:
    if decision.get("order_type") not in TRADE_DECISION_ACTIONABLE_ORDER_TYPES:
        return []
    rr = _compute_risk_reward(
        decision.get("entry_price"),
        decision.get("take_profit_price"),
        decision.get("stop_loss_price"),
        decision.get("order_direction"),
    )
    if rr is None:
        return ["trade_prices_must_form_positive_risk_reward"]

    errors: list[str] = []
    ratio = rr["ratio"]
    if ratio < TRADE_DECISION_MIN_RISK_REWARD:
        errors.append("risk_reward_below_minimum")

    win_rate = _number(decision.get("estimated_win_rate"))
    if win_rate is None:
        errors.append("actionable_decision_requires_estimated_win_rate")
    elif not _passes_trader_equation(win_rate, rr["risk"], rr["reward"]):
        errors.append("trader_equation_fails")
    return errors


def _compute_risk_reward(
    entry: Any,
    take_profit: Any,
    stop_loss: Any,
    direction: Any,
) -> dict[str, float] | None:
    entry_number = _number(entry)
    take_profit_number = _number(take_profit)
    stop_loss_number = _number(stop_loss)
    if entry_number is None or take_profit_number is None or stop_loss_number is None:
        return None

    if direction == "做多":
        risk = entry_number - stop_loss_number
        reward = take_profit_number - entry_number
    elif direction == "做空":
        risk = stop_loss_number - entry_number
        reward = entry_number - take_profit_number
    else:
        return None

    if risk <= 0 or reward <= 0:
        return None
    return {"risk": risk, "reward": reward, "ratio": reward / risk}


def _passes_trader_equation(win_rate_pct: float, risk: float, reward: float) -> bool:
    if risk <= 0 or reward <= 0:
        return False
    probability = max(0.0, min(100.0, win_rate_pct)) / 100.0
    return probability * reward > (1.0 - probability) * risk


def _next_cycle_prediction_errors(prediction: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    unpredictable = prediction.get("unpredictable")
    if not isinstance(unpredictable, bool):
        return ["next_cycle_prediction_unpredictable_must_be_bool"]

    if unpredictable:
        if prediction.get("cycle") is not None:
            errors.append("next_cycle_prediction_cycle_must_be_null_when_unpredictable")
        if prediction.get("direction") is not None:
            errors.append("next_cycle_prediction_direction_must_be_null_when_unpredictable")
        if prediction.get("probabilities") is not None:
            errors.append("next_cycle_prediction_probabilities_must_be_null_when_unpredictable")
        return errors

    cycle = prediction.get("cycle")
    if cycle not in TRADE_DECISION_CYCLE_ORDER:
        errors.append("next_cycle_prediction_cycle_invalid")
    direction = prediction.get("direction")
    if direction not in TRADE_DECISION_DIRECTIONS:
        errors.append("next_cycle_prediction_direction_invalid")

    probabilities = prediction.get("probabilities")
    if not isinstance(probabilities, dict):
        errors.append("next_cycle_prediction_probabilities_must_be_object")
        return errors

    for key in TRADE_DECISION_CYCLE_ORDER:
        value = probabilities.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            errors.append(f"next_cycle_prediction_probability_{key}_invalid")
    if errors:
        return errors

    total = sum(probabilities[key] for key in TRADE_DECISION_CYCLE_ORDER)
    if not 99 <= total <= 101:
        errors.append("next_cycle_prediction_probabilities_sum_invalid")

    max_value = max(probabilities[key] for key in TRADE_DECISION_CYCLE_ORDER)
    winners = [key for key in TRADE_DECISION_CYCLE_ORDER if probabilities[key] == max_value]
    if cycle not in winners:
        errors.append("next_cycle_prediction_cycle_must_match_probability_argmax")
    return errors


def parse_json_object(raw_response: str) -> dict[str, Any]:
    """Parse a required JSON object response.

    Some OpenAI-compatible providers accept ``response_format`` but still wrap
    JSON in Markdown fences. Keep raw response persistence unchanged, but make
    validation tolerant enough to parse those provider responses.
    When ``json.loads`` fails, first try to repair truncated / unbalanced /
    control-char-laden JSON (ported from PA_Agent json_validator.py); only
    fall back to brace-fence extraction if repair returns nothing.
    """
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        repaired = _try_repair_json_syntax(raw_response, allow_tail_inject=True)
        if repaired is not None:
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                parsed = json.loads(_extract_json_object_text(raw_response))
        else:
            parsed = json.loads(_extract_json_object_text(raw_response))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response root must be a JSON object")
    return parsed


def _extract_json_object_text(raw_response: str) -> str:
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        fenced = "\n".join(lines).strip()
        if fenced:
            return fenced

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


# ================================================================
# JSON syntax repair (ported from PA_Agent json_validator.py:144-372)
# ================================================================


def _escape_control_chars_in_json_strings(text: str) -> str:
    """Escape raw newlines/tabs/control chars inside JSON string literals.

    glm-5.x 偶发在 reason 字段里塞裸 \\n / \\t / 其他控制字符,
    会让 ``json.loads`` 直接报 ``JSONDecodeError``。
    本函数用状态机扫描,只在字符串字面量内转义,结构字符不受影响。
    Ported from PA_Agent ``json_validator.py:144-177``.
    """
    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            continue
        if escape:
            escape = False
            out.append(ch)
            continue
        if ch == "\\":
            escape = True
            out.append(ch)
            continue
        if ch == '"':
            in_string = False
            out.append(ch)
            continue
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch < " ":
            # 其他 ASCII 控制字符直接丢弃(LLM 不应在中文 JSON 里塞这些)。
            continue
        else:
            out.append(ch)
    return "".join(out)


def _balance_json_brackets(text: str) -> str:
    """Close unclosed ``{`` / ``[`` outside JSON strings via stack scan.

    处理 LLM 输出被 max_tokens 截断、最后一根 ``}`` 丢失的情况。
    不能简单 append ``}`` —— 需要按栈深度反向闭合。
    Ported from PA_Agent ``json_validator.py:305-331``.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            stack.append("{")
        elif ch == "[":
            stack.append("[")
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    closers = "".join("]" if opener == "[" else "}" for opener in reversed(stack))
    return text + closers


def _inject_stage1_missing_tail(text: str) -> str:
    """Append minimal gate_trace stub when stage1 JSON was truncated mid-object.

    仅在 ``text`` 是**未闭合** JSON(外层 ``{`` 没有对应 ``}``)时才注入。
    完整闭合的 JSON(``{...}``)不应被注入,否则会产生 ``Extra data`` 错误。
    """
    tail = text.rstrip()
    if not tail.endswith((",", "]", "}")):
        return text
    # 检查是否未闭合:用 _balance_json_brackets 试一下,若需要补 } 才说明截断了
    balanced = _balance_json_brackets(tail)
    if balanced == tail:
        # 已经闭合(多余的尾随 , ] } 不存在),不注入
        return text
    if not tail.endswith(","):
        tail += ","
    stub_trace = (
        '{"node_id":"AUTO","question":"输出是否在gate_trace前被截断?",'
        '"answer":"否","reason":"JSON在gate_trace前截断,程序已补全最小闸门记录",'
        '"bar_range":"K1"}'
    )
    tail += f'"gate_trace":[{stub_trace}],"gate_result":"unknown"'
    return _balance_json_brackets(tail)


def _try_repair_json_syntax(
    text: str,
    *,
    allow_tail_inject: bool = False,
) -> str | None:
    """Return repaired JSON text when truncation caused a syntax error, else None.

    编排顺序:
    1. 转义字符串内控制字符(最常见,无副作用)
    2. 若 stage1 且允许,先尝试 tail_inject(补齐 gate_trace 末尾)
    3. 平衡未闭合的括号
    4. 修复后必须能 ``json.loads`` 通过,否则视为不可修复

    Ported from PA_Agent ``json_validator.py:352-372``.
    """
    if not text.strip().startswith("{"):
        return None
    candidate = text.rstrip()
    if allow_tail_inject:
        candidate = _inject_stage1_missing_tail(candidate)
    candidate = _balance_json_brackets(candidate)
    # 控制字符转义必须最后做(否则会破坏栈扫描的状态机)
    escaped = _escape_control_chars_in_json_strings(candidate)
    if escaped != candidate:
        candidate = escaped
    if candidate == text.rstrip():
        return None
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate


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
    _repair_gate_result(diagnosis)  # 校验前先修正 wait/unknown → proceed
    gate_result = str(diagnosis.get("gate_result", "")).lower()
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
        _sync_gate_12_with_cycle(diagnosis)
        _sync_gate_23_with_direction(diagnosis)
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


def _parse_k_seq(value: Any) -> int | None:
    match = _K_ANY_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    return int(match.group(1))


def _feature_row_by_k(
    feature_rows: list[dict[str, Any]] | None,
    seq: int,
) -> dict[str, Any] | None:
    for row in feature_rows or []:
        if str(row.get("k", "")).strip().upper() == f"K{seq}":
            return row
    return None


def _index_of_node(node_ids: list[str], node_id: str) -> int:
    try:
        return node_ids.index(node_id)
    except ValueError:
        return -1


def _first_node_index_with_prefix(node_ids: list[str], prefix: str) -> int:
    for index, node_id in enumerate(node_ids):
        if node_id.startswith(prefix):
            return index
    return -1


def _decision_trace_sort_key(node_id: str) -> tuple[int, int, str]:
    parts = str(node_id or "").split(".", 1)
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        return (999, 999, node_id)
    if len(parts) == 1:
        return (major, 0, node_id)
    minor_match = re.match(r"^(\d+)", parts[1])
    minor = int(minor_match.group(1)) if minor_match else 999
    return (major, minor, node_id)


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


_VALID_DIRECTION_BRANCHES = frozenset({"bullish", "bearish", "neutral"})

# 通用的 yes/no 类分支值,不是合法的 cycle 名(参考 PA_Agent trace_normalize.py:26-28)
_GATE_12_GENERIC_BRANCHES = frozenset(
    {"yes", "no", "y", "n", "是", "否", "true", "false", ""}
)

# 合法的 cycle branch 取值
_VALID_CYCLE_BRANCHES = frozenset(
    {
        "trading_range",
        "trending_tr",
        "extreme_tr",
        "spike",
        "micro_channel",
        "tight_channel",
        "normal_channel",
        "broad_channel",
        "unknown",
    }
)


def _infer_direction_from_reason(reason: str) -> str | None:
    """启发式从 reason 文本推断方向(参考 PA_Agent trace_normalize)。"""
    blob = reason.lower()
    if any(tok in blob for tok in ("多头", "做多", "bullish", "bull", "上涨", "向上")):
        return "bullish"
    if any(tok in blob for tok in ("空头", "做空", "bearish", "bear", "下跌", "向下")):
        return "bearish"
    if any(tok in blob for tok in ("中性", "震荡", "neutral", "横盘")):
        return "neutral"
    return None


def _sync_gate_12_with_cycle(diagnosis: dict[str, Any]) -> None:
    """Normalize gate_trace node 1.2 so branch aligns with top-level cycle_position.

    参考 PA_Agent ``_sync_gate_12_branch_with_cycle``(trace_normalize.py:649-680)。

    LLM 常见偏差:把 node 1.2 的 branch 填成 signal 阶段术语(如 tr_identified)
    或 yes/no,而非真实的 cycle 名(trading_range / spike 等)。
    这里就地修正,避免 ``market_diagnosis_gate_trace_cycle_branch_conflict``。
    """
    gate_trace = diagnosis.get("gate_trace")
    if not isinstance(gate_trace, list):
        return
    cycle = _normalize_cycle_branch(diagnosis.get("cycle_position"))
    if not cycle:
        return
    alt = _normalize_cycle_branch(diagnosis.get("alternative_cycle_position"))
    for item in gate_trace:
        if not isinstance(item, dict):
            continue
        if str(item.get("node_id", "")).strip() != "1.2":
            continue
        if item.get("skipped"):
            return
        br_raw = item.get("branch")
        br = _normalize_cycle_branch(br_raw)
        # branch 是合法 cycle 名 → 仅做格式归一化,不动语义
        if br and br in _VALID_CYCLE_BRANCHES:
            if br != str(br_raw).strip().lower().replace(" ", "_"):
                item["branch"] = br
            return
        # branch 是 yes/no/空 这类通用值,或非法的 signal 术语 → 走兜底
        ans = str(item.get("answer", "") or "").strip()
        if ans == "是":
            item["branch"] = cycle
            return
        if ans == "否":
            item["branch"] = "unknown"
            return
        # answer 也没有线索 → 直接用顶层 cycle 兜底
        item["branch"] = cycle
        return


def _repair_gate_result(diagnosis: dict[str, Any]) -> None:
    """Fix gate_result when AI sets wait/unknown despite no blocking condition.

    参考 PA_Agent ``_repair_gate_result``(trace_normalize.py:778-811)。

    Per prompt rules, gate_result=wait/unknown is only valid for:
    - §1.2 answer≠是 (cannot identify cycle)
    - §1.3 answer=是 (market is extremely chaotic)

    If neither condition holds but gate_result is wait/unknown, force to proceed.
    让"LLM 误判 wait + 末节点为是"的输出在校验前被纠正,避免无谓失败。
    """
    gate_result = str(diagnosis.get("gate_result", "")).strip().lower()
    if gate_result not in ("wait", "unknown"):
        return
    gate = diagnosis.get("gate_trace")
    if not isinstance(gate, list) or not gate:
        return
    node_12_block = any(
        isinstance(item, dict)
        and str(item.get("node_id", "")) == "1.2"
        and str(item.get("answer", "")).strip() != "是"
        for item in gate
    )
    node_13_block = any(
        isinstance(item, dict)
        and str(item.get("node_id", "")) == "1.3"
        and str(item.get("answer", "")).strip() == "是"
        for item in gate
    )
    if not node_12_block and not node_13_block:
        diagnosis["gate_result"] = "proceed"


def _sync_gate_23_with_direction(diagnosis: dict[str, Any]) -> None:
    """Normalize gate_trace node 2.3 so branch/answer align with top-level direction.

    参考 PA_Agent ``_sync_gate_23_answer_with_direction``(trace_normalize.py:746-775)。

    LLM 常见偏差:把 node 2.3 的 branch 填成 signal_quality 术语
    (如 no_valid_breakout),或漏填 branch,导致与顶层 direction 冲突。
    这里就地修正 branch/answer,避免 ``market_diagnosis_gate_trace_direction_branch_conflict``。
    """
    gate_trace = diagnosis.get("gate_trace")
    if not isinstance(gate_trace, list):
        return
    top_dir = _normalize_direction_branch(diagnosis.get("direction"))
    for item in gate_trace:
        if not isinstance(item, dict):
            continue
        if str(item.get("node_id", "")).strip() != "2.3":
            continue
        if item.get("skipped"):
            return
        branch_dir = _normalize_direction_branch(item.get("branch"))
        # branch 值不是合法方向时(如 no_valid_breakout),视为未知,走推断/兜底
        if branch_dir and branch_dir not in _VALID_DIRECTION_BRANCHES:
            branch_dir = None
        if not branch_dir:
            branch_dir = _infer_direction_from_reason(str(item.get("reason", "") or ""))
        if not branch_dir and top_dir:
            branch_dir = top_dir
        if branch_dir:
            item["branch"] = branch_dir
        ans = str(item.get("answer", "") or "").strip()
        if branch_dir in ("bullish", "bearish"):
            if ans == "中性":
                item["answer"] = "是"
        elif branch_dir == "neutral" and ans in ("是", "否"):
            item["answer"] = "中性"
        return


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
        # unknown 是归一化后的兜底值(answer=否 时),不应判冲突
        if (
            branch_cycle
            and branch_cycle != "unknown"
            and cycle
            and branch_cycle not in (cycle, alt_cycle)
        ):
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
