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
        return {"valid": self.valid, "status": self.status, "checks": self.checks, "errors": self.errors}


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
        if diagnosis.get("stage") != "market_diagnosis":
            errors.append("diagnosis_stage_must_be_market_diagnosis")

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


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number
