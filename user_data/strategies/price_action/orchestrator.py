"""PA_Agent-style price-action analysis orchestration."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Callable

from pandas import DataFrame

from .experience import retrieve_experience_cases
from .features import PriceActionFeatureResult, build_price_action_features
from .llm import OpenAIJsonClient
from .prompts import (
    build_incremental_market_diagnosis_messages,
    build_market_diagnosis_messages,
    build_trade_decision_messages,
    prompt_template_metadata,
)
from .repository import PriceActionRepository
from .router import route_strategies
from .validation import DecisionValidator, parse_json_object, validate_market_diagnosis


logger = logging.getLogger(__name__)
USAGE_FIELDS = (
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class AnalysisOutcome:
    """Summary returned to the background worker."""

    status: str
    symbol: str
    timeframe: str
    candle_time: object | None
    decision: dict[str, Any] | None
    errors: list[str]


class PriceActionOrchestrator:
    """Run feature engineering, market diagnosis, strategy routing, and trade decision."""

    def __init__(
        self,
        *,
        repository: PriceActionRepository | None,
        llm_client: OpenAIJsonClient,
        config: dict[str, Any],
        validator: DecisionValidator | None = None,
    ) -> None:
        self.repository = repository
        self.llm_client = llm_client
        self.config = config
        self.validator = validator or DecisionValidator()

    def analyze(
        self,
        *,
        symbol: str,
        dataframe: DataFrame,
        timeframe: str,
        market: str,
        notifier: Any | None = None,
        chart_generator: Callable | None = None,
    ) -> AnalysisOutcome:
        """Run the full pipeline for one symbol/timeframe closed candle."""
        features: PriceActionFeatureResult | None = None
        raw_market_diagnosis: str | None = None
        raw_trade_decision: str | None = None
        diagnosis: dict[str, Any] | None = None
        decision_json: dict[str, Any] | None = None
        market_diagnosis_messages: list[dict[str, str]] | None = None
        trade_decision_messages: list[dict[str, str]] | None = None
        validation_status: str | None = None
        validation_errors: list[str] = []
        raw_responses: dict[str, Any] = {}
        usage_total: dict[str, Any] = {}
        exception_info: dict[str, Any] | None = None
        current_stage = "feature_engineering"
        selected = []
        experience_cases: list[dict[str, Any]] = []
        prompt_metadata: dict[str, Any] = {
            "model": self.llm_client.model,
            "base_url": self.llm_client.base_url,
            "response_format": {"type": "json_object"},
            "prompt_templates": prompt_template_metadata(),
            "decision_stance": _decision_stance(self.config),
        }

        try:
            features = build_price_action_features(
                dataframe,
                symbol=symbol,
                timeframe=timeframe,
                market=market,
                window=int(self.config.get("pa_llm_window", 30)),
                warmup=int(self.config.get("pa_llm_warmup", 50)),
            )

            previous = (
                self.repository.get_previous_successful_analysis(
                    symbol=symbol,
                    timeframe=timeframe,
                    before_time=features.candle_time,
                )
                if self.repository
                else None
            )

            current_stage = "market_diagnosis"
            new_bar_count = _new_bar_count_since_previous(features, previous)
            if _use_incremental_stage1(self.config, previous, new_bar_count):
                market_diagnosis_messages = build_incremental_market_diagnosis_messages(
                    features,
                    previous_analysis=previous or {},
                    new_bar_count=int(new_bar_count or 0),
                )
                prompt_metadata["market_diagnosis_mode"] = "incremental"
                prompt_metadata["incremental_new_bar_count"] = new_bar_count
            else:
                market_diagnosis_messages = build_market_diagnosis_messages(features)
                prompt_metadata["market_diagnosis_mode"] = "full"

            raw_market_diagnosis, diagnosis, market_diagnosis_messages = (
                self._call_market_diagnosis_with_retry(
                    market_diagnosis_messages,
                    feature_rows=features.rows,
                    raw_responses=raw_responses,
                )
            )
            usage_total = _usage_total_from_responses(raw_responses)

            strategies = route_strategies(diagnosis)
            selected = [strategy.as_dict() for strategy in strategies]
            experience_cases = retrieve_experience_cases(
                self.repository,
                market=market,
                timeframe=timeframe,
                diagnosis=diagnosis,
                limit=int(self.config.get("pa_experience_limit", 3)),
            )

            gate_result = str(diagnosis.get("gate_result", "proceed")).lower()
            if gate_result in ("wait", "unknown"):
                current_stage = "trade_decision"
                decision_json = self._build_gate_wait_decision(diagnosis)
                raw_trade_decision = json.dumps(decision_json, ensure_ascii=False)
                raw_responses["trade_decision"] = {
                    "stage": "trade_decision",
                    "model": "program",
                    "content": raw_trade_decision,
                    "usage": {},
                    "generated_by": "gate_short_circuit",
                }
                decision_json, validation = self.validator.validate(
                    raw_trade_decision,
                    diagnosis=diagnosis,
                    price_action_features=features.latest_features,
                    feature_rows=features.rows,
                    strategies=selected,
                )
                validation_status = validation.status
                validation_errors = validation.errors
                status = "success" if validation.valid else "invalid"
                exception_info = (
                    None
                    if validation.valid
                    else _validation_exception("trade_decision", validation_errors)
                )

                self._save(
                    features=features,
                    status=status,
                    market_diagnosis_messages=market_diagnosis_messages,
                    trade_decision_messages=[],
                    diagnosis=diagnosis,
                    selected=selected,
                    experience_cases=experience_cases,
                    decision=decision_json,
                    validation_status=validation_status,
                    validation_errors=validation_errors,
                    prompt_metadata={**prompt_metadata, "validation_checks": validation.checks},
                    raw_responses=raw_responses,
                    usage_total=usage_total,
                    exception=exception_info,
                )

                if validation.valid and self._should_notify(decision_json):
                    self._notify(
                        notifier=notifier,
                        symbol=symbol,
                        dataframe=dataframe,
                        price_action_features=features,
                        diagnosis=diagnosis,
                        selected=selected,
                        decision=decision_json,
                        validation=validation.as_dict(),
                        chart_generator=chart_generator,
                    )

                return AnalysisOutcome(
                    status=status,
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_time=features.candle_time,
                    decision=decision_json,
                    errors=validation_errors,
                )

            current_stage = "trade_decision"
            trade_decision_messages = build_trade_decision_messages(
                features=features,
                diagnosis=diagnosis,
                strategies=strategies,
                experience_cases=experience_cases,
                previous_decision=previous,
                decision_stance=_decision_stance(self.config),
            )
            raw_trade_decision, decision_json, validation, trade_decision_messages = (
                self._call_trade_decision_with_retry(
                    trade_decision_messages,
                    diagnosis=diagnosis,
                    price_action_features=features.latest_features,
                    feature_rows=features.rows,
                    strategies=selected,
                    raw_responses=raw_responses,
                )
            )
            usage_total = _usage_total_from_responses(raw_responses)
            validation_status = validation.status
            validation_errors = validation.errors
            status = "success" if validation.valid else "invalid"
            exception_info = (
                None
                if validation.valid
                else _validation_exception("trade_decision", validation_errors)
            )

            self._save(
                features=features,
                status=status,
                market_diagnosis_messages=market_diagnosis_messages,
                trade_decision_messages=trade_decision_messages,
                diagnosis=diagnosis,
                selected=selected,
                experience_cases=experience_cases,
                decision=decision_json,
                validation_status=validation_status,
                validation_errors=validation_errors,
                prompt_metadata={**prompt_metadata, "validation_checks": validation.checks},
                raw_responses=raw_responses,
                usage_total=usage_total,
                exception=exception_info,
            )

            if validation.valid and self._should_notify(decision_json):
                self._notify(
                    notifier=notifier,
                    symbol=symbol,
                    dataframe=dataframe,
                    price_action_features=features,
                    diagnosis=diagnosis,
                    selected=selected,
                    decision=decision_json,
                    validation=validation.as_dict(),
                    chart_generator=chart_generator,
                )

            return AnalysisOutcome(
                status=status,
                symbol=symbol,
                timeframe=timeframe,
                candle_time=features.candle_time,
                decision=decision_json,
                errors=validation_errors,
            )
        except Exception as exc:
            usage_total = _usage_total_from_responses(raw_responses)
            exception_info = _exception_info(current_stage, exc)
            logger.warning(
                "PA analysis failed for %s %s: %s",
                symbol,
                timeframe,
                exc,
                exc_info=True,
            )
            validation_errors = [f"{type(exc).__name__}:{exc}"]
            if features is not None:
                self._save(
                    features=features,
                    status="failed",
                    market_diagnosis_messages=market_diagnosis_messages,
                    trade_decision_messages=trade_decision_messages,
                    diagnosis=diagnosis,
                    selected=selected,
                    experience_cases=experience_cases,
                    decision=decision_json,
                    validation_status=validation_status or "failed",
                    validation_errors=validation_errors,
                    prompt_metadata=prompt_metadata,
                    raw_responses=raw_responses,
                    usage_total=usage_total,
                    exception=exception_info,
                )
            return AnalysisOutcome(
                status="failed",
                symbol=symbol,
                timeframe=timeframe,
                candle_time=features.candle_time if features else None,
                decision=decision_json,
                errors=validation_errors,
            )

    def _save(
        self,
        *,
        features: PriceActionFeatureResult,
        status: str,
        market_diagnosis_messages: list[dict[str, str]] | None,
        trade_decision_messages: list[dict[str, str]] | None,
        diagnosis: dict[str, Any] | None,
        selected: list[dict[str, Any]],
        experience_cases: list[dict[str, Any]],
        decision: dict[str, Any] | None,
        validation_status: str | None,
        validation_errors: list[str],
        prompt_metadata: dict[str, Any],
        raw_responses: dict[str, Any],
        usage_total: dict[str, Any],
        exception: dict[str, Any] | None,
    ) -> None:
        if not self.repository:
            return
        self.repository.save_analysis(
            market=features.market,
            symbol=features.symbol,
            timeframe=features.timeframe,
            candle_time=features.candle_time,
            status=status,
            kline_table=features.kline_table,
            feature_table=features.feature_table,
            price_action_features=features.as_dict(),
            market_diagnosis_messages=market_diagnosis_messages,
            trade_decision_messages=trade_decision_messages,
            market_diagnosis=diagnosis,
            selected_strategies=selected,
            experience_cases=experience_cases,
            trade_decision=decision,
            validation_status=validation_status,
            validation_errors=validation_errors,
            prompt_metadata=prompt_metadata,
            raw_responses=raw_responses,
            usage_total=usage_total,
            exception=exception,
        )

    def _capture_llm_response(self, stage: str, content: str) -> dict[str, Any]:
        raw = getattr(self.llm_client, "last_response", None)
        if isinstance(raw, dict) and raw.get("stage") == stage:
            return dict(raw)
        return {
            "stage": stage,
            "model": getattr(self.llm_client, "model", None),
            "content": content,
            "usage": {},
        }

    def _call_market_diagnosis_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        feature_rows: list[dict[str, Any]],
        raw_responses: dict[str, Any],
    ) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
        retry_limit = _validation_retry_limit(self.config)
        failed_attempts: list[dict[str, Any]] = []
        current_messages = list(messages)
        last_errors: list[str] = []

        for attempt in range(retry_limit + 1):
            raw_text = self.llm_client.complete_json(
                current_messages,
                stage="market_diagnosis",
            )
            raw_record = self._capture_llm_response("market_diagnosis", raw_text)
            try:
                parsed = parse_json_object(raw_text)
                errors = validate_market_diagnosis(parsed, feature_rows=feature_rows)
            except Exception as exc:
                parsed = None
                errors = [f"{type(exc).__name__}:{exc}"]
            if not errors and isinstance(parsed, dict):
                if failed_attempts:
                    raw_record["retry_attempts"] = failed_attempts
                raw_responses["market_diagnosis"] = raw_record
                return raw_text, parsed, current_messages

            last_errors = errors
            raw_record["validation_errors"] = errors
            if attempt >= retry_limit:
                if failed_attempts:
                    raw_record["retry_attempts"] = failed_attempts
                raw_responses["market_diagnosis"] = raw_record
                break
            failed_attempts.append(raw_record)
            current_messages = _append_retry_feedback(
                current_messages,
                stage="market_diagnosis",
                errors=errors,
            )

        raise ValueError(f"market_diagnosis_invalid:{','.join(last_errors)}")

    def _call_trade_decision_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        diagnosis: dict[str, Any],
        price_action_features: dict[str, Any],
        feature_rows: list[dict[str, Any]],
        strategies: list[dict[str, Any]],
        raw_responses: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None, Any, list[dict[str, str]]]:
        retry_limit = _validation_retry_limit(self.config)
        failed_attempts: list[dict[str, Any]] = []
        current_messages = list(messages)

        for attempt in range(retry_limit + 1):
            raw_text = self.llm_client.complete_json(
                current_messages,
                stage="trade_decision",
            )
            raw_record = self._capture_llm_response("trade_decision", raw_text)
            decision_json, validation = self.validator.validate(
                raw_text,
                diagnosis=diagnosis,
                price_action_features=price_action_features,
                feature_rows=feature_rows,
                strategies=strategies,
            )
            if validation.valid:
                if failed_attempts:
                    raw_record["retry_attempts"] = failed_attempts
                raw_responses["trade_decision"] = raw_record
                return raw_text, decision_json, validation, current_messages

            raw_record["validation_errors"] = validation.errors
            if attempt >= retry_limit:
                if failed_attempts:
                    raw_record["retry_attempts"] = failed_attempts
                raw_responses["trade_decision"] = raw_record
                return raw_text, decision_json, validation, current_messages
            failed_attempts.append(raw_record)
            current_messages = _append_retry_feedback(
                current_messages,
                stage="trade_decision",
                errors=validation.errors,
            )

        raise RuntimeError("unreachable_trade_decision_retry_state")

    def _should_notify(self, decision_json: dict[str, Any] | None) -> bool:
        if not decision_json:
            return False
        decision = decision_json.get("decision") or {}
        order_type = decision.get("order_type")
        order_direction = decision.get("order_direction")
        if order_type in ("限价单", "突破单", "市价单") and order_direction in (
            "做多",
            "做空",
        ):
            return True
        return bool(self.config.get("pa_notify_wait", False))

    def _build_gate_wait_decision(self, diagnosis: dict[str, Any]) -> dict[str, Any]:
        gate_result = str(diagnosis.get("gate_result", "wait")).lower()
        trace = diagnosis.get("gate_trace") or []
        final_reason = ""
        if trace and isinstance(trace[-1], dict):
            final_reason = str(trace[-1].get("reason") or "")
        reason = final_reason or str(diagnosis.get("risk_warning") or "市场诊断闸门未通过")
        confidence = diagnosis.get("diagnosis_confidence")
        try:
            confidence_value = max(0.0, min(float(confidence) / 100.0, 1.0))
        except (TypeError, ValueError):
            confidence_value = 0
        else:
            confidence_value = int(round(confidence_value * 100))
        return {
            "decision": {
                "order_direction": None,
                "order_type": "不下单",
                "entry_price": None,
                "entry_basis_bar": None,
                "entry_basis_extreme": None,
                "entry_rule": None,
                "take_profit_price": None,
                "take_profit_price_2": None,
                "stop_loss_price": None,
                "reasoning": (
                    f"市场诊断 gate_result={gate_result}，跳过交易决策：{reason}"
                ),
                "diagnosis_confidence": confidence_value,
                "diagnosis_confidence_reasoning": str(
                    diagnosis.get("risk_warning") or reason
                ),
                "trade_confidence": 0,
                "trade_confidence_reasoning": "阶段一闸门未通过，未进入交易评估。",
                "estimated_win_rate": None,
                "estimated_win_rate_reasoning": None,
                "key_factors": list(diagnosis.get("key_signals") or []),
                "watch_points": [reason],
                "risk_assessment": str(diagnosis.get("risk_warning") or reason),
                "invalidation_condition": None,
            },
            "diagnosis_summary": {
                "cycle_position": diagnosis.get("cycle_position", "unknown"),
                "direction": diagnosis.get("direction", "neutral"),
                "key_signals": list(diagnosis.get("key_signals") or []),
            },
            "decision_trace": [
                {
                    "node_id": "2.5",
                    "question": "市场诊断闸门是否允许进入交易决策？",
                    "answer": "否" if gate_result == "wait" else "等待",
                    "reason": reason,
                    "branch": gate_result,
                    "section": "市场诊断闸门",
                    "bar_range": "全局",
                }
            ],
            "terminal": {
                "node_id": "2.5",
                "outcome": "wait",
                "label": "市场诊断闸门短路，不进入交易决策",
            },
            "gate_shortcircuited": True,
            "next_cycle_prediction": {
                "cycle": None,
                "direction": None,
                "probabilities": None,
                "reasoning": "阶段一闸门未通过，未进入阶段二周期演变评估。",
                "unpredictable": True,
                "features_used": ["stage1_diagnosis"],
            },
            "next_bar_prediction": {
                "direction": None,
                "probabilities": None,
                "reasoning": "阶段一闸门未通过，未进行下一根 K 预测。",
                "unpredictable": True,
                "features_used": ["stage1_diagnosis"],
            },
        }

    def _notify(
        self,
        *,
        notifier: Any | None,
        symbol: str,
        dataframe: DataFrame,
        price_action_features: PriceActionFeatureResult,
        diagnosis: dict[str, Any],
        selected: list[dict[str, Any]],
        decision: dict[str, Any],
        validation: dict[str, Any],
        chart_generator: Callable | None,
    ) -> None:
        if notifier is None or not hasattr(notifier, "notify_decision"):
            return
        notifier.notify_decision(
            symbol,
            dataframe,
            {
                "price_action_features": price_action_features.as_dict(),
                "diagnosis": diagnosis,
                "selected_strategies": selected,
                "decision": decision,
                "validation": validation,
            },
            chart_generator=chart_generator,
        )


def _usage_total_from_responses(raw_responses: dict[str, Any]) -> dict[str, Any]:
    total = {field: 0 for field in USAGE_FIELDS}
    has_usage = False

    def add_usage(raw: dict[str, Any]) -> None:
        nonlocal has_usage
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return
        prompt_details = usage.get("prompt_tokens_details")
        if (
            isinstance(prompt_details, dict)
            and "cached_prompt_tokens" not in usage
            and prompt_details.get("cached_tokens") is not None
        ):
            usage = {**usage, "cached_prompt_tokens": prompt_details.get("cached_tokens")}
        for field in USAGE_FIELDS:
            value = usage.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            total[field] += int(value)
            has_usage = True

    for raw in raw_responses.values():
        if not isinstance(raw, dict):
            continue
        for attempt in raw.get("retry_attempts") or []:
            if isinstance(attempt, dict):
                add_usage(attempt)
        add_usage(raw)
    return total if has_usage else {}


def _validation_retry_limit(config: dict[str, Any]) -> int:
    try:
        value = int(config.get("pa_validation_retry_max", 0))
    except (TypeError, ValueError):
        return 0
    return max(0, min(value, 3))


def _append_retry_feedback(
    messages: list[dict[str, str]],
    *,
    stage: str,
    errors: list[str],
) -> list[dict[str, str]]:
    feedback = (
        "上一次输出未通过程序校验。\n"
        f"stage={stage}\n"
        f"errors={json.dumps(errors, ensure_ascii=False)}\n\n"
        "请只修正 JSON 输出，不要解释，不要输出 Markdown。"
    )
    return [*messages, {"role": "user", "content": feedback}]


def _decision_stance(config: dict[str, Any]) -> str:
    value = str(config.get("pa_decision_stance") or "conservative").strip().lower()
    allowed = {"conservative", "balanced", "aggressive", "extreme_aggressive"}
    return value if value in allowed else "conservative"


def _use_incremental_stage1(
    config: dict[str, Any],
    previous: dict[str, Any] | None,
    new_bar_count: int | None,
) -> bool:
    if config.get("pa_incremental_stage1_enabled", True) is False:
        return False
    if not previous or new_bar_count is None or new_bar_count <= 0:
        return False
    try:
        max_new = int(config.get("pa_incremental_stage1_max_new_bars", 10))
    except (TypeError, ValueError):
        max_new = 10
    return max_new > 0 and new_bar_count <= max_new


def _new_bar_count_since_previous(
    features: PriceActionFeatureResult,
    previous: dict[str, Any] | None,
) -> int | None:
    if not previous:
        return None
    previous_time = _parse_iso_datetime(previous.get("candle_time"))
    if previous_time is None:
        return None
    count = 0
    for row in features.rows:
        row_time = _parse_iso_datetime(row.get("time"))
        if row_time is None:
            continue
        if row_time > previous_time:
            count += 1
    return count


def _parse_iso_datetime(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        from datetime import datetime

        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)
        return parsed
    except (TypeError, ValueError):
        return None


def _validation_exception(stage: str, errors: list[str]) -> dict[str, Any]:
    return {
        "type": "validation_error",
        "stage": stage,
        "message": ",".join(errors),
        "validation_errors": list(errors),
    }


def _exception_info(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "stage": stage,
        "message": str(exc),
    }
