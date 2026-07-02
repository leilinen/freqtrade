"""PA_Agent-style price-action analysis orchestration."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Callable

from pandas import DataFrame

from .experience import retrieve_experience_cases
from .features import PriceActionFeatureResult, build_price_action_features
from .llm import OpenAIJsonClient
from .normalize import normalize_market_diagnosis, normalize_trade_decision
from .prompts import (
    build_incremental_market_diagnosis_messages,
    build_market_diagnosis_messages,
    build_trade_decision_messages,
    prompt_template_metadata,
)
from .repository import PriceActionRepository
from .router import route_strategies
from .validation import (
    CHECK_JSON_SYNTAX,
    DecisionValidator,
    ValidationResult,
    parse_json_object,
    validate_market_diagnosis,
)


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
                window=_config_int(self.config, "pa_llm_window", "PA_LLM_WINDOW", 30),
                warmup=_config_int(self.config, "pa_llm_warmup", "PA_LLM_WARMUP", 50),
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
                        experience_cases=experience_cases,
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
                    experience_cases=experience_cases,
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
        # retry_limit 改为按错误类别动态计算:首轮用 base 上限兜底,
        # 首次失败后根据具体 errors 精修。
        failed_attempts: list[dict[str, Any]] = []
        current_messages = list(messages)
        last_errors: list[str] = []
        attempt = 0

        while True:
            raw_text = self.llm_client.complete_json(
                current_messages,
                stage="market_diagnosis",
            )
            raw_record = self._capture_llm_response("market_diagnosis", raw_text)
            try:
                parsed = parse_json_object(raw_text)
                parsed = normalize_market_diagnosis(parsed, feature_rows=feature_rows)
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
            retry_limit = _retry_limit_for_errors(errors, self.config)
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
            attempt += 1

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
        failed_attempts: list[dict[str, Any]] = []
        current_messages = list(messages)
        attempt = 0

        while True:
            raw_text = self.llm_client.complete_json(
                current_messages,
                stage="trade_decision",
            )
            raw_record = self._capture_llm_response("trade_decision", raw_text)
            try:
                parsed = parse_json_object(raw_text)
                parsed = normalize_trade_decision(
                    parsed,
                    diagnosis=diagnosis,
                    feature_rows=feature_rows,
                )
            except json.JSONDecodeError as exc:
                parsed = None
                validation = ValidationResult(
                    False, [CHECK_JSON_SYNTAX], [f"invalid_json:{exc.msg}"]
                )
                decision_json = None
            except ValueError:
                parsed = None
                validation = ValidationResult(
                    False, [CHECK_JSON_SYNTAX], ["json_root_must_be_object"]
                )
                decision_json = None
            else:
                decision_json, validation = self.validator.validate_parsed(
                    parsed,
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
            retry_limit = _retry_limit_for_errors(validation.errors, self.config)
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
            attempt += 1

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
        experience_cases: list[dict[str, Any]],
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
                "experience_cases": experience_cases,
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
    """总 retry 上限,默认 0,允许上调到 5。

    上调到 5 对应上游 PA_Agent ``ValidationSettings.retry_max`` 的 ``le=5`` 上限。
    """
    try:
        value = int(config.get("pa_validation_retry_max", 0))
    except (TypeError, ValueError):
        return 0
    return max(0, min(value, 5))


def _semantic_retry_limit(config: dict[str, Any]) -> int:
    """schema/语义类错误的 retry 上限(上游 retry_max_semantic=1)。

    这类错误 retry 成功率低,默认 1 次足够,避免浪费 token。
    """
    try:
        value = int(config.get("pa_validation_retry_semantic_max", 1))
    except (TypeError, ValueError):
        return 1
    return max(0, min(value, 3))


def _categorize_errors(errors: list[str]) -> str:
    """把 validate_market_diagnosis 返回的错误列表归到 a/b/c/d/e 类别之一。

    参考 PA_Agent ``json_validator.py:522-537, 687-696`` 和
    ``retry_policy.py:39-50`` 的 retry 上限映射:

    - ``a`` JSON 语法错误:retry_max=base(默认 3)
    - ``b`` 缺失必填字段:retry_max=base
    - ``c`` schema/语义冲突:retry_max=min(base, semantic=1)
    - ``d`` 非 JSON 纯文本:retry_max=base
    - ``e`` provider quota/rate limit:不 retry(0)
    """
    if not errors:
        return "ok"
    if any("quota" in e.lower() or "rate_limit" in e.lower() for e in errors):
        return "e"
    # a: parse 阶段抛出的 JSONDecodeError(被 orchestrator 包装成 ValueError)
    if any(e.startswith("ValueError") and "JSONDecodeError" in e for e in errors):
        return "a"
    if any("invalid_json" in e for e in errors):
        return "a"
    # d: 非 JSON 纯文本(parse 阶段拿到空内容、纯文本)
    if any("Expecting value" in e for e in errors):
        return "d"
    # b: 缺失必填字段
    if any("required" in e.lower() or "missing" in e.lower() for e in errors):
        return "b"
    # c: 其他 schema/语义冲突
    return "c"


def _retry_limit_for_errors(errors: list[str], config: dict[str, Any]) -> int:
    """根据错误类别返回 retry 上限。"""
    cat = _categorize_errors(errors)
    if cat in ("ok", "e"):
        return 0
    base = _validation_retry_limit(config)
    if cat == "c":
        return min(base, _semantic_retry_limit(config))
    return base


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


def _config_int(
    config: dict[str, Any],
    key: str,
    env_key: str,
    default: int,
) -> int:
    value = config.get(key)
    if value is None or value == "":
        value = os.environ.get(env_key)
    if value is None or value == "":
        return default
    return int(value)


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
