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
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_time_iso=features.candle_time.isoformat(),
                    previous_record=previous,
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
                errors = validate_market_diagnosis(
                    parsed,
                    feature_rows=feature_rows,
                    coherence_checks=bool(self.config.get("pa_coherence_checks", False)),
                )
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
                attempt=attempt + 1,
                retry_limit=retry_limit,
                previous_raw=raw_text,
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
        symbol: str = "",
        timeframe: str = "",
        candle_time_iso: str | None = None,
        previous_record: dict[str, Any] | None = None,
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
                # ── Decision continuity guard ────────────────────────────────
                # Mirrors PA_Agent stage2_normalizer.py:1717-1732: after
                # normalize (which already widened stop / coerced failed
                # metrics to 不下单), force a continuity violation
                # (same-structure flip in cooldown, or Always-In direction
                # breach under direction=neutral) to 不下单 too. This is the
                # post-processing backstop the port was missing (#4 决策连续性守卫).
                if previous_record is not None:
                    try:
                        from .decision_continuity import (
                            apply_continuity_guard,
                            build_continuity_context,
                        )

                        ctx = build_continuity_context(
                            feature_rows=feature_rows,
                            stage1_json=diagnosis,
                            symbol=symbol,
                            timeframe=timeframe,
                            candle_time_iso=candle_time_iso,
                            previous_record=previous_record,
                        )
                        parsed = apply_continuity_guard(parsed, ctx)
                    except Exception as exc:  # pragma: no cover - safety net
                        logger.warning("apply_continuity_guard failed: %s", exc)
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
                attempt=attempt + 1,
                retry_limit=retry_limit,
                previous_raw=raw_text,
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
    """总 retry 上限,默认 3,允许上调到 5。

    对齐上游 PA_Agent ``ValidationSettings.retry_max`` 的 ``default=3, le=5``。
    """
    try:
        value = int(config.get("pa_validation_retry_max", 3))
    except (TypeError, ValueError):
        return 3
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
    attempt: int = 1,
    retry_limit: int = 1,
    previous_raw: str | None = None,
) -> list[dict[str, str]]:
    """追加结构化重试反馈 + 上轮 assistant 输出回灌。

    相比早期"只丢一句 errors 列表"的实现,本版本:
    1. 把内部错误码翻译成 LLM 能理解的人话 + 可操作指引;
    2. 针对常见错误追加枚举提示(如 gate_trace answer 只能用 是/否/中性/等待/不适用);
    3. 列出禁止项,防止 LLM 为通过校验而乱改方向;
    4. 把上轮 assistant 输出回灌,让 LLM 看到自己写错了什么(对照修改)。

    参考 PA_Agent ``retry_feedback.py:58 build_retry_feedback``。
    """
    feedback = _build_retry_feedback(
        stage=stage,
        errors=errors,
        attempt=attempt,
        retry_limit=retry_limit,
    )
    # 上轮 assistant 输出回灌:让 LLM 看到自己上次写了什么(对照修改)。
    # PA_Agent validation_retry.py:197-200 的关键机制。
    new_messages = list(messages)
    if previous_raw and previous_raw.strip():
        new_messages = [
            *new_messages,
            {"role": "assistant", "content": previous_raw},
        ]
    return [*new_messages, {"role": "user", "content": feedback}]


# 错误类别中文标签。参考 PA_Agent retry_feedback.py:12 _CATEGORY_ZH。
_CATEGORY_LABEL: dict[str, str] = {
    "a": "JSON 语法错误",
    "b": "缺少必填字段",
    "c": "字段值/一致性不符合规则",
    "d": "未输出 JSON(正文为空或纯文字)",
    "e": "API 额度/限流",
}

# 错误码 → 中文人话翻译 + 可操作指引。
# 参考 PA_Agent validation_messages.py 的 _PREFIX_RULES,适配 freqtrade 错误码。
_ERROR_HINTS: dict[str, str] = {
    # ── market_diagnosis 阶段 ──
    "market_diagnosis_gate_trace_direction_branch_conflict":
        "gate_trace 中某条的 direction 字段与 branch 字段语义冲突。请检查每条 gate_trace,"
        "direction 取 bullish/bearish/neutral,branch 必须与 direction 自洽"
        "(如 direction=bullish 时 branch 应为 aligned/bullish,不能是 bearish/conflict)。",
    "market_diagnosis_gate_trace_direction_answer_conflict":
        "gate_trace 中某条的 direction 与 answer 冲突。direction=bullish 对应的 answer 应为「是/中性」,"
        "不能是「否」(除非 branch 显式标记 reversal)。",
    "market_diagnosis_gate_trace_cycle_branch_conflict":
        "gate_trace 中 cycle_position 与 branch 冲突。请检查 cycle_position=spike 时 "
        "branch 是否用了 not_spike 类分支。",
    "market_diagnosis_gate_trace_answer_invalid":
        "gate_trace[].answer 只能用 是/否/中性/等待/不适用。"
        "禁止写「同向/冲突/多头/空头/bullish/bearish」等词——方向信息写进 branch。",
    "market_diagnosis_gate_trace_bar_range_required":
        "每条 gate_trace 必须有 bar_range 字段,格式如 K1 或 K2-K1(范围)。",
    "market_diagnosis_gate_trace_bar_range_invalid":
        "gate_trace[].bar_range 必须引用当前 K 线帧内的 K 序号(如 K1、K2-K5),不能是 K0 或超出范围。",
    "market_diagnosis_bar_by_bar_summary_must_cover_k5_to_k1":
        "bar_by_bar_summary 必须包含 K5、K4、K3、K2、K1 这 5 根 K 线各自的总结条目,"
        "缺一不可。请检查是否漏了某根 K 线。",
    "market_diagnosis_bar_by_bar_bar_type_mismatch":
        "bar_by_bar_summary[].bar_type 与程序计算的 K 线几何不一致。"
        "bar_type 必须服从程序几何表(trend_bull/trend_bear/doji 等),不能凭主观判断。",
    "market_diagnosis_bar_by_bar_role_invalid":
        "bar_by_bar_summary[].role 只能用 "
        "structure/signal/entry/confirmation/noise/trap/climax/test。"
        "禁止写 support/resistance/continuation 或 detected_patterns 中的形态名。",
    "market_diagnosis_bar_by_bar_trapped_side_invalid":
        "bar_by_bar_summary[].trapped_side 只能用 bulls/bears/both/none/unknown。"
        "禁止 null/空值;无明确被套方向时写 none。",
    "market_diagnosis_bar_by_bar_context_effect_invalid":
        "bar_by_bar_summary[].context_effect 只能用 "
        "strengthens_bull/weakens_bull/strengthens_bear/weakens_bear/neutral/transition。"
        "注意 strengthens_bear 不要拼成 strengthens_bash。",
    "market_diagnosis_direction_invalid":
        "顶层 direction 只能用 bullish/bearish/neutral。",
    "market_diagnosis_cycle_position_invalid":
        "cycle_position 只能用 "
        "spike/micro_channel/tight_channel/normal_channel/broad_channel/"
        "trending_tr/trading_range/extreme_tr/unknown 之一。",
    # ── trade_decision 阶段 ──
    "risk_reward_below_minimum":
        "盈亏比(R/R)低于最低要求。请重新计算 entry/stop/tp1,"
        "确保 R/R >= 配置的最低值;若市场结构不支持合格 R/R,应改为不下单。",
    "trader_equation_fails":
        "交易者方程不成立(盈亏比 × 胜率 < 1)。请重算或下调 trade_confidence,"
        "若无法满足应改为不下单。",
    "order_direction_conflicts_with_stage1_direction_without_node_2_3":
        "下单方向与 stage1 的 direction 冲突,且没有 node 2.3 的覆盖理由。"
        "若要反向下单,必须在 decision_trace 中加 node 2.3 说明反向依据。",
    "long_stop_must_be_below_entry": "做多止损价必须低于入场价。",
    "long_tp1_must_be_above_entry": "做多止盈1必须高于入场价。",
    "long_tp2_must_be_above_tp1": "做多止盈2必须高于止盈1。",
    "short_stop_must_be_above_entry": "做空止损价必须高于入场价。",
    "short_tp1_must_be_below_entry": "做空止盈1必须低于入场价。",
    "short_tp2_must_be_below_tp1": "做空止盈2必须低于止盈1。",
    "decision_trace_bar_range_invalid":
        "decision_trace[].bar_range 必须引用当前 K 线帧内的 K 序号(如 K1、K2-K5),格式正确。",
    "decision_trace_answer_invalid":
        "decision_trace[].answer 只能用 是/否/中性/等待/不适用。",
}


def _build_retry_feedback(
    *,
    stage: str,
    errors: list[str],
    attempt: int,
    retry_limit: int,
) -> str:
    """生成结构化、可操作的 LLM 重试反馈。

    参考 PA_Agent ``retry_feedback.py:58 build_retry_feedback``。
    """
    category = _categorize_errors(errors)
    stage_zh = "市场诊断(stage1)" if "diagnosis" in stage else "交易决策(stage2)"

    lines = [
        f"## 校验未通过(第 {attempt}/{retry_limit} 次重试)",
        "",
        f"阶段:**{stage_zh}**",
        f"失败类型:**{_CATEGORY_LABEL.get(category, category)}** (category={category})",
        "",
        "**必须修正(仅修下列项;其余字段保持与上一轮一致):**",
    ]

    # 列出具体错误 + 人话翻译
    shown = 0
    for err in errors[:8]:
        hint = _lookup_hint(err)
        lines.append(f"{shown + 1}. {hint}")
        shown += 1
    if len(errors) > 8:
        lines.append(f"…另有 {len(errors) - 8} 条错误")

    # 针对性枚举提示
    err_blob = " ".join(errors)
    if "gate_trace" in err_blob and "answer" in err_blob:
        lines.append("")
        lines.append("**gate_trace answer 枚举提示:**")
        lines.append(
            "- answer 只能用 **是/否/中性/等待/不适用**;"
            "「同向/冲突/背景中性」写在 branch(aligned/conflict/neutral_background),"
            "**禁止**把「冲突」写在 answer。"
        )
    if "bar_by_bar" in err_blob and ("role" in err_blob or "trapped_side" in err_blob):
        lines.append("")
        lines.append("**bar_by_bar_summary 枚举提示:**")
        lines.append(
            "- role 只用 structure/signal/entry/confirmation/noise/trap/climax/test;"
            "- trapped_side 只用 bulls/bears/both/none/unknown(无被套方向写 none,禁止 null);"
            "- context_effect 只用 strengthens_bull/weakens_bull/strengthens_bear/weakens_bear/neutral/transition。"
        )

    # 禁止项:防止 LLM 为通过校验而乱改方向/反转交易结论
    lines.append("")
    lines.append("**禁止为通过校验而修改:**")
    if "diagnosis" in stage:
        forbidden = (
            "顶层 direction / cycle_position(除非有明确 K 线依据);",
            "gate_trace[].answer 之外的字段(除非反馈明确要求);",
            "程序锁定的 K 线几何 bar_type(必须服从程序几何表)。",
        )
    else:
        forbidden = (
            "diagnosis_summary.cycle_position / direction(除非反馈明确要求);",
            "把 order_type 从「不下单」改成下单(或反之)仅为通过校验;",
            "交易者方程的数值结论(须基于真实 entry/stop/target 重算)。",
        )
    for item in forbidden:
        lines.append(f"- {item}")

    lines.append("")
    lines.append(
        f"请根据以上说明,在 assistant 正文输出**完整**{stage_zh}裸 JSON(不要 markdown 围栏)。"
        "交易结论须与 K 线分析一致,不得仅为修字段而反转方向。"
    )
    return "\n".join(lines)


def _lookup_hint(error_code: str) -> str:
    """把单条错误码翻译成人话 + 可操作指引。"""
    code = str(error_code).strip()
    # 精确匹配
    if code in _ERROR_HINTS:
        return f"[{code}] {_ERROR_HINTS[code]}"
    # 缺失字段类:market_diagnosis_missing_<field> / decision_trace_<field>_required
    if "_missing_" in code or code.endswith("_required"):
        field = code.split("_missing_")[-1] if "_missing_" in code else code.replace("_required", "")
        return f"[{code}] 缺少必填字段「{field}」,请补上。"
    # JSON 语法类
    if code.startswith("invalid_json:") or "JSONDecodeError" in code:
        return f"[{code}] JSON 语法错误。请确保输出是合法 JSON:引号成对、逗号正确、无多余字符。"
    if code == "json_root_must_be_object":
        return f"[{code}] JSON 根必须是对象 {{...}},不能是数组或纯文本。"
    # 兜底:原样展示
    return f"[{code}]"


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
