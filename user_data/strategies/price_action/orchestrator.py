"""L1-L4 PA_Agent-style analysis orchestration."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Callable

from pandas import DataFrame

from .experience import retrieve_experience_cases
from .features import L1FeatureResult, build_l1_features
from .llm import OpenAIJsonClient
from .prompts import build_market_diagnosis_messages, build_trade_decision_messages
from .repository import PriceActionRepository
from .router import route_strategies
from .validation import DecisionValidator, parse_json_object, validate_stage1_diagnosis


logger = logging.getLogger(__name__)


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
    """Run L1 features, Stage 1 diagnosis, L3 routing, and L4 decision."""

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
        l1: L1FeatureResult | None = None
        raw_market_diagnosis: str | None = None
        raw_trade_decision: str | None = None
        diagnosis: dict[str, Any] | None = None
        decision_json: dict[str, Any] | None = None
        validation_status: str | None = None
        validation_errors: list[str] = []
        selected = []
        experience_cases: list[dict[str, Any]] = []
        prompt_metadata: dict[str, Any] = {
            "model": self.llm_client.model,
            "base_url": self.llm_client.base_url,
            "response_format": {"type": "json_object"},
        }

        try:
            l1 = build_l1_features(
                dataframe,
                symbol=symbol,
                timeframe=timeframe,
                market=market,
                window=int(self.config.get("pa_llm_window", 30)),
                warmup=int(self.config.get("pa_llm_warmup", 50)),
            )

            market_diagnosis_messages = build_market_diagnosis_messages(l1)
            raw_market_diagnosis = self.llm_client.complete_json(
                market_diagnosis_messages,
                stage="market_diagnosis",
            )
            diagnosis = parse_json_object(raw_market_diagnosis)
            stage1_errors = validate_stage1_diagnosis(diagnosis, l1_rows=l1.rows)
            if stage1_errors:
                raise ValueError(f"stage1_invalid:{','.join(stage1_errors)}")

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
                decision_json = self._build_gate_wait_decision(diagnosis)
                raw_trade_decision = json.dumps(decision_json, ensure_ascii=False)
                decision_json, validation = self.validator.validate(
                    raw_trade_decision,
                    diagnosis=diagnosis,
                    l1_features=l1.latest_features,
                    strategies=selected,
                )
                validation_status = validation.status
                validation_errors = validation.errors
                status = "success" if validation.valid else "invalid"

                self._save(
                    l1=l1,
                    status=status,
                    diagnosis=diagnosis,
                    selected=selected,
                    experience_cases=experience_cases,
                    decision=decision_json,
                    validation_status=validation_status,
                    validation_errors=validation_errors,
                    prompt_metadata={**prompt_metadata, "validation_checks": validation.checks},
                    raw_responses={
                        "market_diagnosis": raw_market_diagnosis,
                        "trade_decision": raw_trade_decision,
                    },
                )

                if validation.valid and self._should_notify(decision_json):
                    self._notify(
                        notifier=notifier,
                        symbol=symbol,
                        dataframe=dataframe,
                        l1=l1,
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
                    candle_time=l1.candle_time,
                    decision=decision_json,
                    errors=validation_errors,
                )

            previous = (
                self.repository.get_previous_successful_analysis(
                    symbol=symbol,
                    timeframe=timeframe,
                    before_time=l1.candle_time,
                )
                if self.repository
                else None
            )

            trade_decision_messages = build_trade_decision_messages(
                l1=l1,
                diagnosis=diagnosis,
                strategies=strategies,
                experience_cases=experience_cases,
                previous_decision=previous,
            )
            raw_trade_decision = self.llm_client.complete_json(
                trade_decision_messages,
                stage="trade_decision",
            )
            decision_json, validation = self.validator.validate(
                raw_trade_decision,
                diagnosis=diagnosis,
                l1_features=l1.latest_features,
                strategies=selected,
            )
            validation_status = validation.status
            validation_errors = validation.errors
            status = "success" if validation.valid else "invalid"

            self._save(
                l1=l1,
                status=status,
                diagnosis=diagnosis,
                selected=selected,
                experience_cases=experience_cases,
                decision=decision_json,
                validation_status=validation_status,
                validation_errors=validation_errors,
                prompt_metadata={**prompt_metadata, "validation_checks": validation.checks},
                raw_responses={
                    "market_diagnosis": raw_market_diagnosis,
                    "trade_decision": raw_trade_decision,
                },
            )

            if validation.valid and self._should_notify(decision_json):
                self._notify(
                    notifier=notifier,
                    symbol=symbol,
                    dataframe=dataframe,
                    l1=l1,
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
                candle_time=l1.candle_time,
                decision=decision_json,
                errors=validation_errors,
            )
        except Exception as exc:
            logger.warning(
                "PA analysis failed for %s %s: %s",
                symbol,
                timeframe,
                exc,
                exc_info=True,
            )
            validation_errors = [f"{type(exc).__name__}:{exc}"]
            if l1 is not None:
                self._save(
                    l1=l1,
                    status="failed",
                    diagnosis=diagnosis,
                    selected=selected,
                    experience_cases=experience_cases,
                    decision=decision_json,
                    validation_status=validation_status or "failed",
                    validation_errors=validation_errors,
                    prompt_metadata=prompt_metadata,
                    raw_responses={
                        "market_diagnosis": raw_market_diagnosis,
                        "trade_decision": raw_trade_decision,
                    },
                )
            return AnalysisOutcome(
                status="failed",
                symbol=symbol,
                timeframe=timeframe,
                candle_time=l1.candle_time if l1 else None,
                decision=decision_json,
                errors=validation_errors,
            )

    def _save(
        self,
        *,
        l1: L1FeatureResult,
        status: str,
        diagnosis: dict[str, Any] | None,
        selected: list[dict[str, Any]],
        experience_cases: list[dict[str, Any]],
        decision: dict[str, Any] | None,
        validation_status: str | None,
        validation_errors: list[str],
        prompt_metadata: dict[str, Any],
        raw_responses: dict[str, Any],
    ) -> None:
        if not self.repository:
            return
        self.repository.save_analysis(
            market=l1.market,
            symbol=l1.symbol,
            timeframe=l1.timeframe,
            candle_time=l1.candle_time,
            status=status,
            kline_table=l1.kline_table,
            feature_table=l1.feature_table,
            l1_features=l1.as_dict(),
            market_diagnosis=diagnosis,
            selected_strategies=selected,
            experience_cases=experience_cases,
            trade_decision=decision,
            validation_status=validation_status,
            validation_errors=validation_errors,
            prompt_metadata=prompt_metadata,
            raw_responses=raw_responses,
        )

    def _should_notify(self, decision_json: dict[str, Any] | None) -> bool:
        if not decision_json:
            return False
        decision_type = str((decision_json.get("decision") or {}).get("type", "")).lower()
        if decision_type in ("enter_long", "enter_short"):
            return True
        return bool(self.config.get("pa_notify_wait", False))

    def _build_gate_wait_decision(self, diagnosis: dict[str, Any]) -> dict[str, Any]:
        gate_result = str(diagnosis.get("gate_result", "wait")).lower()
        trace = diagnosis.get("gate_trace") or []
        final_reason = ""
        if trace and isinstance(trace[-1], dict):
            final_reason = str(trace[-1].get("reason") or "")
        reason = final_reason or str(diagnosis.get("risk_warning") or "阶段一闸门未通过")
        confidence = diagnosis.get("diagnosis_confidence")
        try:
            confidence_value = max(0.0, min(float(confidence) / 100.0, 1.0))
        except (TypeError, ValueError):
            confidence_value = 0.0
        return {
            "stage": "trade_decision",
            "decision": {
                "type": "wait",
                "direction": "neutral",
                "order_type": "none",
                "entry": None,
                "stop_loss": None,
                "take_profit_1": None,
                "take_profit_2": None,
                "risk_reward": None,
                "confidence": confidence_value,
                "reason": f"阶段一 gate_result={gate_result}，跳过阶段二：{reason}",
            },
            "decision_trace": [
                f"stage1_gate_result={gate_result}",
                "stage2_model_call=skipped",
            ],
            "watch_points": [reason],
            "invalidations": [],
        }

    def _notify(
        self,
        *,
        notifier: Any | None,
        symbol: str,
        dataframe: DataFrame,
        l1: L1FeatureResult,
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
                "l1": l1.as_dict(),
                "diagnosis": diagnosis,
                "selected_strategies": selected,
                "decision": decision,
                "validation": validation,
            },
            chart_generator=chart_generator,
        )
