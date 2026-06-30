"""Prompt assembly for the two semantic PA LLM calls."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .features import PriceActionFeatureResult
from .strategy_templates import StrategyTemplate


TEMPLATE_DIR = Path(__file__).with_name("prompt_templates")
MARKET_DIAGNOSIS_TEMPLATE_FILES = (
    "market_diagnosis_system.txt",
    "market_diagnosis_user.txt",
)
TRADE_DECISION_TEMPLATE_FILES = (
    "trade_decision_system.txt",
    "trade_decision_user.txt",
)


MARKET_DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "cycle_position": (
        "spike|micro_channel|tight_channel|normal_channel|broad_channel|"
        "trending_tr|trading_range|extreme_tr|unknown"
    ),
    "alternative_cycle_position": "string|null",
    "direction": "bullish|bearish|neutral",
    "diagnosis_confidence": "integer 0..100",
    "spike_stage": "active|ending|transitioning|null",
    "climax_risk": "none|warning|triggered|null",
    "market_phase": "stable|transitioning",
    "transition_risk": "high|medium|low|null",
    "detected_patterns": ["string"],
    "key_signals": ["string"],
    "htf_context": "string",
    "entry_setup": "string",
    "support_levels": ["price string"],
    "resistance_levels": ["price string"],
    "strategy_files_needed": ["string"],
    "risk_warning": "string",
    "bar_analysis": {
        "always_in": "long|short|neutral",
        "last_closed_bar": "K1",
        "bar_type": (
            "trend_bull|trend_bear|doji|inside|outside_bull|outside_bear|"
            "flat|other"
        ),
        "signal_bar": {
            "bar": "K reference|null",
            "quality": "strong|medium|weak|invalid",
            "pattern": "H1|H2|L1|L2|MTR|wedge|tr_boundary|breakout_pullback|none",
            "reason": "string",
        },
        "entry_setup_type": (
            "H1|H2|L1|L2|MTR|wedge|tr_boundary|breakout_pullback|none"
        ),
        "follow_through": "yes|no|pending|failed",
    },
    "bar_by_bar_summary": [
        {
            "bar": "K1",
            "role": "structure|signal|entry|confirmation|noise|trap|climax|test",
            "bar_type": (
                "trend_bull|trend_bear|doji|inside|outside_bull|outside_bear|"
                "flat|other"
            ),
            "context_effect": (
                "strengthens_bull|weakens_bull|strengthens_bear|weakens_bear|"
                "neutral|transition"
            ),
            "follow_through": "yes|no|pending|failed",
            "trapped_side": "bulls|bears|both|none|unknown",
            "reason": "string",
        }
    ],
    "gate_trace": [
        {
            "node_id": "1.1|1.2|1.3|2.1|2.2|2.3|2.4|2.5",
            "question": "string",
            "answer": "是|否|中性|等待|不适用",
            "reason": "string",
            "branch": "string|null",
            "section": "string",
            "bar_range": "K{older}-K{newer} or K1",
        }
    ],
    "gate_result": "proceed|wait|unknown",
}


TRADE_DECISION_SCHEMA: dict[str, Any] = {
    "stage": "trade_decision",
    "decision": {
        "type": "enter_long|enter_short|wait|avoid",
        "direction": "long|short|neutral",
        "order_type": "market|limit|stop|none",
        "entry": "number|null",
        "stop_loss": "number|null",
        "take_profit_1": "number|null",
        "take_profit_2": "number|null",
        "risk_reward": "number|null",
        "confidence": "0..1",
        "reason": "string",
    },
    "decision_trace": ["string"],
    "watch_points": ["string"],
    "invalidations": ["string"],
}


@dataclass(frozen=True)
class PromptTemplate:
    """Loaded prompt template with deterministic version metadata."""

    name: str
    content: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def render(self, **values: Any) -> str:
        return self.content.format(**values).strip()

    def as_metadata(self) -> dict[str, str]:
        return {
            "name": self.name,
            "sha256": self.sha256,
        }


class PromptAssembler:
    """Load text prompt templates and render PA_Agent-style LLM messages."""

    def __init__(self, template_dir: Path | None = None) -> None:
        self.template_dir = template_dir or TEMPLATE_DIR

    def build_market_diagnosis_messages(
        self,
        features: PriceActionFeatureResult,
    ) -> list[dict[str, str]]:
        system = self._load("market_diagnosis_system.txt").render()
        user = self._load("market_diagnosis_user.txt").render(
            schema_json=json.dumps(MARKET_DIAGNOSIS_SCHEMA, ensure_ascii=False, indent=2),
            symbol=features.symbol,
            market=features.market,
            timeframe=features.timeframe,
            candle_time=features.candle_time.isoformat(),
            kline_table=features.kline_table,
            feature_table=features.feature_table,
            market_features_text=features.market_features_text,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def build_trade_decision_messages(
        self,
        *,
        features: PriceActionFeatureResult,
        diagnosis: dict[str, Any],
        strategies: list[StrategyTemplate],
        experience_cases: list[dict[str, Any]] | None = None,
        previous_decision: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        system = self._load("trade_decision_system.txt").render()
        strategy_text = "\n\n".join(strategy.render() for strategy in strategies)
        user = self._load("trade_decision_user.txt").render(
            diagnosis_json=json.dumps(diagnosis, ensure_ascii=False, indent=2),
            strategy_text=strategy_text,
            kline_table=features.kline_table,
            latest_features_json=json.dumps(features.latest_features, ensure_ascii=False, indent=2),
            experience_text=json.dumps(experience_cases or [], ensure_ascii=False, indent=2),
            previous_text=json.dumps(previous_decision or {}, ensure_ascii=False, indent=2),
            schema_json=json.dumps(TRADE_DECISION_SCHEMA, ensure_ascii=False, indent=2),
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def metadata(self) -> dict[str, Any]:
        market_diagnosis = [
            self._load(name).as_metadata()
            for name in MARKET_DIAGNOSIS_TEMPLATE_FILES
        ]
        trade_decision = [
            self._load(name).as_metadata()
            for name in TRADE_DECISION_TEMPLATE_FILES
        ]
        return {
            "template_dir": str(self.template_dir),
            "market_diagnosis": market_diagnosis,
            "trade_decision": trade_decision,
        }

    def _load(self, name: str) -> PromptTemplate:
        path = self.template_dir / name
        content = path.read_text(encoding="utf-8")
        return PromptTemplate(name=name, content=content)


DEFAULT_ASSEMBLER = PromptAssembler()


def build_market_diagnosis_messages(features: PriceActionFeatureResult) -> list[dict[str, str]]:
    """Assemble the market-diagnosis prompt."""
    return DEFAULT_ASSEMBLER.build_market_diagnosis_messages(features)


def build_trade_decision_messages(
    *,
    features: PriceActionFeatureResult,
    diagnosis: dict[str, Any],
    strategies: list[StrategyTemplate],
    experience_cases: list[dict[str, Any]] | None = None,
    previous_decision: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Assemble the trade-decision prompt."""
    return DEFAULT_ASSEMBLER.build_trade_decision_messages(
        features=features,
        diagnosis=diagnosis,
        strategies=strategies,
        experience_cases=experience_cases,
        previous_decision=previous_decision,
    )


def prompt_template_metadata() -> dict[str, Any]:
    """Return version metadata for all prompt templates used by default."""
    return DEFAULT_ASSEMBLER.metadata()
