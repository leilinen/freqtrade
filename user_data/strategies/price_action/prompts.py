"""Prompt assembly for the two semantic PA LLM calls."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .features import PriceActionFeatureResult
from .price_tick import format_breakout_tick_hint
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
DECISION_STANCE_TEXT = {
    "conservative": (
        "保守：只接受清晰信号、明确止损和通过交易者方程的机会。"
    ),
    "balanced": "平衡：接受结构清晰且风险收益合理的标准 PA 机会。",
    "aggressive": (
        "积极：可接受较早计划型入场，"
        "但仍必须满足止损、目标和交易者方程。"
    ),
    "extreme_aggressive": (
        "极积极：可评估高波动早期机会，但禁止跳过风险收益校验。"
    ),
}


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
    "decision": {
        "order_direction": "做多|做空|null",
        "order_type": "限价单|突破单|市价单|不下单",
        "entry_price": "number|null",
        "entry_basis_bar": "K reference|null",
        "entry_basis_extreme": "high|low|null",
        "entry_rule": "string|null",
        "take_profit_price": "number|null",
        "take_profit_price_2": "number|null",
        "stop_loss_price": "number|null",
        "reasoning": "string",
        "diagnosis_confidence": "integer 0..100",
        "diagnosis_confidence_reasoning": "string",
        "trade_confidence": "integer 0..100",
        "trade_confidence_reasoning": "string",
        "estimated_win_rate": "integer 0..100|null",
        "estimated_win_rate_reasoning": "string|null",
        "key_factors": ["string"],
        "watch_points": ["string"],
        "risk_assessment": "string",
        "invalidation_condition": "string|null",
    },
    "diagnosis_summary": {
        "cycle_position": "string",
        "direction": "bullish|bearish|neutral",
        "key_signals": ["string"],
    },
    "decision_trace": [
        {
            "node_id": "3.x-11.x or 14.x",
            "question": "string",
            "answer": "是|否|中性|等待|不适用",
            "reason": "string",
            "branch": "string|null",
            "section": "string",
            "bar_range": "K{older}-K{newer} or K1",
        }
    ],
    "terminal": {
        "node_id": "string",
        "outcome": "wait|reject|trade|proceed",
        "label": "string",
    },
    "next_cycle_prediction": {
        "cycle": "cycle_position|null",
        "direction": "bullish|bearish|neutral|null",
        "probabilities": "object|null",
        "reasoning": "string",
        "unpredictable": "boolean",
        "features_used": ["stage1_diagnosis|kline_features|experience_library|stage2_decision"],
    },
    "next_bar_prediction": {
        "direction": "bullish|bearish|neutral|null",
        "probabilities": "object|null",
        "reasoning": "string",
        "unpredictable": "boolean",
        "features_used": ["stage1_diagnosis|kline_features|stage2_decision"],
    },
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

    def build_incremental_market_diagnosis_messages(
        self,
        features: PriceActionFeatureResult,
        *,
        previous_analysis: dict[str, Any],
        new_bar_count: int,
    ) -> list[dict[str, str]]:
        """Build Stage 1 as a PA_Agent-style continuation update."""
        system = self._load("market_diagnosis_system.txt").render()
        previous_messages = previous_analysis.get("market_diagnosis_messages") or []
        previous_user = ""
        for message in previous_messages:
            if isinstance(message, dict) and message.get("role") == "user":
                previous_user = str(message.get("content") or "")
                break
        previous_raw = previous_analysis.get("raw_responses") or {}
        previous_stage1 = previous_raw.get("market_diagnosis") or {}
        previous_content = ""
        if isinstance(previous_stage1, dict):
            previous_content = str(previous_stage1.get("content") or "")
        if not previous_content:
            previous_content = json.dumps(
                previous_analysis.get("diagnosis") or {},
                ensure_ascii=False,
                indent=2,
            )
        incremental_user = (
            "任务：增量更新阶段一市场诊断。\n\n"
            f"上一轮成功记录到当前共有 {new_bar_count} 根新增已收盘 K 线。\n"
            "请沿用上一轮阶段一诊断作为上下文，"
            "但必须以当前最新闭合 K 线事实为准；"
            "如果市场周期、方向或闸门结果发生变化，"
            "必须在 gate_trace 中说明依据。\n\n"
            "当前最新 K 线表：\n"
            f"{features.kline_table}\n\n"
            "当前几何特征表：\n"
            f"{features.feature_table}\n\n"
            "当前市场结构辅助特征：\n"
            f"{features.market_features_text}\n\n"
            "输出必须仍符合这个 JSON contract：\n"
            f"{json.dumps(MARKET_DIAGNOSIS_SCHEMA, ensure_ascii=False, indent=2)}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": previous_user or "上一轮阶段一 Prompt 不可用。"},
            {"role": "assistant", "content": previous_content},
            {"role": "user", "content": incremental_user},
        ]

    def build_trade_decision_messages(
        self,
        *,
        features: PriceActionFeatureResult,
        diagnosis: dict[str, Any],
        strategies: list[StrategyTemplate],
        experience_cases: list[dict[str, Any]] | None = None,
        previous_decision: dict[str, Any] | None = None,
        decision_stance: str = "conservative",
    ) -> list[dict[str, str]]:
        system = self._load("trade_decision_system.txt").render()
        strategy_text = "\n\n".join(strategy.render() for strategy in strategies)
        stance_key = str(decision_stance or "conservative").strip().lower()
        stance_text = DECISION_STANCE_TEXT.get(
            stance_key,
            DECISION_STANCE_TEXT["conservative"],
        )
        user = self._load("trade_decision_user.txt").render(
            diagnosis_json=json.dumps(diagnosis, ensure_ascii=False, indent=2),
            strategy_text=strategy_text,
            kline_table=features.kline_table,
            latest_features_json=json.dumps(features.latest_features, ensure_ascii=False, indent=2),
            experience_text=json.dumps(experience_cases or [], ensure_ascii=False, indent=2),
            previous_text=json.dumps(previous_decision or {}, ensure_ascii=False, indent=2),
            decision_stance=stance_key,
            decision_stance_text=stance_text,
            breakout_tick_hint=format_breakout_tick_hint(features.rows),
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


def build_incremental_market_diagnosis_messages(
    features: PriceActionFeatureResult,
    *,
    previous_analysis: dict[str, Any],
    new_bar_count: int,
) -> list[dict[str, str]]:
    """Assemble a continuation-style incremental market-diagnosis prompt."""
    return DEFAULT_ASSEMBLER.build_incremental_market_diagnosis_messages(
        features,
        previous_analysis=previous_analysis,
        new_bar_count=new_bar_count,
    )


def build_trade_decision_messages(
    *,
    features: PriceActionFeatureResult,
    diagnosis: dict[str, Any],
    strategies: list[StrategyTemplate],
    experience_cases: list[dict[str, Any]] | None = None,
    previous_decision: dict[str, Any] | None = None,
    decision_stance: str = "conservative",
) -> list[dict[str, str]]:
    """Assemble the trade-decision prompt."""
    return DEFAULT_ASSEMBLER.build_trade_decision_messages(
        features=features,
        diagnosis=diagnosis,
        strategies=strategies,
        experience_cases=experience_cases,
        previous_decision=previous_decision,
        decision_stance=decision_stance,
    )


def prompt_template_metadata() -> dict[str, Any]:
    """Return version metadata for all prompt templates used by default."""
    return DEFAULT_ASSEMBLER.metadata()
