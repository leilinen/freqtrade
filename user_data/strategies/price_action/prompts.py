"""Prompt assembly for the two semantic PA LLM calls."""
from __future__ import annotations

import json
from typing import Any

from .features import L1FeatureResult
from .strategy_templates import StrategyTemplate


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
            "node_id": "1.2|1.3|2.1|2.2|2.5",
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


def build_market_diagnosis_messages(l1: L1FeatureResult) -> list[dict[str, str]]:
    """Assemble the market-diagnosis prompt."""
    system = (
        "你是 Al Brooks 价格行为分析助手。"
        "阶段一只负责市场诊断与闸门判断，"
        "不评估具体下单、止损、止盈或仓位。"
        "只基于用户提供的已收盘 K 线和"
        "程序特征判断，不得编造外部行情。你必须只输出一个 JSON object，"
        "不要 Markdown。"
    )
    user = f"""
任务：完成阶段一 Stage 1 市场诊断。

约束：
- K1 是最新已收盘 K，未收盘 K 不在表内。
- 输出只描述市场周期、方向、结构、信号质量、支撑阻力与闸门结论。
- 禁止在阶段一给入场、止损、止盈或仓位建议。
- `cycle_position` 必须在 PA_Agent 周期枚举中选择，
  不要使用 trend/breakout/reversal。
- `direction` 只能是 bullish、bearish 或 neutral。
- `diagnosis_confidence` 必须是 0-100 的整数，不能写 high/medium/low。
- `support_levels` 只填当前价格下方支撑，
  `resistance_levels` 只填当前价格上方阻力。
- `bar_by_bar_summary` 分析窗口>=5根时必须恰好 5 条，覆盖 K5-K1。
- `bar_by_bar_summary[].bar_type` 必须照抄特征汇总表中的 `bar_type`，
  禁止自行改写。
- `gate_trace` 按二元决策树前半段输出；gate_result=proceed 时必须包含
  1.2、1.3、2.1、2.2、2.5 五个节点。
- `gate_result=wait/unknown` 只允许出现在 1.2 无法识别周期或 1.3 极端混乱时。
- 2.1/2.5 为否或中性不代表阶段一阻断，通常仍应 gate_result=proceed。
- 每条 gate_trace 必须有 `bar_range`，只能引用 K1 到当前表内最大 K，禁止 K0。
- 输出必须符合这个 JSON contract：
{json.dumps(MARKET_DIAGNOSIS_SCHEMA, ensure_ascii=False, indent=2)}

标的：{l1.symbol}
市场：{l1.market}
周期：{l1.timeframe}
最新已收盘 K：{l1.candle_time.isoformat()}

K线文本表：
{l1.kline_table}

特征汇总表：
{l1.feature_table}

市场结构辅助特征：
{l1.market_features_text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_trade_decision_messages(
    *,
    l1: L1FeatureResult,
    diagnosis: dict[str, Any],
    strategies: list[StrategyTemplate],
    experience_cases: list[dict[str, Any]] | None = None,
    previous_decision: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Assemble the trade-decision prompt."""
    system = (
        "你是价格行为交易决策助手。你只做监控决策，不代表真实下单。"
        "必须返回裸 JSON object，禁止 Markdown、解释性前后缀。"
    )
    strategy_text = "\n\n".join(strategy.render() for strategy in strategies)
    experience_text = json.dumps(experience_cases or [], ensure_ascii=False, indent=2)
    previous_text = json.dumps(previous_decision or {}, ensure_ascii=False, indent=2)
    user = f"""
任务：完成 L4 交易决策。

输入：
1. 阶段一市场诊断 JSON
{json.dumps(diagnosis, ensure_ascii=False, indent=2)}

2. L3 本地路由策略模板
{strategy_text}

3. 最新 K 线表
{l1.kline_table}

4. L1 最新 K 特征
{json.dumps(l1.latest_features, ensure_ascii=False, indent=2)}

5. 可选经验库案例
{experience_text}

6. 上一轮成功决策 trace（用于增量延续，不存在则为空）
{previous_text}

硬性约束：
- 若 ATR_x / atr_expand_ratio > 2 且正在突破闸门，
  必须视作假突破风险，不能直接追价。
- 入场、止损、止盈必须与方向一致：
  多单止损低于入场，止盈高于入场；空单反之。
- 不满足策略模板信号链时输出 wait 或 avoid。
- 输出必须符合这个 JSON contract：
{json.dumps(TRADE_DECISION_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
