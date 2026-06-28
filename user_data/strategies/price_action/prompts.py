"""Prompt assembly for the two semantic PA LLM calls."""
from __future__ import annotations

import json
from typing import Any

from .features import L1FeatureResult
from .strategy_templates import StrategyTemplate


MARKET_DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "stage": "market_diagnosis",
    "market_state": {
        "cycle": "trend|trading_range|breakout|reversal|unknown",
        "direction": "long|short|neutral",
        "strength": "0..1",
        "timeframe": "string",
    },
    "gate": {
        "upper": "number|null",
        "lower": "number|null",
        "position": "above|below|inside|testing_upper|testing_lower|unknown",
        "breakout": "up|down|both|failed_up|failed_down|none",
    },
    "bar_summaries": [
        {"k": "K1", "role": "signal|follow_through|pullback|context", "semantics": "string"}
    ],
    "signal_chain": {
        "direction": "long|short|neutral",
        "patterns": ["inside", "ii", "gate_break_up"],
        "quality": "strong|normal|weak|none",
        "setup": "string",
    },
    "risks": ["string"],
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
        "你是 Al Brooks 价格行为分析助手。只基于用户提供的已收盘 K 线和 L1 特征判断，"
        "不得编造外部行情。你必须只输出一个 JSON object，不要 Markdown。"
    )
    user = f"""
任务：完成 L2 市场诊断。

约束：
- K1 是最新已收盘 K，未收盘 K 不在表内。
- 先判断市场处于趋势、震荡、突破、反转还是未知。
- 闸门使用 L1 表中的 gate_high/gate_low/gate_break/gate_pos。
- 逐 K 摘要聚焦最近 5 根，信号链初判必须说明方向、形态与质量。
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
1. L2 市场诊断 JSON
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
- 若 ATR_x / atr_expand_ratio > 2 且正在突破闸门，必须视作假突破风险，不能直接追价。
- 入场、止损、止盈必须与方向一致：多单止损低于入场，止盈高于入场；空单反之。
- 不满足策略模板信号链时输出 wait 或 avoid。
- 输出必须符合这个 JSON contract：
{json.dumps(TRADE_DECISION_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
