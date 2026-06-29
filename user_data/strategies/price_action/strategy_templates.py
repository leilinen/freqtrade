"""Deterministic strategy templates for PA routing."""
from __future__ import annotations

from dataclasses import dataclass


ATR_VETO = "ATR 突变否决：若最新 K 的 ATR_x/atr_expand_ratio > 2 且突破闸门，按假突破处理，禁止追价。"


@dataclass(frozen=True)
class StrategyTemplate:
    """Local strategy template routed from market diagnosis."""

    template_id: str
    name: str
    cycle: str
    direction: str
    signal_chain: str
    risk_reward: str
    order_plan: str
    veto: str = ATR_VETO

    def render(self) -> str:
        return (
            f"### {self.template_id} - {self.name}\n"
            f"§信号链\n{self.signal_chain}\n"
            f"§风险收益\n{self.risk_reward}\n"
            f"§下单方式\n{self.order_plan}\n"
            f"§否决条件\n{self.veto}"
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "cycle": self.cycle,
            "direction": self.direction,
            "signal_chain": self.signal_chain,
            "risk_reward": self.risk_reward,
            "order_plan": self.order_plan,
            "veto": self.veto,
        }


TEMPLATES: dict[str, StrategyTemplate] = {
    "trend_pullback_long": StrategyTemplate(
        "trend_pullback_long",
        "趋势回调做多",
        "trend",
        "long",
        "背景多头；K1/K2 回踩 EMA20 或前闸门后出现强收盘、H1/H2、ii 上破或外包反转。",
        "止损放在信号 K 或回调低点下方；目标至少 2R，第一目标可取前高/闸门上沿。",
        "优先 stop-buy 突破信号 K 高点；若已突破但距离止损过大，等待二次回踩。",
    ),
    "trend_pullback_short": StrategyTemplate(
        "trend_pullback_short",
        "趋势回调做空",
        "trend",
        "short",
        "背景空头；K1/K2 反抽 EMA20 或前闸门后出现强收盘、L1/L2、ii 下破或外包反转。",
        "止损放在信号 K 或反抽高点上方；目标至少 2R，第一目标可取前低/闸门下沿。",
        "优先 stop-sell 跌破信号 K 低点；若已跌破但风险过宽，等待二次反抽。",
    ),
    "range_reversal_long": StrategyTemplate(
        "range_reversal_long",
        "震荡下沿反转做多",
        "trading_range",
        "long",
        "市场震荡；价格测试闸门下沿后拒绝下破，出现下影、微双底、ii 上破或失败下破。",
        "止损在震荡下沿/信号 K 低点下方；目标先看中轴，再看上沿；低于 1.5R 则等待。",
        "使用 limit-buy 靠近下沿或 stop-buy 突破信号 K 高点，避免区间中部追单。",
    ),
    "range_reversal_short": StrategyTemplate(
        "range_reversal_short",
        "震荡上沿反转做空",
        "trading_range",
        "short",
        "市场震荡；价格测试闸门上沿后拒绝上破，出现上影、微双顶、ii 下破或失败上破。",
        "止损在震荡上沿/信号 K 高点上方；目标先看中轴，再看下沿；低于 1.5R 则等待。",
        "使用 limit-sell 靠近上沿或 stop-sell 跌破信号 K 低点，避免区间中部追单。",
    ),
    "breakout_continuation_long": StrategyTemplate(
        "breakout_continuation_long",
        "闸门上破延续做多",
        "breakout",
        "long",
        "K1 有效上破闸门，收盘靠近高位；后续 K 不应立刻回到闸门内，最好有二次跟随。",
        "止损在突破 K 中点或闸门上沿下方；目标至少 2R，优先用 measured move 或前方磁铁位。",
        "可 stop-buy 跟随突破后的高点；若突破 K 过大或 ATR_x>2，等待回踩确认。",
    ),
    "breakout_continuation_short": StrategyTemplate(
        "breakout_continuation_short",
        "闸门下破延续做空",
        "breakout",
        "short",
        "K1 有效下破闸门，收盘靠近低位；后续 K 不应立刻回到闸门内，最好有二次跟随。",
        "止损在突破 K 中点或闸门下沿上方；目标至少 2R，优先用 measured move 或前方磁铁位。",
        "可 stop-sell 跟随跌破后的低点；若突破 K 过大或 ATR_x>2，等待反抽确认。",
    ),
    "breakout_failure_long": StrategyTemplate(
        "breakout_failure_long",
        "下破失败反向做多",
        "reversal",
        "long",
        "价格跌破下闸门后快速收回，K1/K2 形成失败下破、强多头收盘或微双底。",
        "止损在失败突破低点下方；目标先看区间中轴，强势再看上沿；要求至少 1.8R。",
        "优先 stop-buy 上破失败突破 K 高点；若已经回到区间中部，等待回踩。",
    ),
    "breakout_failure_short": StrategyTemplate(
        "breakout_failure_short",
        "上破失败反向做空",
        "reversal",
        "short",
        "价格上破上闸门后快速跌回，K1/K2 形成失败上破、强空头收盘或微双顶。",
        "止损在失败突破高点上方；目标先看区间中轴，强势再看下沿；要求至少 1.8R。",
        "优先 stop-sell 跌破失败突破 K 低点；若已经回到区间中部，等待反抽。",
    ),
    "barbwire_wait": StrategyTemplate(
        "barbwire_wait",
        "铁丝网观望",
        "trading_range",
        "neutral",
        "连续重叠、ii/iii 密集或实体很小，缺少清晰方向信号链。",
        "没有明确 2R 结构前不交易；等待突破并回踩，或等待区间边缘反转。",
        "输出 wait；只记录观察点，不给入场价。",
    ),
    "ema20_magnet_wait": StrategyTemplate(
        "ema20_magnet_wait",
        "EMA20 磁铁观望",
        "unknown",
        "neutral",
        "价格围绕 EMA20 反复穿越，闸门内震荡，信号链互相抵消。",
        "避免在磁铁位附近给低盈亏比交易；等待脱离 EMA20 后的二次信号。",
        "输出 wait 或 avoid；只保留下一根需要观察的价位。",
    ),
}


def get_template(template_id: str) -> StrategyTemplate:
    return TEMPLATES[template_id]
