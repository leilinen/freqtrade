"""PA_Agent strategy-file templates used by deterministic routing."""
from __future__ import annotations

from dataclasses import dataclass


ATR_VETO = (
    "ATR 突变否决：若 atr_expand_ratio > 2 且正在突破闸门，"
    "按假突破风险处理。"
)


@dataclass(frozen=True)
class StrategyTemplate:
    """Local strategy text routed from market diagnosis."""

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


def _template(
    template_id: str,
    name: str,
    cycle: str,
    direction: str,
    signal_chain: str,
    order_plan: str,
) -> StrategyTemplate:
    return StrategyTemplate(
        template_id,
        name,
        cycle,
        direction,
        signal_chain,
        "止损必须有结构依据；目标优先使用 2R、区间边界、"
        "前高低或 measured move。",
        order_plan,
    )


TEMPLATES: dict[str, StrategyTemplate] = {
    "上涨通道分析识别.txt": _template(
        "上涨通道分析识别.txt",
        "上涨通道分析识别",
        "channel",
        "bullish",
        "识别多头通道、回调、EMA20 支撑、H1/H2 或突破回踩。",
        "只作为结构识别依据，交易决策需等待信号棒和风险收益确认。",
    ),
    "上涨通道交易策略.txt": _template(
        "上涨通道交易策略.txt",
        "上涨通道交易策略",
        "channel",
        "bullish",
        "多头背景下优先顺势做多，避免在通道上沿追高。",
        "可用 stop-buy 突破信号棒，或在回调支撑位限价计划做多。",
    ),
    "下跌通道分析识别.txt": _template(
        "下跌通道分析识别.txt",
        "下跌通道分析识别",
        "channel",
        "bearish",
        "识别空头通道、反抽、EMA20 压制、L1/L2 或突破回踩。",
        "只作为结构识别依据，交易决策需等待信号棒和风险收益确认。",
    ),
    "下跌通道交易策略.txt": _template(
        "下跌通道交易策略.txt",
        "下跌通道交易策略",
        "channel",
        "bearish",
        "空头背景下优先顺势做空，避免在通道下沿追空。",
        "可用 stop-sell 跌破信号棒，或在反抽压力位限价计划做空。",
    ),
    "文件13-窄通道与宽通道策略.txt": _template(
        "文件13-窄通道与宽通道策略.txt",
        "窄通道与宽通道策略",
        "channel",
        "neutral",
        "区分微型/窄/正常/宽通道，判断可追随还是等待回调。",
        "窄通道偏顺势，宽通道更重视边界和二次入场。",
    ),
    "极速上涨分析识别.txt": _template(
        "极速上涨分析识别.txt",
        "极速上涨分析识别",
        "spike",
        "bullish",
        "识别强多头尖峰、连续强阳线、缺口棒和快速突破。",
        "若尖峰过大或 ATR 扩张过强，等待回踩或二次确认。",
    ),
    "极速上涨交易策略.txt": _template(
        "极速上涨交易策略.txt",
        "极速上涨交易策略",
        "spike",
        "bullish",
        "强尖峰后优先顺势，结束阶段需切换到 spike-and-channel。",
        "只在止损可控时追随；否则等待回踩 EMA20 或突破位。",
    ),
    "极速下跌分析识别.txt": _template(
        "极速下跌分析识别.txt",
        "极速下跌分析识别",
        "spike",
        "bearish",
        "识别强空头尖峰、连续强阴线、缺口棒和快速下破。",
        "若尖峰过大或 ATR 扩张过强，等待反抽或二次确认。",
    ),
    "极速下跌交易策略.txt": _template(
        "极速下跌交易策略.txt",
        "极速下跌交易策略",
        "spike",
        "bearish",
        "强尖峰后优先顺势，结束阶段需切换到 spike-and-channel。",
        "只在止损可控时追随；否则等待反抽 EMA20 或突破位。",
    ),
    "震荡区间分析识别.txt": _template(
        "震荡区间分析识别.txt",
        "震荡区间分析识别",
        "trading_range",
        "neutral",
        "识别区间上下沿、中轴、假突破、重叠和交易区间惯性。",
        "区间中部不追单，优先等待边界反转或清晰突破回踩。",
    ),
    "震荡区间交易策略.txt": _template(
        "震荡区间交易策略.txt",
        "震荡区间交易策略",
        "trading_range",
        "neutral",
        "区间边界逆向交易优先；突破必须有跟随或回踩确认。",
        "可在边界计划限价，区间中部输出 wait/avoid。",
    ),
    "文件14-楔形形态分析交易.txt": _template(
        "文件14-楔形形态分析交易.txt",
        "楔形形态分析交易",
        "wedge",
        "neutral",
        "三推、收敛、动能衰竭和楔形突破/反转。",
        "等待第三推后的失败突破、反转棒或突破回踩确认。",
    ),
    "文件15-二次入场机会.txt": _template(
        "文件15-二次入场机会.txt",
        "二次入场机会",
        "reversal",
        "neutral",
        "H2/L2、反转尝试、第一次失败后的第二次信号。",
        "第二次信号优先于第一次逆势尝试；无确认则等待。",
    ),
    "文件18-突破失败与突破测试.txt": _template(
        "文件18-突破失败与突破测试.txt",
        "突破失败与突破测试",
        "breakout",
        "neutral",
        "突破无跟随、快速回到区间、突破回踩和失败的失败。",
        "失败突破可反向；突破回踩顺势时必须确认突破位守住。",
    ),
    "文件19-H1H2-L1L2计数.txt": _template(
        "文件19-H1H2-L1L2计数.txt",
        "H1H2-L1L2计数",
        "count",
        "neutral",
        "High1/High2/Low1/Low2 计数入场和二次入场质量。",
        "H2/L2 通常优于 H1/L1；必须结合背景和信号棒质量。",
    ),
    "文件20-AlwaysIn与20GB.txt": _template(
        "文件20-AlwaysIn与20GB.txt",
        "AlwaysIn与20GB",
        "always_in",
        "neutral",
        "Always In 方向、20 gap bar、强趋势惯性和逆势风险。",
        "顺 Always In 优先；逆势需要二次确认和清晰止损。",
    ),
    "文件21-铁丝网与无交易环境.txt": _template(
        "文件21-铁丝网与无交易环境.txt",
        "铁丝网与无交易环境",
        "barbwire",
        "neutral",
        "密集重叠、区间中部、信号互相抵消和无交易环境。",
        "无清晰边界或突破前输出 wait/avoid，不给完整订单计划。",
    ),
    "文件22-信号失败后的磁力位.txt": _template(
        "文件22-信号失败后的磁力位.txt",
        "信号失败后的磁力位",
        "magnet",
        "neutral",
        "失败信号、被套交易者、价格回吸到磁力位。",
        "利用失败后的磁力目标，但避免在磁力位附近低 R 入场。",
    ),
    "文件24-最终旗形与趋势末端.txt": _template(
        "文件24-最终旗形与趋势末端.txt",
        "最终旗形与趋势末端",
        "final_flag",
        "neutral",
        "趋势末段旗形、最后推进、动能衰竭和反转准备。",
        "趋势末端不追价；等待失败突破、二次信号或明确反转。",
    ),
    "文件25-主要趋势反转MTR.txt": _template(
        "文件25-主要趋势反转MTR.txt",
        "主要趋势反转MTR",
        "mtr",
        "neutral",
        "主要趋势反转、趋势线突破、测试极点和二次入场。",
        "MTR 需完整结构确认；第一次逆势尝试通常只观察。",
    ),
    "文件27-三角形与收敛形态.txt": _template(
        "文件27-三角形与收敛形态.txt",
        "三角形与收敛形态",
        "triangle",
        "neutral",
        "上升/下降/对称/扩张三角形，收敛压缩后的突破选择。",
        "等待尖峰级突破或失败突破，不在收敛中部提前追单。",
    ),
    "文件28-双重顶底与微型结构.txt": _template(
        "文件28-双重顶底与微型结构.txt",
        "双重顶底与微型结构",
        "double_top_bottom",
        "neutral",
        "双顶、双底、微型双顶底和短线测试失败。",
        "结合区间边界或趋势背景；确认失败测试后再决策。",
    ),
}


def get_template(template_id: str) -> StrategyTemplate:
    return TEMPLATES[template_id]
