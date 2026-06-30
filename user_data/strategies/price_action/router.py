"""Deterministic routing from market diagnosis to PA strategy templates."""
from __future__ import annotations

import re
from typing import Any

from .strategy_templates import StrategyTemplate, get_template


BULLISH_CHANNEL_FILES = (
    "上涨通道分析识别.txt",
    "上涨通道交易策略.txt",
)
BEARISH_CHANNEL_FILES = (
    "下跌通道分析识别.txt",
    "下跌通道交易策略.txt",
)
CHANNEL_WIDTH_FILE = "文件13-窄通道与宽通道策略.txt"
BULLISH_SPIKE_FILES = (
    "极速上涨分析识别.txt",
    "极速上涨交易策略.txt",
)
BEARISH_SPIKE_FILES = (
    "极速下跌分析识别.txt",
    "极速下跌交易策略.txt",
)
RANGE_FILES = (
    "震荡区间分析识别.txt",
    "震荡区间交易策略.txt",
)
WEDGE_FILE = "文件14-楔形形态分析交易.txt"
REVERSAL_FILE = "文件15-二次入场机会.txt"
BREAKOUT_FAILURE_FILE = "文件18-突破失败与突破测试.txt"
H1H2_FILE = "文件19-H1H2-L1L2计数.txt"
ALWAYS_IN_FILE = "文件20-AlwaysIn与20GB.txt"
BARBWIRE_FILE = "文件21-铁丝网与无交易环境.txt"
MAGNET_FILE = "文件22-信号失败后的磁力位.txt"
FINAL_FLAG_FILE = "文件24-最终旗形与趋势末端.txt"
MTR_FILE = "文件25-主要趋势反转MTR.txt"
TRIANGLE_FILE = "文件27-三角形与收敛形态.txt"
DOUBLE_TOP_BOTTOM_FILE = "文件28-双重顶底与微型结构.txt"

CHANNEL_STATES = frozenset(
    {"micro_channel", "tight_channel", "normal_channel", "broad_channel"}
)
RANGE_STATES = frozenset({"trading_range", "trending_tr"})
SKIP_STATES = frozenset({"extreme_tr", "unknown"})

ENTRY_SETUP_PATTERN_OVERLAY: dict[str, tuple[str, ...]] = {
    "wedge": ("wedge",),
    "breakout_pullback": ("breakout_pullback",),
    "mtr": ("mtr", "reversal_attempt"),
    "h1": ("h1",),
    "h2": ("h2", "reversal_attempt"),
    "l1": ("l1",),
    "l2": ("l2", "reversal_attempt"),
    "tr_boundary": ("middle_range", "barbwire"),
}
CYCLE_PATTERN_TAGS: dict[str, tuple[str, ...]] = {
    "trading_range": ("middle_range", "barbwire", "overlap"),
    "trending_tr": ("middle_range", "overlap"),
}
PATTERN_KEYWORD_TAGS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("楔形", "三推", "三推动", "wedge"), "wedge"),
    (("突破测试", "突破回踩", "失败的失败"), "breakout_test"),
    (("假突破", "突破失败", "failed breakout", "突破后快速收复"), "breakout_failure"),
    (("mtr", "主要趋势反转", "趋势反转尝试"), "mtr"),
    (("铁丝网", "barbwire", "凝滞区"), "barbwire"),
    (("重叠度高", "重叠多", "k线重叠", "重叠严重"), "overlap"),
    (("区间下沿", "区间上沿", "区间边界", "交易区间"), "middle_range"),
    (("always in", "ail", "ais", "20gb", "缺口棒"), "always_in"),
    (("磁力", "套住", "trapped", "信号失败"), "failed_signal"),
    (("双顶", "双底", "双重顶", "双重底", "double top"), "double_top_bottom"),
    (("上升三角形",), "ascending_triangle"),
    (("下降三角形",), "descending_triangle"),
    (("对称三角形",), "symmetrical_triangle"),
    (("扩张三角形",), "expanding_triangle"),
    (("最终旗形", "final flag", "ffes"), "final_flag"),
)
HL_COUNT_RE = re.compile(
    r"(?<![a-z])(?:h[12]|l[12])(?![a-z])|计数入场|high\s*[12]|low\s*[12]",
    re.IGNORECASE,
)


def route_strategies(diagnosis: dict[str, Any]) -> list[StrategyTemplate]:
    """Return ordered, deduplicated PA strategy templates for trade decision."""
    return [get_template(template_id) for template_id in route_strategy_template_ids(diagnosis)]


def route_strategy_template_ids(diagnosis: dict[str, Any]) -> list[str]:
    """Mirror PA_Agent Stage-1 diagnosis to Stage-2 strategy-file routing."""
    cycle = str(diagnosis.get("cycle_position", "unknown") or "unknown").lower()
    direction = str(diagnosis.get("direction", "neutral") or "neutral").lower()
    spike_stage = diagnosis.get("spike_stage")
    alternative_cycle = diagnosis.get("alternative_cycle_position")
    patterns = set(merge_detected_patterns(diagnosis))

    selected: list[str] = []
    selected.extend(_base_files_for_cycle(cycle, direction, spike_stage=spike_stage))

    trend_context = diagnosis.get("trend_context") or {}
    recent_spike = (
        trend_context.get("recent_spike") if isinstance(trend_context, dict) else None
    )
    if recent_spike == "bullish" and cycle != "spike" and direction == "bullish":
        selected.extend(BULLISH_SPIKE_FILES)
    elif recent_spike == "bearish" and cycle != "spike" and direction == "bearish":
        selected.extend(BEARISH_SPIKE_FILES)

    if alternative_cycle and str(alternative_cycle).lower() != cycle:
        selected.extend(
            _base_files_for_cycle(
                str(alternative_cycle).lower(),
                direction,
                spike_stage=None,
            )
        )

    if "wedge" in patterns:
        selected.append(WEDGE_FILE)
    if (
        cycle in CHANNEL_STATES
        or "reversal_attempt" in patterns
        or "mtr" in patterns
        or "final_flag" in patterns
        or "h2" in patterns
        or "l2" in patterns
    ):
        selected.append(REVERSAL_FILE)
    if "mtr" in patterns:
        selected.append(MTR_FILE)
    if "final_flag" in patterns:
        selected.append(FINAL_FLAG_FILE)
    if cycle in CHANNEL_STATES or any(p in patterns for p in ("h1", "h2", "l1", "l2")):
        selected.append(H1H2_FILE)
    if any(
        p in patterns
        for p in ("breakout_failure", "failed_breakout", "breakout_test", "breakout_pullback")
    ):
        selected.append(BREAKOUT_FAILURE_FILE)
    if any(p in patterns for p in ("always_in", "ail", "ais", "20gb", "gap_bar")):
        selected.append(ALWAYS_IN_FILE)
    if cycle in RANGE_STATES or any(
        p in patterns for p in ("barbwire", "wire", "overlap", "middle_range")
    ):
        selected.append(BARBWIRE_FILE)
    if any(
        p in patterns
        for p in (
            "failed_signal",
            "breakout_failure",
            "failed_breakout",
            "magnet",
            "trapped_traders",
        )
    ):
        selected.append(MAGNET_FILE)
    if any(
        p in patterns
        for p in (
            "ascending_triangle",
            "descending_triangle",
            "symmetrical_triangle",
            "expanding_triangle",
        )
    ):
        selected.append(TRIANGLE_FILE)
    if "double_top_bottom" in patterns:
        selected.append(DOUBLE_TOP_BOTTOM_FILE)

    return _dedupe(selected)


def merge_detected_patterns(diagnosis: dict[str, Any]) -> list[str]:
    """Merge model tags, entry setup overlays, cycle overlays, and key-signal hints."""
    patterns: list[str] = []
    seen: set[str] = set()
    for raw in diagnosis.get("detected_patterns") or []:
        key = str(raw).strip().lower()
        if key and key not in seen:
            seen.add(key)
            patterns.append(key)

    bar_analysis = diagnosis.get("bar_analysis") or {}
    setup = str(
        bar_analysis.get("entry_setup_type")
        or diagnosis.get("entry_setup")
        or ""
    ).strip().lower()
    for key in ENTRY_SETUP_PATTERN_OVERLAY.get(setup, ()):
        if key not in seen:
            seen.add(key)
            patterns.append(key)

    cycle = str(diagnosis.get("cycle_position", "") or "").strip().lower()
    for key in CYCLE_PATTERN_TAGS.get(cycle, ()):
        if key not in seen:
            seen.add(key)
            patterns.append(key)

    signal_texts = [str(s) for s in diagnosis.get("key_signals") or []]
    signal_texts.append(str(diagnosis.get("risk_warning") or ""))
    text = " ".join(signal_texts).lower()
    if HL_COUNT_RE.search(text):
        for key in ("h1", "h2", "l1", "l2"):
            if re.search(rf"(?<![a-z]){key}(?![a-z])", text) and key not in seen:
                seen.add(key)
                patterns.append(key)
    for keywords, tag in PATTERN_KEYWORD_TAGS:
        if any(keyword.lower() in text for keyword in keywords) and tag not in seen:
            seen.add(tag)
            patterns.append(tag)
    return patterns


def _base_files_for_cycle(
    cycle: str,
    direction: str,
    *,
    spike_stage: Any = None,
) -> list[str]:
    if cycle == "spike" and spike_stage == "transitioning":
        return _channel_files(direction)

    files: list[str] = []
    if cycle in CHANNEL_STATES:
        files.extend(_channel_files(direction))
        if cycle == "micro_channel" and spike_stage in ("active", "ending"):
            if direction == "bullish":
                files.extend(BULLISH_SPIKE_FILES)
            elif direction == "bearish":
                files.extend(BEARISH_SPIKE_FILES)
    elif cycle == "spike":
        if direction == "bullish":
            files.extend(BULLISH_SPIKE_FILES)
        elif direction == "bearish":
            files.extend(BEARISH_SPIKE_FILES)
        if spike_stage == "ending":
            files.extend(_channel_files(direction))
    elif cycle in RANGE_STATES:
        files.extend(RANGE_FILES)
    elif cycle in SKIP_STATES:
        pass
    return files


def _channel_files(direction: str) -> list[str]:
    files: list[str] = []
    if direction == "bullish":
        files.extend(BULLISH_CHANNEL_FILES)
    elif direction == "bearish":
        files.extend(BEARISH_CHANNEL_FILES)
    else:
        files.extend(RANGE_FILES)
    files.append(CHANNEL_WIDTH_FILE)
    return files


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
