"""KlineFrame snapshot helpers.

Ported from PA_Agent ``pa_agent/data/snapshot.py`` (baseline 1090a5b).

Dropped vs. upstream: the bars-list builders (``build_analysis_frame``,
``build_live_frame``, ``build_display_frame``, ``_newest_closed_slice``) and
their ``bar_close_wait`` forming-bar detection. freqtrade dataframes contain
only closed candles (the exchange layer drops the incomplete candle by
default), so ``pa_core.util.df_adapter.df_to_kline_frame`` replaces that whole
path with a direct DataFrame → KlineFrame conversion using the same
warmup/seq-rebase semantics.

What is kept: ``compute_indicators`` (warmup-aware EMA20/ATR14 aligned
newest-first), the warmup constant, and the frame comparison helpers used by
incremental analysis and tests.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pa_core.indicators import atr_full, ema_full

if TYPE_CHECKING:
    from pa_core.data_structures import IndicatorBundle, KlineFrame

# Extra closed bars fetched before the AI window so EMA20/ATR14 can warm up.
# Only the newest *n* bars are sent to the model; indicators use this buffer.
INDICATOR_WARMUP_BARS = 50


def compute_indicators(bars: list) -> "IndicatorBundle":
    """Compute EMA20 and ATR14 for *bars* (newest-first order).

    Indicators are computed on the reversed (oldest-first) sequence and then
    reversed back so that index *i* aligns with ``bars[i]`` (K1 at index 0).
    """
    from pa_core.data_structures import IndicatorBundle

    # bars is newest-first; indicators need oldest-first input
    bars_asc = list(reversed(bars))

    closes = [b.close for b in bars_asc]
    highs = [b.high for b in bars_asc]
    lows = [b.low for b in bars_asc]

    ema20_asc = ema_full(closes, period=20)
    atr14_asc = atr_full(highs, lows, closes, period=14)

    # Reverse back to newest-first
    ema20 = tuple(reversed(ema20_asc))
    atr14 = tuple(reversed(atr14_asc))

    return IndicatorBundle(ema20=ema20, atr14=atr14)


def frame_is_pure_closed(frame: "KlineFrame") -> bool:
    """True when every bar on the frame is marked closed (no forming slot)."""
    return bool(frame.bars) and all(b.closed for b in frame.bars)


def frames_equal_for_chart(a: "KlineFrame", b: "KlineFrame") -> bool:
    """True when two frames would render the same candles and EMA (ignore snapshot time)."""
    if a.symbol != b.symbol or a.timeframe != b.timeframe:
        return False
    if len(a.bars) != len(b.bars):
        return False
    if a.bars != b.bars:
        return False
    return _indicators_equal(a.indicators, b.indicators)


def _indicators_equal(a: "IndicatorBundle", b: "IndicatorBundle") -> bool:
    if len(a.ema20) != len(b.ema20) or len(a.atr14) != len(b.atr14):
        return False
    for x, y in zip(a.ema20, b.ema20, strict=True):
        if not _float_equal(x, y):
            return False
    for x, y in zip(a.atr14, b.atr14, strict=True):
        if not _float_equal(x, y):
            return False
    return True


def _float_equal(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b
