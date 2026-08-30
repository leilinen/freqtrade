"""Core data types for the price action pipeline.

Ported from PA_Agent ``pa_agent/data/base.py`` (baseline 1090a5b).
Dropped vs. upstream: the ``DataSource`` ABC and its error classes — freqtrade
provides the data layer, and ``pa_core.util.df_adapter`` converts its
DataFrames into these types instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from pa_core.datetime_ts import ts_open_to_ms


# ── KlineBar ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KlineBar:
    """A single OHLCV bar with sequence number and closed flag."""
    seq: int           # 1 = newest closed bar, N = oldest; 0 = forming bar (not counted)
    ts_open: float     # Unix timestamp in milliseconds (UTC) of bar open
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0   # turnover amount (成交额); 0 when unavailable
    pct_chg: float | None = None  # daily change % from API when available
    closed: bool = True   # False for the currently-forming bar


def normalize_kline_bar(bar: KlineBar) -> KlineBar:
    """Ensure canonical ``ts_open`` (ms), ``high >= low``, and ``low <= close <= high``."""
    ts_ms = ts_open_to_ms(bar.ts_open)
    high = max(bar.high, bar.low)
    low = min(bar.high, bar.low)
    close = max(low, min(high, bar.close))
    if (
        high == bar.high
        and low == bar.low
        and close == bar.close
        and ts_ms == bar.ts_open
    ):
        return bar
    return KlineBar(
        seq=bar.seq,
        ts_open=ts_ms,
        open=bar.open,
        high=high,
        low=low,
        close=close,
        volume=bar.volume,
        amount=getattr(bar, "amount", 0.0),
        pct_chg=getattr(bar, "pct_chg", None),
        closed=bar.closed,
    )


# ── IndicatorBundle ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndicatorBundle:
    """Per-bar indicator values aligned to a KlineFrame's bars list."""
    ema20: tuple[float, ...]   # len == len(bars); nan for warm-up period
    atr14: tuple[float, ...]   # len == len(bars); nan for warm-up period


# ── KlineFrame ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KlineFrame:
    """Immutable snapshot of N bars plus computed indicators.

    bars[0] is the newest bar (seq=1, closed=True in analysis frames).
    bars[-1] is the oldest bar (seq=N).
    snapshot_ts_local_ms is the local machine time when the snapshot was taken.
    """
    symbol: str
    timeframe: str
    bars: tuple[KlineBar, ...]
    indicators: IndicatorBundle
    snapshot_ts_local_ms: int   # milliseconds since epoch, local time
