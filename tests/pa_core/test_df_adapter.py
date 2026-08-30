"""Smoke tests for pa_core layer 1: indicators + freqtrade DataFrame adapter.

Validates the porting contract from docs/pa-migration-plan.md:
- EMA/ATR seeding semantics survive the port unchanged.
- df_to_kline_frame reproduces PA_Agent's build_analysis_frame semantics:
  newest-first bars, seq rebase (1 = newest), warmup-extended indicators
  sliced to the frame window.
"""
import math

import numpy as np
import pandas as pd

from pa_core.data_structures import KlineFrame
from pa_core.indicators import atr_full, ema_full
from pa_core.snapshot import frame_is_pure_closed
from pa_core.util.df_adapter import df_to_kline_frame


def _make_df(rows: int = 80) -> pd.DataFrame:
    """Synthetic freqtrade-style OHLCV frame (ascending, tz-aware UTC dates)."""
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    dates = pd.date_range(start=start, periods=rows, freq="1h")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.standard_normal(rows))
    open_ = close + rng.standard_normal(rows) * 0.1
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(rows)) * 0.2
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(rows)) * 0.2
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.random(rows) * 1000,
        }
    )


def test_ema_seed_is_simple_mean():
    result = ema_full([1.0, 2.0, 3.0, 4.0, 5.0], period=3)
    assert all(math.isnan(x) for x in result[:2])
    assert math.isclose(result[2], 2.0)  # mean(1, 2, 3)


def test_atr_seed_is_mean_true_ranges():
    highs = [10.0, 11.0, 12.0, 13.0]
    lows = [9.0, 10.0, 10.5, 12.0]
    closes = [9.5, 10.5, 11.5, 12.5]
    result = atr_full(highs, lows, closes, period=3)
    assert all(math.isnan(x) for x in result[:2])
    trs = [1.0, max(1.0, abs(11 - 9.5), abs(10 - 9.5)), max(1.5, abs(12 - 10.5), abs(10.5 - 10.5))]
    assert math.isclose(result[2], sum(trs[:3]) / 3)


def test_adapter_frame_structure():
    df = _make_df(80)
    frame = df_to_kline_frame(df, "BTC/USDT", "1h", 30, now_ms=1_700_000_000_000)
    assert isinstance(frame, KlineFrame)
    assert frame.symbol == "BTC/USDT"
    assert frame.timeframe == "1h"
    assert frame.snapshot_ts_local_ms == 1_700_000_000_000
    assert len(frame.bars) == 30
    # newest-first: bars[0] is the last df row
    assert frame.bars[0].seq == 1
    assert math.isclose(frame.bars[0].close, float(df.iloc[-1]["close"]))
    # seq strictly ascending 1..n, all closed
    assert [b.seq for b in frame.bars] == list(range(1, 31))
    assert frame_is_pure_closed(frame)
    # ts_open is canonical milliseconds
    assert frame.bars[0].ts_open > 1e12
    # freqtrade df has no amount/pct_chg columns → defaults
    assert frame.bars[0].amount == 0.0
    assert frame.bars[0].pct_chg is None


def test_adapter_indicator_alignment_and_warmup():
    df = _make_df(80)
    frame = df_to_kline_frame(df, "BTC/USDT", "1h", 30, warmup_bars=50)
    assert len(frame.indicators.ema20) == 30
    assert len(frame.indicators.atr14) == 30
    # warmup applied: newest EMA must equal EMA over ALL 80 closes, not just 30
    closes_all = [float(x) for x in df["close"]]
    ema_all = ema_full(closes_all, period=20)
    assert math.isclose(frame.indicators.ema20[0], ema_all[-1])
    highs_all = [float(x) for x in df["high"]]
    lows_all = [float(x) for x in df["low"]]
    atr_all = atr_full(highs_all, lows_all, closes_all, period=14)
    assert math.isclose(frame.indicators.atr14[0], atr_all[-1])
    # 80 bars ≫ warm-up periods → no nan inside the returned window
    assert not any(math.isnan(x) for x in frame.indicators.ema20)
    assert not any(math.isnan(x) for x in frame.indicators.atr14)


def test_adapter_oldest_bar_matches_window_start():
    df = _make_df(80)
    frame = df_to_kline_frame(df, "BTC/USDT", "1h", 30)
    # bars[-1] is the 30th-newest row of the df
    expected_date = df.iloc[-30]["date"]
    from pa_core.datetime_ts import datetime_to_ts_ms

    assert math.isclose(frame.bars[-1].ts_open, datetime_to_ts_ms(expected_date))


def test_adapter_insufficient_bars_returns_none():
    df = _make_df(10)
    assert df_to_kline_frame(df, "BTC/USDT", "1h", 30) is None
    assert df_to_kline_frame(df.iloc[0:0], "BTC/USDT", "1h", 30) is None
    assert df_to_kline_frame(df, "BTC/USDT", "1h", 0) is None
