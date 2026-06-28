"""Tests for PA_Agent-inspired market background detection."""
# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STRAT_DIR = str(Path(__file__).resolve().parents[2] / "user_data" / "strategies")
if _STRAT_DIR not in sys.path:
    sys.path.insert(0, _STRAT_DIR)

from price_action.background import MarketBackgroundAnalyzer


def _trend_df(
    rows: int,
    *,
    start: float = 100.0,
    step: float = 1.0,
    body: float = 0.7,
    atr: float = 1.0,
) -> pd.DataFrame:
    closes = start + np.arange(rows, dtype=float) * step
    bullish = step >= 0
    opens = closes - body if bullish else closes + body
    highs = np.maximum(opens, closes) + 0.2
    lows = np.minimum(opens, closes) - 0.2
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1000.0,
        }
    )
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["atr14"] = atr
    return df


def _range_df(rows: int) -> pd.DataFrame:
    closes = 100.0 + np.sin(np.arange(rows, dtype=float) / 3.0) * 0.4
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "open": opens,
            "high": 101.0,
            "low": 99.0,
            "close": closes,
            "volume": 1000.0,
        }
    )
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["atr14"] = 2.0
    return df


def _conflict_df() -> pd.DataFrame:
    older_down = _trend_df(80, start=220.0, step=-1.0)
    recent_up = _trend_df(40, start=130.0, step=1.2)
    df = pd.concat([older_down, recent_up], ignore_index=True)
    df["date"] = pd.date_range("2026-01-01", periods=len(df), freq="h")
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["atr14"] = 1.0
    return df


class TestMarketBackgroundAnalyzer:
    def test_detects_aligned_up_background(self):
        result = MarketBackgroundAnalyzer().evaluate(_trend_df(130, step=1.0))
        last = result.iloc[-1]

        assert last["background_direction"] == "up"
        assert last["trading_direction"] == "up"
        assert last["trend_relationship"] == "aligned"
        assert last["recent_momentum"] == "bullish"
        assert last["trend_strength"] > 0.6

    def test_detects_aligned_down_background(self):
        result = MarketBackgroundAnalyzer().evaluate(_trend_df(130, start=230.0, step=-1.0))
        last = result.iloc[-1]

        assert last["background_direction"] == "down"
        assert last["trading_direction"] == "down"
        assert last["trend_relationship"] == "aligned"
        assert last["recent_momentum"] == "bearish"
        assert last["trend_strength"] > 0.6

    def test_detects_range_background(self):
        result = MarketBackgroundAnalyzer().evaluate(_range_df(130))
        last = result.iloc[-1]

        assert last["background_direction"] == "range"
        assert last["trading_direction"] == "range"
        assert last["trend_relationship"] == "range"
        assert last["trend_strength"] < 0.3

    def test_detects_background_recent_conflict(self):
        result = MarketBackgroundAnalyzer().evaluate(_conflict_df())
        last = result.iloc[-1]

        assert last["background_direction"] == "down"
        assert last["trading_direction"] == "up"
        assert last["trend_relationship"] == "conflict"
        assert last["recent_momentum"] == "bullish"

    def test_short_history_is_unknown_for_major_windows(self):
        result = MarketBackgroundAnalyzer().evaluate(_trend_df(6, step=1.0))
        last = result.iloc[-1]

        assert last["background_direction"] == "unknown"
        assert last["trading_direction"] == "unknown"
        assert last["trend_relationship"] == "unknown"
