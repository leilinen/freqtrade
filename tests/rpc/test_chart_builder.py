# pragma pylint: disable=missing-docstring, protected-access, unused-argument

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("mplfinance")  # skip module if dep missing

from freqtrade.rpc.chart_builder import generate_candlestick_chart


def _make_df(n: int) -> pd.DataFrame:
    """Build a synthetic OHLCV df with `n` rows."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 100 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.uniform(10, 100, n),
        }
    )


def test_generate_candlestick_chart_returns_png():
    df = _make_df(50)
    png = generate_candlestick_chart("BTC/USDT", "5m", df, num_candles=20)
    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "PNG header missing"


def test_generate_candlestick_chart_default_20_candles():
    df = _make_df(40)
    png = generate_candlestick_chart("ETH/USDT", "5m", df)
    assert png[:4] == b"\x89PNG"


def test_generate_candlestick_chart_not_enough_candles():
    df = _make_df(10)
    with pytest.raises(ValueError, match="Not enough candles"):
        generate_candlestick_chart("BTC/USDT", "5m", df, num_candles=20)


def test_generate_candlestick_chart_invalid_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises((KeyError, ValueError)):
        generate_candlestick_chart("BTC/USDT", "5m", df, num_candles=2)
