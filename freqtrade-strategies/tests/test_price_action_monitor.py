"""
Unit tests for PriceActionMonitor._persist_kline and PaKline model.

Run from repo root with the project venv:
  .venv/bin/pytest freqtrade-strategies/tests/test_price_action_monitor.py -v
"""
import logging
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# The strategy module lives outside the freqtrade package and imports
# freqtrade.strategy.IStrategy + talib at module level.  We mock the
# heavy deps so the tests can run without a full freqtrade install.
# ---------------------------------------------------------------------------
_mock_ft_strategy = MagicMock()
sys.modules.setdefault("freqtrade", MagicMock())
sys.modules.setdefault("freqtrade.strategy", _mock_ft_strategy)
sys.modules.setdefault("talib", MagicMock())
sys.modules.setdefault("talib.abstract", MagicMock())

# Provide a concrete class for IStrategy so isinstance / super() works
class _FakeIStrategy:
    def __init__(self, config):
        self.config = config
        self.dp = None
        self.timeframe = "1h"

_mock_ft_strategy.IStrategy = _FakeIStrategy

from price_action_monitor import (  # noqa: E402
    PaKline,
    PriceActionMonitor,
    _Base,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(rows, timeframe="1h", last_date=None):
    """Build a minimal OHLCV DataFrame.

    `rows` is the number of candles.  Dates are hourly going back from
    `last_date` (defaults to 2025-01-15 10:00 UTC).
    """
    if last_date is None:
        last_date = pd.Timestamp("2025-01-15 10:00", tz="UTC")
    dates = pd.date_range(end=last_date, periods=rows, freq=timeframe)
    rng = np.random.default_rng(42)
    close_vals = 100 + rng.random(rows).cumsum()
    return pd.DataFrame(
        {
            "date": dates,
            "open": close_vals - 0.5,
            "high": close_vals + 1.0,
            "low": close_vals - 1.0,
            "close": close_vals,
            "volume": 1000.0 * rng.random(rows),
        }
    )


def _make_strategy(**overrides):
    """Create a PriceActionMonitor instance with mocked PG session."""
    s = PriceActionMonitor(config={})
    s.timeframe = overrides.pop("timeframe", "1h")

    # Mock _pg_session_factory so we can inspect calls
    mock_session = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_session)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    factory = MagicMock(return_value=mock_ctx)
    s._pg_session_factory = factory

    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ===================================================================
# Tests: PaKline model
# ===================================================================


class TestPaKlineModel:
    """Verify the ORM model is correctly defined."""

    def test_table_name(self):
        assert PaKline.__tablename__ == "pa_kline"

    def test_model_is_on_base(self):
        """PaKline must share _Base so create_all() picks it up."""
        assert PaKline in _Base.registry._class_registry.values()

    def test_unique_constraint_name(self):
        constraints = [c for c in PaKline.__table_args__ if hasattr(c, "name")]
        names = [c.name for c in constraints]
        assert "ux_pa_kline_identity" in names

    def test_columns_exist(self):
        col_names = {c.name for c in PaKline.__table__.columns}
        expected = {
            "id", "created_at", "symbol", "timeframe", "candle_time",
            "open", "high", "low", "close", "volume",
        }
        assert expected == col_names


# ===================================================================
# Tests: _persist_kline
# ===================================================================


class TestPersistKline:
    """Unit tests for PriceActionMonitor._persist_kline."""

    def test_skips_when_session_factory_is_none(self):
        """Should no-op silently when PG is not configured — no TypeError on calling None."""
        s = _make_strategy()
        s._pg_session_factory = None
        assert s._pg_session_factory is None
        df = _make_ohlcv_df(5)

        # If the guard is removed, this would raise TypeError: None is not callable
        s._persist_kline("BTC/USDT", df)

    def test_skips_empty_dataframe(self):
        """Should return immediately for an empty DataFrame."""
        s = _make_strategy()
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        s._persist_kline("BTC/USDT", df)

        # session_factory should not have been called
        s._pg_session_factory.assert_not_called()

    def test_writes_latest_closed_candle(self):
        """Should UPSERT the last closed candle (date < current period start)."""
        s = _make_strategy(timeframe="1h")

        # 5 hourly candles ending at 10:00 UTC.
        # Pretend "now" is 10:30 UTC → period_start = 10:00.
        # Candles at 06:00-09:00 are closed; 10:00 is current (not closed).
        df = _make_ohlcv_df(5, "1h", pd.Timestamp("2025-01-15 10:00", tz="UTC"))

        fake_now = pd.Timestamp("2025-01-15 10:30", tz="UTC")
        with patch("price_action_monitor.pd.Timestamp.now", return_value=fake_now):
            s._persist_kline("BTC/USDT", df)

        # session factory was entered once
        s._pg_session_factory.assert_called_once()
        session = s._pg_session_factory.return_value.__enter__.return_value
        session.execute.assert_called_once()
        session.commit.assert_called_once()

        # Verify the params passed to session.execute
        call_args = session.execute.call_args
        sql_text = call_args[0][0].text  # sqlalchemy.text() stores SQL in .text
        assert "INSERT INTO pa_kline" in sql_text
        assert "ON CONFLICT" in sql_text

        params = call_args[0][1]
        assert params["symbol"] == "BTC/USDT"
        assert params["timeframe"] == "1h"
        # The latest closed candle should be the one at 09:00 (index 3)
        assert params["candle_time"] == pd.Timestamp("2025-01-15 09:00", tz="UTC")
        assert params["close"] == pytest.approx(df.iloc[3]["close"])

    def test_skips_when_all_candles_are_current(self):
        """If no candle is older than period_start, nothing should be written."""
        s = _make_strategy(timeframe="1h")

        # Candle at 10:00 UTC, "now" is 10:05 → period_start = 10:00
        # The candle at 10:00 is NOT < 10:00, so nothing is closed.
        df = _make_ohlcv_df(1, "1h", pd.Timestamp("2025-01-15 10:00", tz="UTC"))

        fake_now = pd.Timestamp("2025-01-15 10:05", tz="UTC")
        with patch("price_action_monitor.pd.Timestamp.now", return_value=fake_now):
            s._persist_kline("BTC/USDT", df)

        s._pg_session_factory.assert_not_called()

    def test_handles_db_exception_gracefully(self, caplog):
        """DB errors should be caught and logged, not propagated."""
        caplog.set_level(logging.DEBUG)
        s = _make_strategy(timeframe="1h")
        df = _make_ohlcv_df(5, "1h", pd.Timestamp("2025-01-15 10:00", tz="UTC"))

        # Make session.execute raise
        session = s._pg_session_factory.return_value.__enter__.return_value
        session.execute.side_effect = Exception("connection lost")

        fake_now = pd.Timestamp("2025-01-15 10:30", tz="UTC")
        with patch("price_action_monitor.pd.Timestamp.now", return_value=fake_now):
            # Must not raise
            s._persist_kline("BTC/USDT", df)

        assert any("K-line write failed" in r.message for r in caplog.records)

    def test_uses_strategy_timeframe(self):
        """The timeframe in the UPSERT params must match self.timeframe."""
        s = _make_strategy(timeframe="4h")
        df = _make_ohlcv_df(5, "4h", pd.Timestamp("2025-01-15 12:00", tz="UTC"))

        fake_now = pd.Timestamp("2025-01-15 13:00", tz="UTC")
        with patch("price_action_monitor.pd.Timestamp.now", return_value=fake_now):
            s._persist_kline("ETH/USDT", df)

        session = s._pg_session_factory.return_value.__enter__.return_value
        params = session.execute.call_args[0][1]
        assert params["timeframe"] == "4h"
        assert params["symbol"] == "ETH/USDT"


# ===================================================================
# Tests: populate_entry_trend integration
# ===================================================================


class TestPopulateEntryTrendKline:
    """Verify _persist_kline is called from populate_entry_trend."""

    def test_calls_persist_kline_in_dry_run(self):
        """In dry_run mode, _persist_kline should be called."""
        s = _make_strategy()
        # Mock dp to report dry_run
        s.dp = MagicMock()
        s.dp.runmode.value = "dry_run"

        df = _make_ohlcv_df(5)
        # Need signal columns to avoid _check_and_notify errors
        df["signal_quality"] = "none"
        df["signal_direction"] = "none"

        with patch.object(s, "_persist_kline") as mock_persist, \
             patch.object(s, "_check_and_notify"):
            s.populate_entry_trend(df, {"pair": "BTC/USDT"})

        mock_persist.assert_called_once_with("BTC/USDT", df)

    def test_does_not_call_persist_kline_without_dp(self):
        """When dp is None, _persist_kline should NOT be called."""
        s = _make_strategy()
        s.dp = None

        df = _make_ohlcv_df(5)

        with patch.object(s, "_persist_kline") as mock_persist:
            s.populate_entry_trend(df, {"pair": "BTC/USDT"})

        mock_persist.assert_not_called()

    def test_does_not_call_persist_kline_in_backtest(self):
        """In non-live/dry_run mode, _persist_kline should NOT be called."""
        s = _make_strategy()
        s.dp = MagicMock()
        s.dp.runmode.value = "backtest"

        df = _make_ohlcv_df(5)

        with patch.object(s, "_persist_kline") as mock_persist:
            s.populate_entry_trend(df, {"pair": "BTC/USDT"})

        mock_persist.assert_not_called()

    def test_returns_dataframe_unchanged(self):
        """populate_entry_trend must return the dataframe as-is."""
        s = _make_strategy()
        s.dp = MagicMock()
        s.dp.runmode.value = "dry_run"

        df = _make_ohlcv_df(5)
        df["signal_quality"] = "none"
        df["signal_direction"] = "none"

        with patch.object(s, "_persist_kline"), \
             patch.object(s, "_check_and_notify"):
            result = s.populate_entry_trend(df, {"pair": "BTC/USDT"})

        assert result is df
