"""
Unit tests for PriceActionMonitor._persist_kline and PaKline model.

Run from repo root with the project venv:
  .venv/bin/pytest freqtrade-strategies/tests/test_price_action_monitor.py -v
"""
import json
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

def _make_ohlcv_df(rows, timeframe="1h", last_date=None, tz_naive=False):
    """Build a minimal OHLCV DataFrame.

    `rows` is the number of candles.  Dates are hourly going back from
    `last_date` (defaults to 2025-01-15 10:00 UTC).

    When `tz_naive=True`, dates are tz-naive (like freqtrade/Binance returns).
    When `tz_naive=False`, dates are tz-aware with UTC (like A-share sources).
    """
    if last_date is None:
        last_date = pd.Timestamp("2025-01-15 10:00", tz="UTC")
    dates = pd.date_range(end=last_date, periods=rows, freq=timeframe)
    if tz_naive:
        dates = dates.tz_localize(None)
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
            "ema20": close_vals,  # simplisitic: ema20 ≈ close
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

        # 5 hourly candles ending at 10:00 — tz-naive (like Binance/freqtrade).
        # Pretend "now" is 10:30 → period_start = 10:00.
        # Candles at 06:00-09:00 are closed; 10:00 is current (not closed).
        df = _make_ohlcv_df(5, "1h", pd.Timestamp("2025-01-15 10:00"), tz_naive=True)

        fake_now = pd.Timestamp("2025-01-15 10:30", tz="UTC")
        with patch("price_action_monitor.pd.Timestamp.utcnow", return_value=fake_now):
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
        assert params["candle_time"] == pd.Timestamp("2025-01-15 09:00")
        assert params["close"] == pytest.approx(df.iloc[3]["close"])

    def test_writes_with_tz_naive_dataframe(self):
        """Should work with tz-naive dataframe dates (like freqtrade/Binance returns).

        This is the core bug fix: pd.Timestamp.utcnow() in pandas 3 returns
        tz-aware timestamps. Adding .tz_localize(None) makes it tz-naive so
        comparison with tz-naive dataframe dates doesn't raise TypeError.
        """
        s = _make_strategy(timeframe="1h")

        # tz-naive dates — this is what freqtrade returns from Binance
        df = _make_ohlcv_df(
            5, "1h",
            pd.Timestamp("2025-01-15 10:00"),  # tz-naive
            tz_naive=True,
        )

        # pd.Timestamp.utcnow() in pandas 3 returns tz-aware like 10:30+00:00
        # .tz_localize(None) strips tz → tz-naive 10:30
        # The mock returns a tz-aware timestamp to simulate real behavior
        fake_now = pd.Timestamp("2025-01-15 10:30", tz="UTC")
        with patch("price_action_monitor.pd.Timestamp.utcnow", return_value=fake_now):
            # This would raise TypeError if .tz_localize(None) were missing
            s._persist_kline("BTC/USDT", df)

        # Verify the write succeeded
        s._pg_session_factory.assert_called_once()
        session = s._pg_session_factory.return_value.__enter__.return_value
        session.execute.assert_called_once()
        session.commit.assert_called_once()

        params = session.execute.call_args[0][1]
        assert params["symbol"] == "BTC/USDT"
        assert params["timeframe"] == "1h"

    def test_skips_when_all_candles_are_current(self):
        """If no candle is older than period_start, nothing should be written."""
        s = _make_strategy(timeframe="1h")

        # Candle at 10:00, "now" is 10:05 → period_start = 10:00
        # The candle at 10:00 is NOT < 10:00, so nothing is closed.
        df = _make_ohlcv_df(1, "1h", pd.Timestamp("2025-01-15 10:00"), tz_naive=True)

        fake_now = pd.Timestamp("2025-01-15 10:05", tz="UTC")
        with patch("price_action_monitor.pd.Timestamp.utcnow", return_value=fake_now):
            s._persist_kline("BTC/USDT", df)

        s._pg_session_factory.assert_not_called()

    def test_handles_db_exception_gracefully(self, caplog):
        """DB errors should be caught and logged, not propagated."""
        caplog.set_level(logging.WARNING)
        s = _make_strategy(timeframe="1h")
        df = _make_ohlcv_df(5, "1h", pd.Timestamp("2025-01-15 10:00"), tz_naive=True)

        # Make session.execute raise
        session = s._pg_session_factory.return_value.__enter__.return_value
        session.execute.side_effect = Exception("connection lost")

        fake_now = pd.Timestamp("2025-01-15 10:30", tz="UTC")
        with patch("price_action_monitor.pd.Timestamp.utcnow", return_value=fake_now):
            # Must not raise
            s._persist_kline("BTC/USDT", df)

        assert any("K-line write failed" in r.message for r in caplog.records)

    def test_uses_strategy_timeframe(self):
        """The timeframe in the UPSERT params must match self.timeframe."""
        s = _make_strategy(timeframe="4h")
        df = _make_ohlcv_df(5, "4h", pd.Timestamp("2025-01-15 12:00"), tz_naive=True)

        fake_now = pd.Timestamp("2025-01-15 13:00", tz="UTC")
        with patch("price_action_monitor.pd.Timestamp.utcnow", return_value=fake_now):
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


# ===================================================================
# Tests: _generate_chart
# ===================================================================


class TestGenerateChart:
    """Unit tests for PriceActionMonitor._generate_chart."""

    def test_returns_png_bytes(self):
        """Should return non-empty PNG bytes."""
        s = _make_strategy()
        df = _make_ohlcv_df(25)

        result = s._generate_chart("BTC/USDT", df)

        assert isinstance(result, bytes)
        assert len(result) > 0
        # PNG magic bytes
        assert result[:4] == b"\x89PNG"

    def test_handles_fewer_than_20_rows(self):
        """Should work with fewer than 20 candles (uses all available)."""
        s = _make_strategy()
        df = _make_ohlcv_df(5)

        result = s._generate_chart("ETH/USDT", df)

        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_uses_last_20_candles(self):
        """Should use only the last 20 candles from a larger dataframe."""
        s = _make_strategy()
        df = _make_ohlcv_df(50)

        # Just verify it doesn't crash and returns valid PNG
        result = s._generate_chart("BTC/USDT", df)
        assert result[:4] == b"\x89PNG"


# ===================================================================
# Tests: _notify_tg_bot with chart
# ===================================================================


class TestNotifyTgBotChart:
    """Tests for _notify_tg_bot multipart POST with chart."""

    def _make_signal_row(self):
        """Create a mock signal row (pd.Series)."""
        return pd.Series({
            "signal_direction": "long",
            "signal_quality": "good",
            "body_pct": 0.8,
            "close_location": 0.9,
            "body_ratio": 1.5,
            "above_ema20": True,
            "ema_gap": 0.5,
            "bull_strength_5": 0.7,
            "close": 42000.0,
            "low": 41800.0,
            "high": 42100.0,
            "is_inside": False,
            "is_engulfing": False,
            "is_surprise": False,
            "is_2k_reversal": False,
            "is_doji": False,
        })

    @patch("price_action_monitor.http_requests.post")
    @patch.object(PriceActionMonitor, "_generate_chart", return_value=b"\x89PNG_fake")
    def test_posts_multipart_with_chart(self, mock_chart, mock_post):
        """Should POST multipart with chart file and payload JSON."""
        s = _make_strategy()
        s.config = {"tg_api_url": "http://tg-bot:8090"}
        df = _make_ohlcv_df(25)
        row = self._make_signal_row()

        s._notify_tg_bot("BTC/USDT", row, df)

        # Verify chart was generated
        mock_chart.assert_called_once_with("BTC/USDT", df)

        # Verify POST was called with multipart
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "http://tg-bot:8090/signal"
        assert "files" in call_kwargs[1]
        assert "chart" in call_kwargs[1]["files"]
        assert "data" in call_kwargs[1]

        # Verify payload contains expected fields
        payload_json = call_kwargs[1]["data"]["payload"]
        payload = json.loads(payload_json)
        assert payload["symbol"] == "BTC/USDT"
        assert payload["direction"] == "long"
        assert payload["quality"] == "good"
        assert payload["entry_price"] == 42000.0
        assert payload["stop_loss"] == 41800.0

    @patch("price_action_monitor.http_requests.post", side_effect=Exception("timeout"))
    @patch.object(PriceActionMonitor, "_generate_chart", return_value=b"\x89PNG_fake")
    def test_handles_http_error_gracefully(self, mock_chart, mock_post):
        """HTTP errors should be caught, not propagated."""
        s = _make_strategy()
        s.config = {"tg_api_url": "http://tg-bot:8090"}
        df = _make_ohlcv_df(25)
        row = self._make_signal_row()

        # Must not raise
        s._notify_tg_bot("BTC/USDT", row, df)

    @patch("price_action_monitor.http_requests.post")
    @patch.object(PriceActionMonitor, "_generate_chart", return_value=b"\x89PNG_fake")
    def test_posts_to_default_url_when_no_config(self, mock_chart, mock_post):
        """Should POST to default URL when tg_api_url is not in config."""
        s = _make_strategy()
        s.config = {}  # no tg_api_url
        df = _make_ohlcv_df(25)
        row = self._make_signal_row()

        s._notify_tg_bot("BTC/USDT", row, df)

        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == "http://tg-bot:8090/signal"


# ===================================================================
# Tests: _check_and_notify passes dataframe
# ===================================================================


class TestCheckAndNotifyDataframe:
    """Verify _check_and_notify passes dataframe through to _notify_tg_bot."""

    @patch.object(PriceActionMonitor, "_notify_tg_bot")
    @patch.object(PriceActionMonitor, "_save_signal")
    def test_passes_dataframe_to_notify(self, mock_save, mock_notify):
        """_check_and_notify should pass dataframe to _notify_tg_bot."""
        s = _make_strategy()
        df = _make_ohlcv_df(25)
        # Make the last row a "good long" signal that passes EMA filter
        last = df.iloc[-1].copy()
        last["signal_quality"] = "good"
        last["signal_direction"] = "long"
        last["above_ema20"] = True
        last["bull_strength_5"] = 0.8

        s._check_and_notify("BTC/USDT", last, df)

        mock_notify.assert_called_once()
        # Third argument should be the dataframe
        assert mock_notify.call_args[0][2] is df
