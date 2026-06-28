"""
Unit tests for PriceActionMonitor._persist_kline and PaKline model.

Run from repo root with the project venv:
  .venv/bin/pytest tests/price_action/test_price_action_monitor.py -v
"""
import json
import logging
import os
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

# price_action_monitor.py lives in user_data/strategies/, which is outside the
# freqtrade package and not on sys.path by default.  Add it so the tests run
# from any cwd. (tests/price_action/ -> ../../ = repo root)
_STRAT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "user_data", "strategies")
)
if _STRAT_DIR not in sys.path:
    sys.path.insert(0, _STRAT_DIR)

from price_action_monitor import (  # noqa: E402
    PaKline,
    PriceActionMonitor,
    WatchPair,
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
    # Default: query().filter_by().first() returns None (no rows)
    mock_session.query.return_value.filter_by.return_value.first.return_value = None
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
# Tests: WatchPair model
# ===================================================================


class TestWatchPairModel:
    """Verify the WatchPair ORM model includes display_name column."""

    def test_table_name(self):
        assert WatchPair.__tablename__ == "watch_pair"

    def test_display_name_column_exists(self):
        col_names = {c.name for c in WatchPair.__table__.columns}
        assert "display_name" in col_names

    def test_display_name_is_nullable(self):
        col = {c.name: c for c in WatchPair.__table__.columns}["display_name"]
        assert col.nullable is True


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

    def test_writes_all_strategy_candles_batch(self):
        """Should batch-UPSERT every candle Freqtrade passes to the strategy."""
        s = _make_strategy(timeframe="1h")

        # 5 hourly candles ending at 10:00 — tz-naive (like Binance/freqtrade).
        # Freqtrade has already dropped incomplete exchange candles before analysis.
        df = _make_ohlcv_df(5, "1h", pd.Timestamp("2025-01-15 10:00"), tz_naive=True)

        s._persist_kline("BTC/USDT", df)

        # session factory was entered once
        s._pg_session_factory.assert_called_once()
        session = s._pg_session_factory.return_value.__enter__.return_value
        session.execute.assert_called_once()
        session.commit.assert_called_once()

        # Verify the params passed to session.execute — now a list of row dicts
        call_args = session.execute.call_args
        sql_text = call_args[0][0].text  # sqlalchemy.text() stores SQL in .text
        assert "INSERT INTO pa_kline" in sql_text
        assert "ON CONFLICT" in sql_text

        params = call_args[0][1]
        assert isinstance(params, list)
        assert len(params) == 5
        for row in params:
            assert row["symbol"] == "BTC/USDT"
            assert row["timeframe"] == "1h"
        candle_times = [r["candle_time"] for r in params]
        assert max(candle_times) == pd.Timestamp("2025-01-15 10:00")
        # Close of the newest row matches the source dataframe
        newest = max(params, key=lambda r: r["candle_time"])
        assert newest["close"] == pytest.approx(df.iloc[4]["close"])

    def test_writes_with_tz_naive_dataframe(self):
        """Should work with tz-naive dataframe dates (like freqtrade/Binance returns).

        Regression guard: batch UPSERT must not break on tz-naive dates.
        """
        s = _make_strategy(timeframe="1h")

        # tz-naive dates — this is what freqtrade returns from Binance
        df = _make_ohlcv_df(
            5, "1h",
            pd.Timestamp("2025-01-15 10:00"),  # tz-naive
            tz_naive=True,
        )

        s._persist_kline("BTC/USDT", df)

        # Verify the write succeeded
        s._pg_session_factory.assert_called_once()
        session = s._pg_session_factory.return_value.__enter__.return_value
        session.execute.assert_called_once()
        session.commit.assert_called_once()

        params = session.execute.call_args[0][1]
        assert isinstance(params, list)
        assert len(params) == 5
        for row in params:
            assert row["symbol"] == "BTC/USDT"
            assert row["timeframe"] == "1h"

    def test_writes_with_tz_aware_dataframe(self):
        """A-share sources (Sina/Tencent) return tz-aware UTC dates; the batch
        UPSERT must normalize them to tz-naive datetimes without raising.

        This is the 2026-06-25 fix: Sina 1h candles carry half-hour timestamps
        (02:30/03:30/06:00/07:00 UTC); they must all be persisted as-is.
        """
        s = _make_strategy(timeframe="1h")

        # tz-aware dates — this is what A-share sources return
        df = _make_ohlcv_df(
            5, "1h",
            pd.Timestamp("2025-01-15 10:00", tz="UTC"),
            tz_naive=False,
        )

        # Must not raise even though dates are tz-aware
        s._persist_kline("515050/SH", df)

        s._pg_session_factory.assert_called_once()
        session = s._pg_session_factory.return_value.__enter__.return_value
        params = session.execute.call_args[0][1]
        assert isinstance(params, list)
        assert len(params) == 5
        # candle_time params must be UTC tz-naive datetimes (DB column has no tz)
        for row in params:
            assert row["symbol"] == "515050/SH"
            ct = row["candle_time"]
            assert ct is not None
            assert ct.tzinfo is None

    def test_writes_single_strategy_candle(self):
        """A single strategy candle is still a valid analyzed candle."""
        s = _make_strategy(timeframe="1h")

        df = _make_ohlcv_df(1, "1h", pd.Timestamp("2025-01-15 10:00"), tz_naive=True)

        s._persist_kline("BTC/USDT", df)

        s._pg_session_factory.assert_called_once()
        session = s._pg_session_factory.return_value.__enter__.return_value
        params = session.execute.call_args[0][1]
        assert len(params) == 1
        assert params[0]["candle_time"] == pd.Timestamp("2025-01-15 10:00")

    def test_handles_db_exception_gracefully(self, caplog):
        """DB errors should be caught and logged, not propagated."""
        caplog.set_level(logging.WARNING)
        s = _make_strategy(timeframe="1h")
        df = _make_ohlcv_df(5, "1h", pd.Timestamp("2025-01-15 10:00"), tz_naive=True)

        # Make session.execute raise
        session = s._pg_session_factory.return_value.__enter__.return_value
        session.execute.side_effect = Exception("connection lost")

        # Must not raise
        s._persist_kline("BTC/USDT", df)

        assert any("K-line write failed" in r.message for r in caplog.records)

    def test_uses_strategy_timeframe(self):
        """The timeframe in the UPSERT params must match self.timeframe."""
        s = _make_strategy(timeframe="4h")
        df = _make_ohlcv_df(5, "4h", pd.Timestamp("2025-01-15 12:00"), tz_naive=True)

        s._persist_kline("ETH/USDT", df)

        session = s._pg_session_factory.return_value.__enter__.return_value
        params = session.execute.call_args[0][1]
        assert isinstance(params, list)
        for row in params:
            assert row["timeframe"] == "4h"
            assert row["symbol"] == "ETH/USDT"


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
        mock_chart.assert_called_once_with("BTC/USDT", "1h", df)

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
    def test_candidate_waits_without_save_or_immediate_notify(self, mock_save, mock_notify):
        """A fresh signal bar waits for follow-through without polluting signal history."""
        s = _make_strategy()
        df = _make_ohlcv_df(25)
        # Make the last row a "good long" signal that passes EMA filter
        last = df.iloc[-1].copy()
        last["signal_quality"] = "good"
        last["signal_direction"] = "long"
        last["above_ema20"] = True
        last["bull_strength_5"] = 0.8

        s._check_and_notify("BTC/USDT", last, df)

        mock_save.assert_not_called()
        mock_notify.assert_not_called()

    @patch.object(PriceActionMonitor, "_notify_tg_bot")
    @patch.object(PriceActionMonitor, "_save_signal", return_value=True)
    def test_confirmed_candidate_passes_dataframe_to_notify(self, mock_save, mock_notify):
        """A prior candidate with follow-through should be saved as confirmed and notified."""
        s = _make_strategy()
        df = _make_ohlcv_df(25)
        df["signal_quality"] = "none"
        df["signal_direction"] = "none"
        idx = len(df) - 2
        df.loc[idx, "open"] = 100.0
        df.loc[idx, "high"] = 102.0
        df.loc[idx, "low"] = 99.0
        df.loc[idx, "close"] = 101.8
        df.loc[idx, "signal_quality"] = "good"
        df.loc[idx, "signal_direction"] = "long"
        df.loc[idx, "above_ema20"] = True
        df.loc[idx, "bull_strength_5"] = 0.8
        df.loc[idx, "ema_gap"] = 0.5
        df.loc[idx + 1, "high"] = 103.0
        df.loc[idx + 1, "low"] = 101.0
        df.loc[idx + 1, "close"] = 102.5

        s._check_and_notify("BTC/USDT", df.iloc[-1], df)

        mock_notify.assert_called_once()
        # Third argument should be the dataframe
        assert mock_notify.call_args[0][2] is df
        assert mock_save.call_args.kwargs["signal_type"] == "confirmed_signal_bar_good"

    @patch.object(PriceActionMonitor, "_notify_tg_bot")
    @patch.object(PriceActionMonitor, "_save_signal", return_value=True)
    def test_untriggered_candidate_is_not_confirmed(self, mock_save, mock_notify):
        """A candidate without a break of its signal-bar high/low should not notify."""
        s = _make_strategy()
        df = _make_ohlcv_df(25)
        df["signal_quality"] = "none"
        df["signal_direction"] = "none"
        idx = len(df) - 2
        df.loc[idx, "open"] = 100.0
        df.loc[idx, "high"] = 102.0
        df.loc[idx, "low"] = 99.0
        df.loc[idx, "close"] = 101.8
        df.loc[idx, "signal_quality"] = "good"
        df.loc[idx, "signal_direction"] = "long"
        df.loc[idx, "above_ema20"] = True
        df.loc[idx, "bull_strength_5"] = 0.8
        df.loc[idx, "ema_gap"] = 0.5
        df.loc[idx + 1, "high"] = 101.9
        df.loc[idx + 1, "low"] = 100.5
        df.loc[idx + 1, "close"] = 101.0

        s._check_and_notify("BTC/USDT", df.iloc[-1], df)

        mock_save.assert_not_called()
        mock_notify.assert_not_called()


# ===================================================================
# Tests: _market attribute and _save_signal market field
# ===================================================================


class TestMarketAttribute:
    """Verify that self._market is correctly derived from config and
    passed through to saved signals."""

    def test_default_market_is_crypto(self):
        """__init__ should set _market = 'crypto' as default."""
        s = PriceActionMonitor(config={})
        assert s._market == "crypto"

    def test_ashare_config_sets_market(self):
        """When config exchange.name is 'ashare', _init_default_pairs should set
        self._market = 'ashare'."""
        s = PriceActionMonitor(config={"exchange": {"name": "ashare", "pair_whitelist": []}})
        s.timeframe = "1h"
        # Mock PG session so _init_default_pairs doesn't fail
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_session)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        s._pg_session_factory = MagicMock(return_value=mock_ctx)
        # Need to stub the ashare name import
        with patch.dict("sys.modules", {"freqtrade.exchange.ashare": MagicMock()}):
            s._init_default_pairs()
        assert s._market == "ashare"

    def test_crypto_config_sets_market(self):
        """When config exchange.name is 'okx' (or missing), _market stays 'crypto'."""
        s = PriceActionMonitor(config={"exchange": {"name": "okx"}})
        s.timeframe = "1h"
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_session)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        s._pg_session_factory = MagicMock(return_value=mock_ctx)
        s._init_default_pairs()
        assert s._market == "crypto"

    def test_empty_config_market_stays_crypto(self):
        """When no exchange config, _market should remain default 'crypto'."""
        s = PriceActionMonitor(config={})
        s.timeframe = "1h"
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_session)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        s._pg_session_factory = MagicMock(return_value=mock_ctx)
        s._init_default_pairs()
        assert s._market == "crypto"

    def test_save_signal_uses_self_market(self):
        """_save_signal should use self._market, not hardcoded 'crypto'."""
        s = _make_strategy()
        # Simulate an ashare container
        s._market = "ashare"
        s.timeframe = "1h"

        row = pd.Series({
            "signal_quality": "good",
            "signal_direction": "long",
            "body_pct": 0.8,
            "close_location": 0.9,
            "body_ratio": 1.5,
            "above_ema20": True,
            "ema_gap": 0.5,
            "bull_strength_5": 0.7,
            "close": 42000.0,
            "low": 41800.0,
            "high": 42100.0,
            "open": 41900.0,
            "volume": 100.0,
            "ema20": 41950.0,
            "atr14": 200.0,
            "ema20_position": 0.6,
            "is_inside": False,
            "is_engulfing": False,
            "is_surprise": False,
            "is_2k_reversal": False,
            "is_doji": False,
            "date": pd.Timestamp("2025-06-18 10:00", tz="UTC"),
        })

        s._save_signal("588290/SH", row, "test_reason")

        # Get the PaSignal object added to the mock session
        session = s._pg_session_factory.return_value.__enter__.return_value
        added_signal = session.add.call_args[0][0]
        assert added_signal.market == "ashare"

    def test_save_signal_crypto_market(self):
        """_save_signal with default crypto market should save market='crypto'."""
        s = _make_strategy()
        # Default is crypto, no need to set
        assert s._market == "crypto"

        row = pd.Series({
            "signal_quality": "good",
            "signal_direction": "short",
            "body_pct": 0.6,
            "close_location": 0.2,
            "body_ratio": 0.8,
            "above_ema20": False,
            "ema_gap": -0.3,
            "bull_strength_5": 0.3,
            "close": 50000.0,
            "low": 49500.0,
            "high": 50500.0,
            "open": 50200.0,
            "volume": 200.0,
            "ema20": 50100.0,
            "atr14": 500.0,
            "ema20_position": 0.4,
            "is_inside": False,
            "is_engulfing": False,
            "is_surprise": False,
            "is_2k_reversal": False,
            "is_doji": False,
            "date": pd.Timestamp("2025-06-18 10:00", tz="UTC"),
        })

        s._save_signal("BTC/USDT", row, "test_reason")

        session = s._pg_session_factory.return_value.__enter__.return_value
        added_signal = session.add.call_args[0][0]
        assert added_signal.market == "crypto"


# ===================================================================
# Tests: _detect_ema20_cross
# ===================================================================


def _make_ema_df(closes, ema_vals):
    """Build a minimal df with close + ema20 columns for cross detection."""
    return pd.DataFrame({
        "close": closes,
        "ema20": ema_vals,
    })


class TestDetectEma20Cross:
    """Unit tests for EMA20 crossover detection."""

    def test_cross_up(self):
        """prev_close < prev_ema20 and curr_close > curr_ema20 → long."""
        df = _make_ema_df(
            closes=[95.0, 105.0],   # 95 < 100, 105 > 100
            ema_vals=[100.0, 100.0],
        )
        s = _make_strategy()
        s._detect_ema20_cross(df)
        assert df.loc[1, "ema20_cross"] == "long"

    def test_cross_down(self):
        """prev_close > prev_ema20 and curr_close < curr_ema20 → short."""
        df = _make_ema_df(
            closes=[105.0, 95.0],   # 105 > 100, 95 < 100
            ema_vals=[100.0, 100.0],
        )
        s = _make_strategy()
        s._detect_ema20_cross(df)
        assert df.loc[1, "ema20_cross"] == "short"

    def test_no_cross_when_above_both(self):
        """prev above and curr above → none."""
        df = _make_ema_df(
            closes=[105.0, 110.0],
            ema_vals=[100.0, 100.0],
        )
        s = _make_strategy()
        s._detect_ema20_cross(df)
        assert df.loc[1, "ema20_cross"] == "none"

    def test_no_cross_when_below_both(self):
        """prev below and curr below → none."""
        df = _make_ema_df(
            closes=[95.0, 90.0],
            ema_vals=[100.0, 100.0],
        )
        s = _make_strategy()
        s._detect_ema20_cross(df)
        assert df.loc[1, "ema20_cross"] == "none"

    def test_first_row_always_none(self):
        """First row has no prev (NaN from shift), should not trigger a cross."""
        df = _make_ema_df(
            closes=[105.0, 95.0],
            ema_vals=[100.0, 100.0],
        )
        s = _make_strategy()
        s._detect_ema20_cross(df)
        # First row cannot cross — no previous bar
        assert df.loc[0, "ema20_cross"] == "none"

    def test_multi_bar_sequence(self):
        """A sequence with a clear up-cross then later a down-cross."""
        closes = [95.0, 95.0, 105.0, 110.0, 95.0]
        ema_vals = [100.0] * 5
        df = _make_ema_df(closes, ema_vals)
        s = _make_strategy()
        s._detect_ema20_cross(df)
        # row 2: prev 95<100, curr 105>100 → long
        # row 4: prev 110>100, curr 95<100 → short
        assert df.loc[2, "ema20_cross"] == "long"
        assert df.loc[4, "ema20_cross"] == "short"
        # row 1 (95→95): both below → none
        assert df.loc[1, "ema20_cross"] == "none"
        # row 3 (105→110): both above → none
        assert df.loc[3, "ema20_cross"] == "none"
