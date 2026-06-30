"""
Unit tests for PriceActionMonitor._persist_kline and PaKline model.

Run from repo root with the project venv:
  .venv/bin/pytest tests/price_action/test_price_action_monitor.py -v
"""
import json
import logging
import os
import sys
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
from price_action.models import PaAnalysis  # noqa: E402
from price_action.rules import PriceActionSignalRules  # noqa: E402


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


class TestPaAnalysisModel:
    """Verify full PA_Agent-style analysis record fields."""

    def test_usage_and_exception_columns_exist(self):
        col_names = {c.name for c in PaAnalysis.__table__.columns}
        assert "usage_total" in col_names
        assert "exception" in col_names


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
    """Verify _persist_kline + PA submission flow from populate_entry_trend."""

    def test_calls_persist_kline_in_dry_run(self):
        """In dry_run mode, _persist_kline should be called."""
        s = _make_strategy()
        # Mock dp to report dry_run
        s.dp = MagicMock()
        s.dp.runmode.value = "dry_run"

        df = _make_ohlcv_df(5)

        with patch.object(s, "_persist_kline") as mock_persist:
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

        with patch.object(s, "_persist_kline"):
            result = s.populate_entry_trend(df, {"pair": "BTC/USDT"})

        assert result is df

    def test_submits_pa_analysis_when_worker_present(self):
        """In dry_run mode with a worker, _submit_pa_analysis should be invoked."""
        s = _make_strategy()
        s.dp = MagicMock()
        s.dp.runmode.value = "dry_run"
        s._worker = MagicMock()
        s._worker.submit.return_value = True

        df = _make_ohlcv_df(5)

        with patch.object(s, "_persist_kline"):
            s.populate_entry_trend(df, {"pair": "BTC/USDT"})

        s._worker.submit.assert_called_once()
        call_kwargs = s._worker.submit.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC/USDT"
        assert call_kwargs["timeframe"] == "1h"

    def test_does_not_submit_without_worker(self):
        """No worker (pipeline disabled) → submit must not be attempted."""
        s = _make_strategy()
        s.dp = MagicMock()
        s.dp.runmode.value = "dry_run"
        s._worker = None

        df = _make_ohlcv_df(5)

        with patch.object(s, "_persist_kline"):
            s.populate_entry_trend(df, {"pair": "BTC/USDT"})
        # No exception, no submission path exercised (worker is None)


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
# Tests: SignalNotifier.notify_decision (trade decision -> /decision)
# ===================================================================


class TestNotifyDecision:
    """Tests for the LLM trade-decision notification path (tg-bot /decision)."""

    def _make_decision_payload(self):
        return {
            "price_action_features": {"candle_time": "2026-06-01T10:00:00"},
            "diagnosis": {},
            "selected_strategies": [],
            "decision": {
                "decision": {
                    "order_direction": "做多",
                    "order_type": "突破单",
                    "entry_price": 42000.0,
                    "stop_loss_price": 41800.0,
                    "take_profit_price": 42400.0,
                    "take_profit_price_2": None,
                    "trade_confidence": 70,
                    "reasoning": "EMA20 pullback reversal",
                },
                "decision_trace": [
                    {"node_id": "10.3", "reason": "trend up"},
                    {"node_id": "11.1", "reason": "strong close"},
                ],
            },
            "validation": {"valid": True},
        }

    @patch("price_action.notification.http_requests.post")
    def test_posts_decision_to_decision_endpoint(self, mock_post):
        """notify_decision should POST the trade decision to /decision with chart."""
        s = _make_strategy()
        s.config = {"tg_api_url": "http://tg-bot:8090"}
        df = _make_ohlcv_df(25)
        notifier = s._get_notifier()
        payload = self._make_decision_payload()

        notifier.notify_decision(
            "BTC/USDT",
            df,
            payload,
            chart_generator=lambda *a, **kw: b"\x89PNG_fake",
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://tg-bot:8090/decision"
        assert "files" in call_args[1]
        assert "chart" in call_args[1]["files"]
        body = json.loads(call_args[1]["data"]["payload"])
        assert body["symbol"] == "BTC/USDT"
        assert body["decision_type"] == "enter_long"
        assert body["entry"] == 42000.0
        assert body["stop_loss"] == 41800.0
        assert body["risk_reward"] == 2.0

    @patch("price_action.notification.http_requests.post", side_effect=Exception("timeout"))
    def test_handles_http_error_gracefully(self, mock_post):
        """HTTP errors should be caught, not propagated."""
        s = _make_strategy()
        s.config = {"tg_api_url": "http://tg-bot:8090"}
        df = _make_ohlcv_df(25)
        notifier = s._get_notifier()
        payload = self._make_decision_payload()

        # Must not raise
        notifier.notify_decision(
            "BTC/USDT", df, payload, chart_generator=lambda *a, **kw: b"\x89PNG_fake"
        )

    @patch("price_action.notification.http_requests.post")
    def test_posts_decision_without_chart_as_form_payload(self, mock_post):
        """If chart rendering fails, notifier still sends the decision payload."""
        s = _make_strategy()
        s.config = {"tg_api_url": "http://tg-bot:8090"}
        df = _make_ohlcv_df(25)
        notifier = s._get_notifier()
        payload = self._make_decision_payload()

        def raise_chart_error(*args, **kwargs):
            raise RuntimeError("chart failed")

        notifier.notify_decision(
            "BTC/USDT",
            df,
            payload,
            chart_generator=raise_chart_error,
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://tg-bot:8090/decision"
        assert "files" not in call_args[1]
        body = json.loads(call_args[1]["data"]["payload"])
        assert body["symbol"] == "BTC/USDT"
        assert body["decision_type"] == "enter_long"


# ===================================================================
# Tests: structure context filters
# ===================================================================


class TestStructureContextRules:
    """Unit tests for PA_Agent-inspired deterministic context filters.

    These now exercise the rules engine (PriceActionSignalRules) directly,
    since the strategy no longer wraps the signal-bar flow (replaced by the
    LLM PA analysis pipeline). The rules engine is still used by repository/notifier.
    """

    def _rules(self):
        return PriceActionSignalRules()

    def _base_row(self, **overrides):
        row = {
            "signal_quality": "acceptable",
            "signal_direction": "long",
            "above_ema20": True,
            "ema_gap": 0.5,
            "bull_strength_5": 0.7,
            "is_barbwire": False,
            "range_zone": "unknown",
            "is_inside": False,
            "is_engulfing": False,
            "is_surprise": False,
            "is_2k_reversal": False,
            "is_ioi": False,
            "inside_sequence": "none",
            "micro_double": "none",
            "breakout_prev_5": "none",
        }
        row.update(overrides)
        return pd.Series(row)

    def test_barbwire_filters_even_good_signal(self):
        rules = self._rules()
        row = self._base_row(signal_quality="good", is_barbwire=True)

        assert rules.candidate_signal_ok(row) is False

    def test_middle_range_filters_plain_acceptable_signal(self):
        rules = self._rules()
        row = self._base_row(signal_quality="acceptable", range_zone="middle")

        assert rules.candidate_signal_ok(row) is False

    def test_middle_range_allows_acceptable_with_breakout_context(self):
        rules = self._rules()
        row = self._base_row(
            signal_quality="acceptable",
            range_zone="middle",
            breakout_prev_5="up",
        )

        assert rules.candidate_signal_ok(row) is True

    def test_fair_signal_allowed_at_range_edge(self):
        rules = self._rules()
        row = self._base_row(signal_quality="fair", range_zone="lower")

        assert rules.candidate_signal_ok(row) is True

    def test_fair_signal_filtered_in_middle_without_context(self):
        rules = self._rules()
        row = self._base_row(signal_quality="fair", range_zone="middle")

        assert rules.candidate_signal_ok(row) is False

    def test_detects_inside_sequences_ioi_micro_double_and_breakout(self):
        rules = self._rules()
        df = pd.DataFrame({
            "open": [5.0, 4.0, 5.0, 6.0, 5.5, 7.0],
            "high": [10.0, 9.0, 11.0, 10.0, 10.0, 12.0],
            "low": [0.0, 1.0, -1.0, 0.0, 0.0, -2.0],
            "close": [5.0, 6.0, 4.0, 7.0, 6.0, 11.0],
        })
        df["volume"] = 100.0
        df = rules.calc_basic_indicators(df)
        df["atr14"] = 10.0
        df = rules.detect_special_bars(df)
        df = rules.calc_structure_context(df)

        assert bool(df.loc[3, "is_ioi"]) is True
        assert bool(df.loc[3, "is_inside"]) is True
        assert df.loc[4, "inside_sequence"] == "ii"
        assert bool(df.loc[4, "is_mdb"]) is True
        assert bool(df.loc[4, "is_mdt"]) is True
        assert df.loc[5, "breakout_prev_5"] == "both"

        tags = rules.get_bar_types(df.loc[5])
        assert "breakout_up" in tags
        assert "breakout_down" in tags


# ===================================================================
# Tests: _market attribute derivation from config
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
    """Unit tests for EMA20 crossover detection (PriceActionSignalRules).

    The strategy no longer wraps this; tested directly on the rules engine.
    """

    def _rules(self):
        return PriceActionSignalRules()

    def test_cross_up(self):
        """prev_close < prev_ema20 and curr_close > curr_ema20 → long."""
        df = _make_ema_df(
            closes=[95.0, 105.0],   # 95 < 100, 105 > 100
            ema_vals=[100.0, 100.0],
        )
        self._rules().detect_ema20_cross(df)
        assert df.loc[1, "ema20_cross"] == "long"

    def test_cross_down(self):
        """prev_close > prev_ema20 and curr_close < curr_ema20 → short."""
        df = _make_ema_df(
            closes=[105.0, 95.0],   # 105 > 100, 95 < 100
            ema_vals=[100.0, 100.0],
        )
        self._rules().detect_ema20_cross(df)
        assert df.loc[1, "ema20_cross"] == "short"

    def test_no_cross_when_above_both(self):
        """prev above and curr above → none."""
        df = _make_ema_df(
            closes=[105.0, 110.0],
            ema_vals=[100.0, 100.0],
        )
        self._rules().detect_ema20_cross(df)
        assert df.loc[1, "ema20_cross"] == "none"

    def test_no_cross_when_below_both(self):
        """prev below and curr below → none."""
        df = _make_ema_df(
            closes=[95.0, 90.0],
            ema_vals=[100.0, 100.0],
        )
        self._rules().detect_ema20_cross(df)
        assert df.loc[1, "ema20_cross"] == "none"

    def test_first_row_always_none(self):
        """First row has no prev (NaN from shift), should not trigger a cross."""
        df = _make_ema_df(
            closes=[105.0, 95.0],
            ema_vals=[100.0, 100.0],
        )
        self._rules().detect_ema20_cross(df)
        # First row cannot cross — no previous bar
        assert df.loc[0, "ema20_cross"] == "none"

    def test_multi_bar_sequence(self):
        """A sequence with a clear up-cross then later a down-cross."""
        closes = [95.0, 95.0, 105.0, 110.0, 95.0]
        ema_vals = [100.0] * 5
        df = _make_ema_df(closes, ema_vals)
        self._rules().detect_ema20_cross(df)
        # row 2: prev 95<100, curr 105>100 → long
        # row 4: prev 110>100, curr 95<100 → short
        assert df.loc[2, "ema20_cross"] == "long"
        assert df.loc[4, "ema20_cross"] == "short"
        # row 1 (95→95): both below → none
        assert df.loc[1, "ema20_cross"] == "none"
        # row 3 (105→110): both above → none
        assert df.loc[3, "ema20_cross"] == "none"
