"""Tests for the PA_Agent-style L1-L4 pipeline modules."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


_STRAT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "user_data", "strategies")
)
if _STRAT_DIR not in sys.path:
    sys.path.insert(0, _STRAT_DIR)

# Stub freqtrade.strategy so price_action_monitor can import without a full
# freqtrade install. Provide a concrete IStrategy base (with a real __init__)
# so PriceActionMonitor subclasses work and attribute access behaves normally.
# talib is expected to be installed (Docker) or stubbed by the conftest;
# rules.py imports it eagerly via repository.
if "freqtrade" not in sys.modules:
    sys.modules["freqtrade"] = MagicMock()


class _FakeIStrategy:
    """Minimal concrete IStrategy base for strategy wiring tests."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.dp = None
        self.timeframe = "1h"


if "freqtrade.strategy" not in sys.modules:
    _ft_strategy_mod = MagicMock()
    _ft_strategy_mod.IStrategy = _FakeIStrategy
    sys.modules["freqtrade.strategy"] = _ft_strategy_mod

from price_action.features import (  # noqa: E402
    build_l1_features,
    calculate_atr,
    calculate_ema,
)
from price_action.llm import OpenAIJsonClient  # noqa: E402
from price_action.repository import PriceActionRepository  # noqa: E402
from price_action.validation import DecisionValidator  # noqa: E402
from price_action.worker import PaAnalysisWorker  # noqa: E402


def _df_from_ohlc(ohlc, start="2026-06-01 08:00", freq="1h"):
    dates = pd.date_range(start=start, periods=len(ohlc), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [x[0] for x in ohlc],
            "high": [x[1] for x in ohlc],
            "low": [x[2] for x in ohlc],
            "close": [x[3] for x in ohlc],
            "volume": 1000.0,
        }
    )


class TestL1Features:
    def test_latest_closed_k_is_k1_and_forming_tail_is_dropped(self):
        df = _df_from_ohlc(
            [
                (100, 102, 99, 101),
                (101, 103, 100, 102),
                (102, 104, 101, 103),
                (103, 105, 102, 104),
            ],
            start="2026-06-01 08:00",
        )

        result = build_l1_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=4,
            warmup=0,
            now=pd.Timestamp("2026-06-01 11:30", tz="UTC"),
        )

        assert result.latest_features["k"] == "K1"
        assert result.latest_features["time"] == "2026-06-01T10:00:00"
        assert "2026-06-01T11:00:00" not in result.kline_table

    def test_ema_uses_sma_seed(self):
        closes = pd.Series(range(1, 22), dtype=float)

        ema = calculate_ema(closes, period=20)

        assert pd.isna(ema.iloc[18])
        assert ema.iloc[19] == pytest.approx(10.5)
        assert ema.iloc[20] == pytest.approx(11.5)

    def test_atr_uses_wilder_seed(self):
        df = pd.DataFrame(
            {
                "high": [11.0] * 15,
                "low": [9.0] * 15,
                "close": [10.0] * 15,
            }
        )

        atr = calculate_atr(df, period=14)

        assert pd.isna(atr.iloc[12])
        assert atr.iloc[13] == pytest.approx(2.0)
        assert atr.iloc[14] == pytest.approx(2.0)

    def test_inside_sequences_are_reported_newest_first(self):
        df = _df_from_ohlc(
            [
                (5, 10, 0, 6),
                (5, 9, 1, 6),
                (5, 8, 2, 6),
                (5, 7, 3, 6),
            ]
        )

        result = build_l1_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=4,
            warmup=0,
            now=pd.Timestamp("2026-06-01 12:30", tz="UTC"),
        )

        assert result.latest_features["inside"] is True
        assert result.latest_features["inside_sequence"] == "iii"
        assert "iii" in result.latest_features["patterns"]

    def test_ashare_daily_uses_local_market_close(self):
        df = _df_from_ohlc(
            [
                (1.00, 1.05, 0.98, 1.03),
                (1.03, 1.08, 1.02, 1.06),
            ],
            start="2026-06-01",
            freq="D",
        )

        result = build_l1_features(
            df,
            symbol="588290/SH",
            timeframe="1d",
            market="ashare",
            window=2,
            warmup=0,
            now=pd.Timestamp("2026-06-02 15:30", tz="Asia/Shanghai"),
        )

        assert result.latest_features["time"] == "2026-06-02T00:00:00"


class TestOpenAIJsonClient:
    def test_uses_json_object_response_format(self):
        completions = MagicMock()
        completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        client = OpenAIJsonClient(
            base_url="https://api.deepseek.com",
            api_key="test",
            model="deepseek-chat",
            client=fake_client,
        )

        content = client.complete_json([{"role": "user", "content": "x"}], stage="test")

        assert json.loads(content) == {"ok": True}
        kwargs = completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["model"] == "deepseek-chat"


class TestDecisionValidator:
    diagnosis = {"stage": "market_diagnosis"}
    l1 = {
        "high": 105.0,
        "low": 95.0,
        "atr14": 2.0,
        "gate_break": "none",
        "atr_expand_ratio": 1.0,
    }

    def test_json_syntax_stops_first(self):
        parsed, result = DecisionValidator().validate(
            "{bad",
            diagnosis=self.diagnosis,
            l1_features=self.l1,
        )

        assert parsed is None
        assert result.checks == ["json_syntax"]

    def test_stage_consistency_runs_before_semantic(self):
        raw = json.dumps(
            {
                "stage": "wrong",
                "decision": {
                    "type": "enter_long",
                    "direction": "long",
                    "entry": 100,
                    "stop_loss": 110,
                    "take_profit_1": 120,
                    "confidence": 2,
                },
            }
        )

        _, result = DecisionValidator().validate(raw, diagnosis=self.diagnosis, l1_features=self.l1)

        assert result.checks == ["json_syntax", "stage_consistency"]
        assert "stage_must_be_trade_decision" in result.errors

    def test_semantic_runs_before_numeric(self):
        raw = json.dumps(
            {
                "stage": "trade_decision",
                "decision": {
                    "type": "enter_long",
                    "direction": "long",
                    "entry": 100,
                    "stop_loss": 110,
                    "take_profit_1": 120,
                    "confidence": 2,
                },
            }
        )

        _, result = DecisionValidator().validate(raw, diagnosis=self.diagnosis, l1_features=self.l1)

        assert result.checks == [
            "json_syntax",
            "stage_consistency",
            "semantic_reasonableness",
        ]
        assert "long_stop_must_be_below_entry" in result.errors

    def test_numeric_range_runs_last(self):
        raw = json.dumps(
            {
                "stage": "trade_decision",
                "decision": {
                    "type": "enter_long",
                    "direction": "long",
                    "entry": 100,
                    "stop_loss": 95,
                    "take_profit_1": 110,
                    "risk_reward": 2,
                    "confidence": 2,
                },
            }
        )

        _, result = DecisionValidator().validate(raw, diagnosis=self.diagnosis, l1_features=self.l1)

        assert result.checks == [
            "json_syntax",
            "stage_consistency",
            "semantic_reasonableness",
            "numeric_range",
        ]
        assert "confidence_must_be_0_to_1" in result.errors

    def test_atr_expansion_veto_is_semantic(self):
        l1 = {**self.l1, "gate_break": "up", "atr_expand_ratio": 2.1}
        raw = json.dumps(
            {
                "stage": "trade_decision",
                "decision": {
                    "type": "enter_long",
                    "direction": "long",
                    "entry": 100,
                    "stop_loss": 95,
                    "take_profit_1": 110,
                    "risk_reward": 2,
                    "confidence": 0.7,
                },
            }
        )

        _, result = DecisionValidator().validate(raw, diagnosis=self.diagnosis, l1_features=l1)

        assert result.checks[-1] == "semantic_reasonableness"
        assert "atr_expansion_over_2x_vetoes_breakout_entry" in result.errors


class TestPriceActionRepository:
    def test_save_analysis_uses_semantic_analysis_fields(self):
        session = MagicMock()
        session.query.return_value.filter_by.return_value.first.return_value = None
        ctx = MagicMock()
        ctx.__enter__.return_value = session
        ctx.__exit__.return_value = False
        factory = MagicMock(return_value=ctx)
        repository = PriceActionRepository(factory, timeframe="1h", market="crypto")

        saved = repository.save_analysis(
            market="crypto",
            symbol="BTC/USDT",
            timeframe="1h",
            candle_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            status="success",
            market_diagnosis={"stage": "market_diagnosis"},
            trade_decision={"stage": "trade_decision"},
            raw_responses={
                "market_diagnosis": '{"stage":"market_diagnosis"}',
                "trade_decision": '{"stage":"trade_decision"}',
            },
        )

        assert saved is True
        row = session.add.call_args.args[0]
        assert row.market_diagnosis == {"stage": "market_diagnosis"}
        assert row.trade_decision == {"stage": "trade_decision"}
        assert row.raw_responses == {
            "market_diagnosis": '{"stage":"market_diagnosis"}',
            "trade_decision": '{"stage":"trade_decision"}',
        }
        session.commit.assert_called_once()

    def test_previous_successful_analysis_reads_semantic_analysis_fields(self):
        row = SimpleNamespace(
            candle_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            trade_decision={"stage": "trade_decision"},
            market_diagnosis={"stage": "market_diagnosis"},
            validation_status="valid",
        )
        query = MagicMock()
        query.filter.return_value.order_by.return_value.first.return_value = row
        session = MagicMock()
        session.query.return_value = query
        ctx = MagicMock()
        ctx.__enter__.return_value = session
        ctx.__exit__.return_value = False
        factory = MagicMock(return_value=ctx)
        repository = PriceActionRepository(factory, timeframe="1h", market="crypto")

        previous = repository.get_previous_successful_analysis(
            symbol="BTC/USDT",
            timeframe="1h",
            before_time=datetime(2026, 6, 2, tzinfo=timezone.utc),
        )

        assert previous == {
            "candle_time": "2026-06-01T00:00:00+00:00",
            "decision": {"stage": "trade_decision"},
            "diagnosis": {"stage": "market_diagnosis"},
            "validation_status": "valid",
        }


class TestPaAnalysisWorker:
    def test_deduplicates_same_closed_candle(self):
        orchestrator = MagicMock()
        worker = PaAnalysisWorker(orchestrator)
        df = _df_from_ohlc([(1, 2, 0.5, 1.5), (2, 3, 1.5, 2.5)])

        first = worker.submit(
            symbol="BTC/USDT",
            dataframe=df,
            timeframe="1h",
            market="crypto",
        )
        second = worker.submit(
            symbol="BTC/USDT",
            dataframe=df,
            timeframe="1h",
            market="crypto",
        )

        assert first is True
        assert second is False


class TestPriceActionMonitorPipeline:
    """Verify the strategy wires the LLM pipeline and degrades gracefully."""

    def _make_strategy(self, config=None):
        # Import lazily so the freqtrade stub is in place.
        from price_action_monitor import PriceActionMonitor

        s = PriceActionMonitor(config=config or {})
        s.timeframe = "1h"
        return s

    def test_pipeline_disabled_by_config(self):
        """pa_agent_enabled=false → no worker/orchestrator created."""
        s = self._make_strategy({"pa_agent_enabled": False})
        s._init_pa_pipeline()

        assert s._worker is None
        assert s._orchestrator is None
        assert s._llm_client is None

    def test_pipeline_disabled_without_api_key(self):
        """No api_key (no config, no env var) → graceful disable, no exception."""
        s = self._make_strategy({})
        # Ensure no key in env
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEPSEEK_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            s._init_pa_pipeline()

        assert s._worker is None
        assert s._llm_client is None

    def test_pipeline_starts_with_api_key(self):
        """With pa_llm_api_key set, worker + orchestrator are created and started."""
        s = self._make_strategy({"pa_llm_api_key": "sk-test"})

        started = []
        with patch.object(PaAnalysisWorker, "start", lambda self: started.append(self)):
            s._init_pa_pipeline()

        assert s._llm_client is not None
        assert s._llm_client.api_key == "sk-test"
        assert s._orchestrator is not None
        assert s._worker is not None
        assert len(started) == 1
        # stop the worker thread pool cleanly
        s._worker._started = False

    def test_should_notify_only_for_enter_decisions(self):
        from price_action.orchestrator import PriceActionOrchestrator

        orc = PriceActionOrchestrator(
            repository=None,
            llm_client=MagicMock(),
            config={},
        )
        assert orc._should_notify({"decision": {"type": "enter_long"}}) is True
        assert orc._should_notify({"decision": {"type": "enter_short"}}) is True
        assert orc._should_notify({"decision": {"type": "wait"}}) is False
        assert orc._should_notify({"decision": {"type": "avoid"}}) is False
        assert orc._should_notify(None) is False
        # pa_notify_wait=True surfaces wait/avoid too
        orc_wait = PriceActionOrchestrator(
            repository=None, llm_client=MagicMock(), config={"pa_notify_wait": True}
        )
        assert orc_wait._should_notify({"decision": {"type": "wait"}}) is True
