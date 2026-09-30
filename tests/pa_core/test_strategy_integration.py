"""Integration tests for the PriceActionWatch strategy (needs freqtrade deps).

Skipped automatically in environments without freqtrade's full dependency set
(.venv311 only carries pa_core deps). Runs in the project's dev environment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("freqtrade.strategy", reason="freqtrade deps not installed in this env")

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = REPO_ROOT / "user_data" / "strategies"
sys.path.insert(0, str(STRATEGY_DIR))

from freqtrade.enums import RunMode  # noqa: E402

from pa_core.records.watch_pair_store import WatchPairStore  # noqa: E402

from price_action_watch import (  # noqa: E402
    PriceActionWatch,
    _build_settings,
    _direction_sign,
    _is_order_plan,
)

CONFIG = {
    "timeframe": "1h",
    "strategy": "PriceActionWatch",
    "pa_db_url": "sqlite:///:memory:",
    "pa_llm": {"model": "deepseek-v4-flash", "api_key": ""},
    "stake_currency": "USDT",
    "dry_run": True,
}


def _make_df(rows: int = 200, freq: str = "1h") -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    dates = pd.date_range(start=start, periods=rows, freq=freq)
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.standard_normal(rows))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + rng.standard_normal(rows) * 0.1,
            "high": close + np.abs(rng.standard_normal(rows)) * 0.5,
            "low": close - np.abs(rng.standard_normal(rows)) * 0.5,
            "close": close,
            "volume": rng.random(rows) * 1000,
        }
    )


def test_settings_mapping_from_freqtrade_config():
    settings = _build_settings(
        {"pa_llm": {"model": "m1", "base_url": "http://x", "api_key": "k",
                    "analysis_bar_count": 60, "decision_stance": "aggressive"}}
    )
    assert settings.provider.model == "m1"
    assert settings.provider.api_key == "k"
    assert settings.general.analysis_bar_count == 60
    assert settings.general.decision_stance == "aggressive"


def test_defaults_without_pa_llm_block():
    settings = _build_settings({})
    assert settings.provider.model == "deepseek-v4-flash"
    assert settings.general.analysis_bar_count == 100


def test_decision_helpers():
    assert _is_order_plan({"order_type": "限价单"}) is True
    assert _is_order_plan({"order_type": "不下单"}) is False
    assert _is_order_plan(None) is False
    assert _direction_sign({"order_direction": "做多"}) == 1
    assert _direction_sign({"order_direction": "做空"}) == -1


class _FakeDP:
    runmode = RunMode.DRY_RUN

    def __init__(self, whitelist=None, ohlcv_data=None):
        self._whitelist = whitelist or ["BTC/USDT"]
        self._ohlcv = ohlcv_data or {}

    def current_whitelist(self):
        return list(self._whitelist)

    def ohlcv(self, pair, timeframe=None, candle_type=None, copy=True):
        return self._ohlcv.get((pair, timeframe))

    def send_msg(self, message, *, always_send=False):
        pass


def test_strategy_instantiates_and_bot_start_builds_pipeline(tmp_path):
    strategy = PriceActionWatch(dict(CONFIG, pa_db_url=f"sqlite:///{tmp_path}/s.db"))
    strategy.dp = _FakeDP()
    strategy.bot_start()
    assert strategy._orchestrator is not None
    assert strategy._store is not None
    assert strategy.startup_candle_count == 100 + 50 + 10


def test_bot_start_seeds_watch_pair_defaults_on_cold_start(tmp_path):
    db = f"sqlite:///{tmp_path}/cold.db"
    strategy = PriceActionWatch(dict(CONFIG, pa_db_url=db))
    strategy.dp = _FakeDP()
    strategy.bot_start()

    rows = WatchPairStore(db).list_pairs()
    assert [r["symbol"] for r in rows] == ["BTC/USDT", "ETH/USDT"]
    assert all(r["enabled"] for r in rows)


def test_bot_start_keeps_existing_watch_pair_rows(tmp_path):
    db = f"sqlite:///{tmp_path}/warm.db"
    WatchPairStore(db).add("SOL/USDT")  # non-empty market → no cold-start seeding

    strategy = PriceActionWatch(dict(CONFIG, pa_db_url=db))
    strategy.dp = _FakeDP()
    strategy.bot_start()

    rows = WatchPairStore(db).list_pairs()
    assert [r["symbol"] for r in rows] == ["SOL/USDT"]


def test_populate_entry_trend_live_handles_llm_failure(tmp_path, mocker):
    """Without openai/credentials the orchestrator returns an exception record;
    populate_* must not raise and must leave entry columns untouched."""
    strategy = PriceActionWatch(dict(CONFIG, pa_db_url=f"sqlite:///{tmp_path}/s.db"))
    strategy.dp = _FakeDP()
    strategy.bot_start()

    # Force a completed-with-exception record (no real LLM call).
    def fake_submit(frame, cancel_token, on_event, **kwargs):
        from pa_core.records.schema import AnalysisRecord, RecordMeta

        return AnalysisRecord(
            meta=RecordMeta(
                timestamp_local_iso="2026-06-30T15:30:00",
                timestamp_local_ms=1,
                symbol=frame.symbol,
                timeframe=frame.timeframe,
                bar_count=len(frame.bars),
                ai_provider={},
            ),
            kline_data=[{"ts_open": b.ts_open} for b in frame.bars],
            htf_text="",
            stage1_messages=[],
            stage1_response=None,
            stage1_diagnosis=None,
            stage2_messages=[],
            stage2_response=None,
            stage2_decision=None,
            strategy_files_used=[],
            experience_loaded=[],
            exception={"category": "network_error"},
            usage_total={},
        )

    mocker.patch.object(strategy._orchestrator, "submit", side_effect=fake_submit)
    df = _make_df(200).copy()
    out = strategy.populate_entry_trend(df, {"pair": "BTC/USDT"})
    assert "enter_long" not in out.columns or (out["enter_long"] == 0).all()


def test_order_plan_applies_entry_when_not_watch_only(tmp_path):
    from types import SimpleNamespace

    strategy = PriceActionWatch(
        dict(CONFIG, pa_db_url=f"sqlite:///{tmp_path}/s.db", pa_llm={"watch_only": False})
    )
    strategy.dp = _FakeDP()
    strategy.bot_start()
    assert strategy.watch_only is False  # bot_start reads it from config

    decision = {
        "order_type": "限价单",
        "order_direction": "做多",
        "entry_price": 100.0,
        "stop_loss_price": 98.0,
        "take_profit_price": 104.0,
    }

    def _record(dec):
        return SimpleNamespace(
            stage2_decision={"decision": dec, "trade_confidence": 60},
            stage1_diagnosis={"cycle_position": "normal_channel", "direction": "bullish"},
        )

    df = _make_df(200).copy()
    strategy._apply_decision(df, "BTC/USDT", _record(decision), decision, "1h")
    assert df.iloc[-1]["enter_long"] == 1
    # watch-only off + short decision
    df2 = _make_df(200).copy()
    short = {**decision, "order_direction": "做空"}
    strategy._apply_decision(df2, "BTC/USDT", _record(short), short, "1h")
    assert df2.iloc[-1].get("enter_short", 0) == 1


def test_watch_config_json_is_valid_and_watch_safe():
    cfg_path = REPO_ROOT / "user_data" / "watch_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["dry_run"] is True
    assert cfg["max_open_trades"] == 0
    assert cfg["strategy"] == "PriceActionWatch"
    assert "pa_llm" in cfg and "api_key" in cfg["pa_llm"]
    assert "pa_db_url" in cfg
    # whitelist comes from the PG watch_pair table; the config carries no
    # pair_whitelist anymore (cold start seeds defaults into watch_pair)
    assert cfg["pairlists"][0]["method"] == "DatabasePairList"
    assert "refresh_period" in cfg["pairlists"][0]
    assert "pair_whitelist" not in cfg["exchange"]


def _fake_ok_submit(frames: list):
    """submit() double capturing KlineFrames, returning successful records."""

    def fake_submit(frame, cancel_token, on_event, **kwargs):
        from pa_core.records.schema import AnalysisRecord, RecordMeta

        frames.append(frame)
        return AnalysisRecord(
            meta=RecordMeta(
                timestamp_local_iso="2026-06-30T15:30:00",
                timestamp_local_ms=1,
                symbol=frame.symbol,
                timeframe=frame.timeframe,
                bar_count=len(frame.bars),
                ai_provider={},
            ),
            kline_data=[{"ts_open": b.ts_open} for b in frame.bars],
            htf_text="",
            stage1_messages=[],
            stage1_response=None,
            stage1_diagnosis={"cycle_position": "normal_channel", "direction": "bullish"},
            stage2_messages=[],
            stage2_response=None,
            stage2_decision={"decision": {"order_type": "不下单"}, "trade_confidence": 30},
            strategy_files_used=[],
            experience_loaded=[],
            exception=None,
            usage_total={},
        )

    return fake_submit


def test_off_timeframe_pair_analyzes_on_own_candle_close(tmp_path, mocker):
    db = f"sqlite:///{tmp_path}/tf.db"
    WatchPairStore(db).add("SOL/USDT", timeframe="1d")

    df_1h = _make_df(200, "1h")
    df_1d = _make_df(120, "1d")
    dp = _FakeDP(whitelist=["SOL/USDT"], ohlcv_data={("SOL/USDT", "1d"): df_1d})
    strategy = PriceActionWatch(dict(CONFIG, pa_db_url=db))
    strategy.dp = dp
    strategy.bot_start()

    # off-tf pairs are declared as informative; default pairs are not
    assert strategy.informative_pairs() == [("SOL/USDT", "1d")]

    frames: list = []
    mocker.patch.object(strategy._orchestrator, "submit", side_effect=_fake_ok_submit(frames))

    strategy.populate_entry_trend(df_1h.copy(), {"pair": "SOL/USDT"})
    assert len(frames) == 1
    assert frames[0].timeframe == "1d"
    assert frames[0].symbol == "SOL/USDT"

    # engine fired again on a new 1h candle, but the 1d bar hasn't closed
    strategy.populate_entry_trend(df_1h.copy(), {"pair": "SOL/USDT"})
    assert len(frames) == 1

    # 1d candle closes -> analyze again at 1d
    dp._ohlcv[("SOL/USDT", "1d")] = _make_df(121, "1d")
    strategy.populate_entry_trend(df_1h.copy(), {"pair": "SOL/USDT"})
    assert len(frames) == 2
    assert frames[1].timeframe == "1d"


def test_default_timeframe_pair_still_uses_engine_dataframe(tmp_path, mocker):
    """Pairs without a timeframe override keep the main-timeframe path."""
    db = f"sqlite:///{tmp_path}/default.db"
    WatchPairStore(db).add("BTC/USDT")  # no timeframe -> strategy default

    dp = _FakeDP(whitelist=["BTC/USDT"], ohlcv_data={("BTC/USDT", "1d"): _make_df(10, "1d")})
    strategy = PriceActionWatch(dict(CONFIG, pa_db_url=db))
    strategy.dp = dp
    strategy.bot_start()
    assert strategy.informative_pairs() == []

    frames: list = []
    mocker.patch.object(strategy._orchestrator, "submit", side_effect=_fake_ok_submit(frames))
    strategy.populate_entry_trend(_make_df(200, "1h").copy(), {"pair": "BTC/USDT"})
    assert len(frames) == 1
    assert frames[0].timeframe == "1h"


def test_no_off_tf_candles_skips_analysis(tmp_path, mocker):
    db = f"sqlite:///{tmp_path}/nocandles.db"
    WatchPairStore(db).add("SOL/USDT", timeframe="1d")

    dp = _FakeDP(whitelist=["SOL/USDT"])  # ohlcv returns None (not refreshed yet)
    strategy = PriceActionWatch(dict(CONFIG, pa_db_url=db))
    strategy.dp = dp
    strategy.bot_start()

    frames: list = []
    mocker.patch.object(strategy._orchestrator, "submit", side_effect=_fake_ok_submit(frames))
    strategy.populate_entry_trend(_make_df(200, "1h").copy(), {"pair": "SOL/USDT"})
    assert frames == []
