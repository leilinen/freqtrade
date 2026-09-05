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

REPO_ROOT = Path(__file__).resolve().parents[3]
STRATEGY_DIR = REPO_ROOT / "user_data" / "strategies"
sys.path.insert(0, str(STRATEGY_DIR))

from freqtrade.enums import RunMode  # noqa: E402

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


def _make_df(rows: int = 200) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    dates = pd.date_range(start=start, periods=rows, freq="1h")
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

    def current_whitelist(self):
        return ["BTC/USDT"]

    def send_msg(self, message, *, always_send=False):
        pass


def test_strategy_instantiates_and_bot_start_builds_pipeline(tmp_path):
    strategy = PriceActionWatch(dict(CONFIG, pa_db_url=f"sqlite:///{tmp_path}/s.db"))
    strategy.dp = _FakeDP()
    strategy.bot_start()
    assert strategy._orchestrator is not None
    assert strategy._store is not None
    assert strategy.startup_candle_count == 100 + 50 + 10


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
    strategy = PriceActionWatch(dict(CONFIG, pa_db_url=f"sqlite:///{tmp_path}/s.db"))
    strategy.watch_only.value = False
    strategy.dp = _FakeDP()
    strategy.bot_start()

    decision = {
        "order_type": "限价单",
        "order_direction": "做多",
        "entry_price": 100.0,
        "stop_loss_price": 98.0,
        "take_profit_price": 104.0,
    }
    df = _make_df(200).copy()
    strategy._apply_decision(df, "BTC/USDT", mocker.sentinel.record, decision)
    assert df.iloc[-1]["enter_long"] == 1
    # watch-only off + short decision
    df2 = _make_df(200).copy()
    strategy._apply_decision(
        df2, "BTC/USDT", mocker.sentinel.record, {**decision, "order_direction": "做空"}
    )
    assert df2.iloc[-1].get("enter_short", 0) == 1


def test_watch_config_json_is_valid_and_watch_safe():
    cfg_path = REPO_ROOT / "user_data" / "watch_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["dry_run"] is True
    assert cfg["max_open_trades"] == 0
    assert cfg["strategy"] == "PriceActionWatch"
    assert "pa_llm" in cfg and "api_key" in cfg["pa_llm"]
    assert "pa_db_url" in cfg
