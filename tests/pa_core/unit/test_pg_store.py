"""Tests for pa_core PG persistence (run on SQLite; schema is dialect-neutral)."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from pa_core.records.pg_store import AnalysisRecordRow, PaBase, PgRecordStore, SignalRow
from pa_core.records.schema import AnalysisRecord, FollowupTurn, RecordMeta


def _make_record(
    *,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    decision: dict | None = None,
    stage2: dict | None = None,
    k1_ts_open_ms: float = 1_782_000_000_000.0,
) -> AnalysisRecord:
    if stage2 is None:
        stage2 = {
            "decision": decision
            or {
                "order_type": "不下单",
                "order_direction": None,
                "entry_price": None,
                "stop_loss_price": None,
                "take_profit_price": None,
            },
            "terminal": {"outcome": "wait", "node_id": "9.1", "label": "等待"},
            "trade_confidence": 30,
        }
    return AnalysisRecord(
        meta=RecordMeta(
            timestamp_local_iso="2026-06-30T15:30:00",
            timestamp_local_ms=1_782_000_600_000,
            symbol=symbol,
            timeframe=timeframe,
            bar_count=100,
            ai_provider={"model": "deepseek-v4-flash"},
        ),
        kline_data=[
            {"seq": 1, "ts_open": k1_ts_open_ms, "open": 100, "high": 101, "low": 99, "close": 100.5},
            {"seq": 2, "ts_open": k1_ts_open_ms - 3_600_000, "open": 99, "high": 100, "low": 98, "close": 100},
        ],
        htf_text="",
        stage1_messages=[{"role": "user", "content": "s1"}],
        stage1_response={"content": "{}"},
        stage1_diagnosis={
            "cycle_position": "normal_channel",
            "direction": "bullish",
            "diagnosis_confidence": 70,
            "detected_patterns": ["HH+HL"],
            "support_levels": [99.0],
            "resistance_levels": [110.0],
        },
        stage2_messages=[{"role": "user", "content": "s2"}],
        stage2_response={"content": "{}"},
        stage2_decision=stage2,
        strategy_files_used=["二元决策.txt"],
        experience_loaded=[],
        exception=None,
        usage_total={"prompt_tokens": 1000, "completion_tokens": 200},
    )


@pytest.fixture()
def store(tmp_path):
    return PgRecordStore(f"sqlite:///{tmp_path}/pa_test.db")


def test_save_full_without_order_writes_record_but_no_signal(store: PgRecordStore):
    record = _make_record()  # default 不下单
    row_id = store.save_full(record)
    assert row_id is not None
    with Session(store._engine) as s:
        assert s.query(AnalysisRecordRow).count() == 1
        assert s.query(SignalRow).count() == 0


def test_save_full_with_order_writes_signal_ledger(store: PgRecordStore):
    decision = {
        "order_type": "限价单",
        "order_direction": "做多",
        "entry_price": 100.0,
        "stop_loss_price": 98.0,
        "take_profit_price": 104.0,
        "take_profit_price_2": 108.0,
        "estimated_win_rate": 55,
    }
    row_id = store.save_full(_make_record(decision=decision))
    assert row_id is not None
    signals = store.query_signals(symbol="BTC/USDT")
    assert len(signals) == 1
    sig = signals[0]
    assert sig["order_type"] == "限价单"
    assert sig["order_direction"] == "做多"
    assert sig["entry_price"] == 100.0
    assert sig["risk_reward"] == pytest.approx(2.0)  # (104-100)/(100-98)
    assert sig["estimated_win_rate"] == 55
    assert sig["cycle_position"] == "normal_channel"
    assert sig["direction"] == "bullish"
    assert sig["record_id"] == row_id
    assert sig["executed_trade_id"] is None
    assert sig["candle_time"].startswith("2026-06")  # K1 ts_open → datetime


def test_find_latest_successful_roundtrip(store: PgRecordStore):
    record = _make_record()
    store.save_full(record)
    restored = store.find_latest_successful(symbol="BTC/USDT", timeframe="1h")
    assert restored is not None
    assert restored.meta.symbol == "BTC/USDT"
    assert restored.stage1_diagnosis["cycle_position"] == "normal_channel"
    assert restored.kline_data[0]["ts_open"] == 1_782_000_000_000.0
    # full fidelity: model roundtrip equals original dump
    assert restored.model_dump() == record.model_dump()


def test_find_latest_skips_partial_and_wrong_pair(store: PgRecordStore):
    store.save_full(_make_record())
    store.save_partial(_make_record(k1_ts_open_ms=1_782_003_600_000.0), "network_error")
    store.save_full(_make_record(symbol="ETH/USDT"))
    got = store.find_latest_successful(symbol="BTC/USDT", timeframe="1h")
    assert got is not None
    assert got.meta.symbol == "BTC/USDT"
    # partial never returned
    with Session(store._engine) as s:
        assert s.query(AnalysisRecordRow).filter_by(is_partial=True).count() == 1


def test_api_key_masked_in_stored_json(store: PgRecordStore):
    record = _make_record()
    record.meta.ai_provider = {"model": "deepseek-v4-flash", "api_key": "sk-SECRET123"}
    store.save_full(record, api_key="sk-SECRET123")
    with Session(store._engine) as s:
        row = s.query(AnalysisRecordRow).one()
        raw = str(row.record_json)
        assert "sk-SECRET123" not in raw
        assert "***" in raw  # mask_secret output


def test_append_followup(store: PgRecordStore):
    row_id = store.save_full(_make_record())
    turn = FollowupTurn(turn=1, ts_ms=1, user="why", ai_content="because", ai_reasoning=None, usage={})
    store.append_followup(row_id, turn)
    with Session(store._engine) as s:
        row = s.get(AnalysisRecordRow, row_id)
        assert len(row.followups) == 1
        assert row.followups[0]["user"] == "why"


def test_two_tables_on_separate_metadata():
    # pa_core tables must not leak into freqtrade's metadata.
    # Skip when freqtrade's own deps are not installed in this env.
    pytest.importorskip("humanize", reason="freqtrade deps not installed")
    from freqtrade.persistence.base import ModelBase

    pa_tables = set(PaBase.metadata.tables)
    ft_tables = set(ModelBase.metadata.tables)
    assert "signal" in pa_tables
    assert "analysis_record" in pa_tables
    assert not (pa_tables & ft_tables), "pa_core metadata must stay separate"
