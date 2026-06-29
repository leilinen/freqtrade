"""Tests for the PA_Agent-style price-action pipeline modules."""
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
    build_price_action_features,
    calculate_atr,
    calculate_ema,
)
from price_action.experience import retrieve_experience_cases  # noqa: E402
from price_action.llm import OpenAIJsonClient  # noqa: E402
from price_action.orchestrator import PriceActionOrchestrator  # noqa: E402
from price_action.prompts import (  # noqa: E402
    build_market_diagnosis_messages,
    prompt_template_metadata,
)
from price_action.repository import PriceActionRepository  # noqa: E402
from price_action.router import route_strategies  # noqa: E402
from price_action.validation import DecisionValidator, validate_market_diagnosis  # noqa: E402
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


def _market_diagnosis(
    features,
    *,
    gate_result="proceed",
    cycle_position="normal_channel",
    direction="bullish",
):
    rows = features.rows[:5]
    if gate_result == "proceed":
        gate_trace = [
            {
                "node_id": "1.2",
                "question": "是否能识别出当前市场周期？",
                "answer": "是",
                "reason": "通道结构可识别",
                "branch": cycle_position,
                "section": "K线识别",
                "bar_range": "K5-K1",
            },
            {
                "node_id": "1.3",
                "question": "市场是否不是极端混乱？",
                "answer": "是",
                "reason": "",
                "branch": None,
                "section": "K线识别",
                "bar_range": "K5-K1",
            },
            {
                "node_id": "2.1",
                "question": "近期结构是否呈现明确惯性方向？",
                "answer": "是",
                "reason": "",
                "branch": direction,
                "section": "方向判断",
                "bar_range": "K5-K1",
            },
            {
                "node_id": "2.2",
                "question": "长程背景是否支持近期方向？",
                "answer": "中性",
                "reason": "",
                "branch": "mixed",
                "section": "背景判断",
                "bar_range": "K5-K1",
            },
            {
                "node_id": "2.5",
                "question": "当前惯性强度是否足以进入交易决策？",
                "answer": "是",
                "reason": "闸门通过，进入交易决策",
                "branch": direction,
                "section": "闸门",
                "bar_range": "K3-K1",
            },
        ]
    else:
        gate_trace = [
            {
                "node_id": "1.2",
                "question": "是否能识别出当前市场周期？",
                "answer": "否",
                "reason": "周期无法识别，等待更多 K 线",
                "branch": "unknown",
                "section": "K线识别",
                "bar_range": "K5-K1",
            }
        ]
    return {
        "cycle_position": cycle_position,
        "alternative_cycle_position": None,
        "direction": direction,
        "diagnosis_confidence": 75,
        "spike_stage": None,
        "climax_risk": "none",
        "market_phase": "stable",
        "transition_risk": None,
        "detected_patterns": ["breakout_up"] if direction == "bullish" else [],
        "key_signals": ["K1 close"],
        "htf_context": "背景中性",
        "entry_setup": "breakout_pullback",
        "support_levels": ["100"],
        "resistance_levels": ["120"],
        "strategy_files_needed": [],
        "risk_warning": "等待更多确认",
        "bar_analysis": {
            "always_in": {
                "bullish": "long",
                "bearish": "short",
                "neutral": "neutral",
            }[direction],
            "last_closed_bar": "K1",
            "bar_type": features.latest_features["bar_type"],
            "signal_bar": {
                "bar": "K1",
                "quality": "medium",
                "pattern": "breakout_pullback",
                "reason": "测试信号",
            },
            "entry_setup_type": "breakout_pullback",
            "follow_through": features.latest_features["follow_through_1_2"],
        },
        "bar_by_bar_summary": [
            {
                "bar": row["k"],
                "role": "structure",
                "bar_type": row["bar_type"],
                "context_effect": "neutral",
                "follow_through": row["follow_through_1_2"],
                "trapped_side": "none",
                "reason": f"{row['k']} 结构摘要",
            }
            for row in rows
        ],
        "gate_trace": gate_trace,
        "gate_result": gate_result,
    }


class TestPriceActionFeatures:
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

        result = build_price_action_features(
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

        result = build_price_action_features(
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

    def test_bar_type_classifies_outside_and_trend_bars(self):
        df = _df_from_ohlc(
            [
                (11.0, 13.0, 10.0, 12.0),
                (10.0, 15.0, 9.0, 14.5),
            ]
        )

        result = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=2,
            warmup=0,
            now=pd.Timestamp("2026-06-01 11:30", tz="UTC"),
        )

        assert result.rows[0]["bar_type"] == "outside_bull"
        assert result.rows[0]["overlap_prev_ratio"] == pytest.approx(0.5)
        assert result.rows[1]["bar_type"] == "trend_bull"

    def test_ioi_gap_ema_gap_count_and_breakout_prev_match_pa_agent_context(self):
        base = [(9.0, 9.5, 8.5, 9.0)] * 20
        ioi_tail = [
            (10.0, 15.0, 9.0, 14.0),
            (10.5, 12.0, 10.4, 11.5),
            (11.0, 14.0, 10.2, 13.5),
            (12.0, 13.0, 11.0, 12.8),
        ]
        df = _df_from_ohlc(base + ioi_tail)

        result = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=4,
            warmup=20,
            now=pd.Timestamp("2026-06-02 09:30", tz="UTC"),
        )

        latest = result.latest_features
        assert latest["ioi_pattern"] is True
        assert latest["gap_bar"] == "bull_gap"
        assert latest["ema_gap_count"] == 3
        assert latest["breakout_prev"] == "none"
        assert "ioi" in latest["patterns"]
        assert "bull_gap" in latest["patterns"]

    def test_inside_sequence_and_micro_double_are_reported(self):
        df = _df_from_ohlc(
            [
                (12.0, 14.0, 9.0, 13.0),
                (11.0, 13.0, 10.0, 12.0),
                (10.0, 12.0, 10.0, 11.0),
            ]
        )

        result = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=3,
            warmup=0,
            now=pd.Timestamp("2026-06-01 12:30", tz="UTC"),
        )

        assert result.latest_features["inside_sequence"] == "ii"
        assert result.latest_features["micro_double"] == "MDB"
        assert "MDB" in result.latest_features["patterns"]

    def test_flat_bar_type_when_zero_range_and_not_inside(self):
        df = _df_from_ohlc(
            [
                (9.0, 9.0, 8.0, 8.5),
                (10.0, 10.0, 10.0, 10.0),
            ]
        )

        result = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=2,
            warmup=0,
            now=pd.Timestamp("2026-06-01 11:30", tz="UTC"),
        )

        assert result.latest_features["bar_type"] == "flat"

    def test_follow_through_uses_newer_closes_for_bull_and_bear_failures(self):
        bull_df = _df_from_ohlc(
            [
                (10.0, 12.0, 9.0, 11.0),
                (8.0, 9.0, 7.5, 7.8),
            ]
        )
        bull_result = build_price_action_features(
            bull_df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=2,
            warmup=0,
            now=pd.Timestamp("2026-06-01 11:30", tz="UTC"),
        )
        assert bull_result.rows[1]["follow_through_1_2"] == "failed"

        bear_df = _df_from_ohlc(
            [
                (10.0, 11.0, 8.0, 9.0),
                (11.5, 12.0, 10.5, 11.2),
            ]
        )
        bear_result = build_price_action_features(
            bear_df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=2,
            warmup=0,
            now=pd.Timestamp("2026-06-01 11:30", tz="UTC"),
        )
        assert bear_result.rows[1]["follow_through_1_2"] == "failed"

    def test_ashare_daily_uses_local_market_close(self):
        df = _df_from_ohlc(
            [
                (1.00, 1.05, 0.98, 1.03),
                (1.03, 1.08, 1.02, 1.06),
            ],
            start="2026-06-01",
            freq="D",
        )

        result = build_price_action_features(
            df,
            symbol="588290/SH",
            timeframe="1d",
            market="ashare",
            window=2,
            warmup=0,
            now=pd.Timestamp("2026-06-02 15:30", tz="Asia/Shanghai"),
        )

        assert result.latest_features["time"] == "2026-06-02T00:00:00"


class TestMarketStructureFeatures:
    def test_range_position_upper_third(self):
        df = _df_from_ohlc([(105.0, 110.0, 100.0, 108.0)] * 8)

        result = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=8,
            warmup=0,
            now=pd.Timestamp("2026-06-01 16:30", tz="UTC"),
        )

        mf = result.market_features
        assert mf["range_high"] == 110.0
        assert mf["range_low"] == 100.0
        assert mf["zone"] == "upper_third"
        assert mf["price_position"] > 2 / 3

    def test_swing_pivots_and_structure_label_are_exposed(self):
        df = _df_from_ohlc(
            [
                (100.0, 105.0, 99.0, 104.0),
                (104.0, 105.0, 100.0, 102.0),
                (102.0, 108.0, 101.5, 107.0),
                (107.0, 107.5, 104.0, 105.0),
                (105.0, 110.0, 104.5, 109.0),
                (109.0, 111.0, 108.0, 110.0),
            ]
        )

        result = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=6,
            warmup=0,
            now=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
        )

        swings = result.market_features["swings"]
        assert {s["kind"] for s in swings} >= {"high", "low"}
        swing_structure = result.market_features["swing_structure"]
        assert swing_structure in {"HH+HL", "LL+LH", "mixed", "insufficient"}

    def test_hl_count_triggers_on_high_breaks(self):
        df = _df_from_ohlc(
            [
                (9.8, 10.0, 9.6, 9.9),
                (9.9, 10.1, 9.7, 10.0),
                (10.0, 10.2, 9.8, 10.1),
                (10.2, 10.4, 10.0, 10.3),
            ]
        )

        result = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=4,
            warmup=0,
            now=pd.Timestamp("2026-06-01 12:30", tz="UTC"),
        )

        hl_count = result.market_features["hl_count"]
        assert hl_count["bull_count"] >= 2
        assert hl_count["bull_candidate"] in ("h2", "h3")
        assert hl_count["last_bull_trigger_seq"] == 1

    def test_breakout_failure_detected(self):
        df = _df_from_ohlc(
            [
                (97.0, 98.0, 96.5, 97.5),
                (98.0, 99.0, 97.5, 98.5),
                (99.0, 100.0, 98.5, 99.2),
                (100.0, 101.5, 99.8, 101.2),
                (99.2, 99.8, 98.8, 99.3),
                (99.0, 99.5, 98.5, 99.0),
            ]
        )

        result = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=6,
            warmup=0,
            now=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
        )

        failed = [
            event for event in result.market_features["breakout_events"]
            if event["event"] == "failed"
        ]
        assert failed
        assert failed[0]["level_kind"] == "range_high"

    def test_measured_move_range_projection_and_prompt_render(self):
        df = _df_from_ohlc([(105.0, 110.0, 100.0, 105.0)] * 6)

        result = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=6,
            warmup=0,
            now=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
        )

        range_up = [
            move for move in result.market_features["measured_moves"]
            if move["kind"] == "range_up"
        ]
        assert range_up
        assert range_up[0]["height"] == 10.0
        assert range_up[0]["target_price"] == 120.0
        assert "程序结构辅助特征" in result.market_features_text
        assert "Measured Move" in result.market_features_text

        messages = build_market_diagnosis_messages(result)
        assert "市场结构辅助特征" in messages[1]["content"]
        assert "Measured Move" in messages[1]["content"]


class TestMarketDiagnosisPrompt:
    def test_prompt_template_metadata_exposes_file_hashes(self):
        metadata = prompt_template_metadata()

        assert metadata["market_diagnosis"][0]["name"] == "market_diagnosis_system.txt"
        assert metadata["trade_decision"][1]["name"] == "trade_decision_user.txt"
        assert len(metadata["market_diagnosis"][0]["sha256"]) == 64

    def test_prompt_uses_pa_agent_market_diagnosis_contract(self):
        df = _df_from_ohlc([(100.0, 104.0, 99.0, 103.0)] * 6)
        features = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=6,
            warmup=0,
            now=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
        )

        messages = build_market_diagnosis_messages(features)
        content = messages[1]["content"]

        assert "任务：完成市场诊断" in content
        assert '"cycle_position"' in content
        assert '"gate_trace"' in content
        assert '"gate_result"' in content
        assert "1.2、1.3、2.1、2.2、2.5" in content
        assert '"market_state"' not in content
        assert '"signal_chain"' not in content

    def test_market_diagnosis_validation_accepts_pa_agent_contract(self):
        df = _df_from_ohlc([(100.0, 104.0, 99.0, 103.0)] * 6)
        features = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=6,
            warmup=0,
            now=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
        )

        errors = validate_market_diagnosis(_market_diagnosis(features), feature_rows=features.rows)

        assert errors == []

    def test_market_diagnosis_validation_rejects_bar_type_mismatch(self):
        df = _df_from_ohlc([(100.0, 104.0, 99.0, 103.0)] * 6)
        features = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=6,
            warmup=0,
            now=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
        )
        diagnosis = _market_diagnosis(features)
        diagnosis["bar_analysis"]["bar_type"] = "trend_bear"

        errors = validate_market_diagnosis(diagnosis, feature_rows=features.rows)

        assert "market_diagnosis_bar_analysis_bar_type_mismatch" in errors

    def test_market_diagnosis_validation_requires_proceed_gate_nodes(self):
        df = _df_from_ohlc([(100.0, 104.0, 99.0, 103.0)] * 6)
        features = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=6,
            warmup=0,
            now=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
        )
        diagnosis = _market_diagnosis(features)
        diagnosis["gate_trace"] = diagnosis["gate_trace"][:2]

        errors = validate_market_diagnosis(diagnosis, feature_rows=features.rows)

        assert "market_diagnosis_gate_trace_missing_proceed_nodes" in errors


class TestMarketDiagnosisRouting:
    def test_routes_from_cycle_position_direction_and_detected_patterns(self):
        diagnosis = {
            "cycle_position": "normal_channel",
            "direction": "bullish",
            "detected_patterns": [],
            "entry_setup": "H2",
            "bar_analysis": {"entry_setup_type": "H2"},
        }

        routed = route_strategies(diagnosis)

        assert [template.template_id for template in routed] == [
            "trend_pullback_long",
            "ema20_magnet_wait",
        ]

    def test_routes_breakout_pattern_without_old_gate_shape(self):
        diagnosis = {
            "cycle_position": "normal_channel",
            "direction": "bullish",
            "detected_patterns": ["breakout_up"],
            "entry_setup": "breakout_pullback",
            "bar_analysis": {"entry_setup_type": "breakout_pullback"},
        }

        routed = route_strategies(diagnosis)

        assert routed[0].template_id == "breakout_continuation_long"

    def test_experience_lookup_uses_market_diagnosis_fields(self):
        repository = MagicMock()
        repository.query_experience.return_value = [{"id": 1}]
        diagnosis = {
            "cycle_position": "trading_range",
            "direction": "bearish",
            "detected_patterns": ["breakout_failure"],
        }

        cases = retrieve_experience_cases(
            repository,
            market="crypto",
            timeframe="1h",
            diagnosis=diagnosis,
            limit=2,
        )

        assert cases == [{"id": 1}]
        repository.query_experience.assert_called_once_with(
            market="crypto",
            timeframe="1h",
            cycle_position="trading_range",
            direction="bearish",
            patterns=["breakout_failure"],
            limit=2,
        )


class TestMarketDiagnosisOrchestrator:
    def test_gate_wait_short_circuits_trade_decision_model_call(self):
        df = _df_from_ohlc([(100.0, 104.0, 99.0, 103.0)] * 6)
        features = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=6,
            warmup=0,
            now=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
        )
        diagnosis = _market_diagnosis(
            features,
            gate_result="wait",
            cycle_position="unknown",
            direction="neutral",
        )
        llm = MagicMock()
        llm.model = "test-model"
        llm.base_url = "https://example.test"
        llm.complete_json.return_value = json.dumps(diagnosis, ensure_ascii=False)
        orchestrator = PriceActionOrchestrator(
            repository=None,
            llm_client=llm,
            config={"pa_llm_window": 6, "pa_llm_warmup": 0},
        )

        outcome = orchestrator.analyze(
            symbol="BTC/USDT",
            dataframe=df,
            timeframe="1h",
            market="crypto",
        )

        assert outcome.status == "success"
        assert outcome.decision["decision"]["type"] == "wait"
        assert "trade_decision_model_call=skipped" in outcome.decision["decision_trace"]
        llm.complete_json.assert_called_once()

    def test_successful_analysis_persists_prompt_messages_and_template_metadata(self):
        df = _df_from_ohlc([(100.0, 104.0, 99.0, 103.0)] * 6)
        features = build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=6,
            warmup=0,
            now=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
        )
        diagnosis = _market_diagnosis(features)
        decision = {
            "stage": "trade_decision",
            "decision": {
                "type": "wait",
                "direction": "neutral",
                "order_type": "none",
                "entry": None,
                "stop_loss": None,
                "take_profit_1": None,
                "take_profit_2": None,
                "risk_reward": None,
                "confidence": 0.6,
                "reason": "等待",
            },
            "decision_trace": ["trade_decision"],
            "watch_points": [],
            "invalidations": [],
        }
        llm = MagicMock()
        llm.model = "test-model"
        llm.base_url = "https://example.test"
        llm.complete_json.side_effect = [
            json.dumps(diagnosis, ensure_ascii=False),
            json.dumps(decision, ensure_ascii=False),
        ]
        repository = MagicMock()
        repository.query_experience.return_value = []
        repository.get_previous_successful_analysis.return_value = None
        repository.save_analysis.return_value = True
        orchestrator = PriceActionOrchestrator(
            repository=repository,
            llm_client=llm,
            config={"pa_llm_window": 6, "pa_llm_warmup": 0},
        )

        outcome = orchestrator.analyze(
            symbol="BTC/USDT",
            dataframe=df,
            timeframe="1h",
            market="crypto",
        )

        assert outcome.status == "success"
        assert llm.complete_json.call_count == 2
        kwargs = repository.save_analysis.call_args.kwargs
        assert kwargs["market_diagnosis_messages"][0]["role"] == "system"
        assert kwargs["trade_decision_messages"][1]["role"] == "user"
        assert "任务：完成市场诊断" in kwargs["market_diagnosis_messages"][1]["content"]
        assert "任务：完成交易决策" in kwargs["trade_decision_messages"][1]["content"]
        assert kwargs["prompt_metadata"]["prompt_templates"]["market_diagnosis"][0]["sha256"]


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
    diagnosis = {
        "cycle_position": "normal_channel",
        "direction": "bullish",
        "gate_result": "proceed",
    }
    features = {
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
            price_action_features=self.features,
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

        _, result = DecisionValidator().validate(
            raw,
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

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

        _, result = DecisionValidator().validate(
            raw,
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

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

        _, result = DecisionValidator().validate(
            raw,
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

        assert result.checks == [
            "json_syntax",
            "stage_consistency",
            "semantic_reasonableness",
            "numeric_range",
        ]
        assert "confidence_must_be_0_to_1" in result.errors

    def test_atr_expansion_veto_is_semantic(self):
        features = {**self.features, "gate_break": "up", "atr_expand_ratio": 2.1}
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

        _, result = DecisionValidator().validate(
            raw,
            diagnosis=self.diagnosis,
            price_action_features=features,
        )

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
        market_diagnosis = {
            "cycle_position": "normal_channel",
            "direction": "bullish",
            "gate_result": "proceed",
        }

        saved = repository.save_analysis(
            market="crypto",
            symbol="BTC/USDT",
            timeframe="1h",
            candle_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            status="success",
            market_diagnosis_messages=[{"role": "user", "content": "market diagnosis"}],
            trade_decision_messages=[{"role": "user", "content": "trade decision"}],
            market_diagnosis=market_diagnosis,
            trade_decision={"stage": "trade_decision"},
            raw_responses={
                "market_diagnosis": json.dumps(market_diagnosis),
                "trade_decision": '{"stage":"trade_decision"}',
            },
        )

        assert saved is True
        row = session.add.call_args.args[0]
        assert row.market_diagnosis_messages == [
            {"role": "user", "content": "market diagnosis"}
        ]
        assert row.trade_decision_messages == [
            {"role": "user", "content": "trade decision"}
        ]
        assert row.market_diagnosis == market_diagnosis
        assert row.trade_decision == {"stage": "trade_decision"}
        assert row.raw_responses == {
            "market_diagnosis": json.dumps(market_diagnosis),
            "trade_decision": '{"stage":"trade_decision"}',
        }
        session.commit.assert_called_once()

    def test_previous_successful_analysis_reads_semantic_analysis_fields(self):
        market_diagnosis = {
            "cycle_position": "normal_channel",
            "direction": "bullish",
            "gate_result": "proceed",
        }
        row = SimpleNamespace(
            candle_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            trade_decision={"stage": "trade_decision"},
            market_diagnosis=market_diagnosis,
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
            "diagnosis": market_diagnosis,
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
