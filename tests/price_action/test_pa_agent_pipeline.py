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
from price_action.llm import (  # noqa: E402
    AnthropicJsonClient,
    OpenAIJsonClient,
    build_llm_client,
)
from price_action.orchestrator import (  # noqa: E402
    PriceActionOrchestrator,
    _categorize_errors,
    _config_int,
    _retry_limit_for_errors,
    _semantic_retry_limit,
    _validation_retry_limit,
)
from price_action.prompts import (  # noqa: E402
    build_market_diagnosis_messages,
    build_trade_decision_messages,
    prompt_template_metadata,
)
from price_action.repository import PriceActionRepository  # noqa: E402
from price_action.router import route_strategies  # noqa: E402
from price_action.validation import (  # noqa: E402
    DecisionValidator,
    parse_json_object,
    validate_market_diagnosis,
)
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
                "node_id": "1.1",
                "question": "K线数据是否足够完成市场诊断？",
                "answer": "是",
                "reason": "已提供足够的已收盘 K 线",
                "branch": None,
                "section": "数据检查",
                "bar_range": "K5-K1",
            },
            {
                "node_id": "1.2",
                "question": "是否能识别出当前市场周期？",
                "answer": "是",
                "reason": "通道结构可识别",
                "branch": cycle_position,
                "section": "K线识别",
                "bar_range": "K4-K1",
            },
            {
                "node_id": "1.3",
                "question": "市场是否不是极端混乱？",
                "answer": "是",
                "reason": "",
                "branch": None,
                "section": "K线识别",
                "bar_range": "K3-K1",
            },
            {
                "node_id": "2.1",
                "question": "近期结构是否呈现明确惯性方向？",
                "answer": "是",
                "reason": "",
                "branch": direction,
                "section": "方向判断",
                "bar_range": "K4-K2",
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
                "node_id": "2.3",
                "question": "Always In 方向是否与顶层方向一致？",
                "answer": "是",
                "reason": "方向判断一致",
                "branch": direction,
                "section": "方向判断",
                "bar_range": "K2-K1",
            },
            {
                "node_id": "2.4",
                "question": "当前是否没有明显反向陷阱？",
                "answer": "是",
                "reason": "未见反向陷阱",
                "branch": None,
                "section": "陷阱检查",
                "bar_range": "K3-K2",
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


def _trade_decision(
    *,
    order_direction="做多",
    order_type="突破单",
    entry=100.0,
    stop=95.0,
    tp1=110.0,
    tp2=115.0,
    trade_confidence=70,
    estimated_win_rate=55,
    diagnosis=None,
):
    diagnosis = diagnosis or {
        "cycle_position": "normal_channel",
        "direction": "bullish",
        "key_signals": ["K1 close"],
    }
    is_trade = order_type != "不下单"
    return {
        "decision": {
            "order_direction": order_direction if is_trade else None,
            "order_type": order_type,
            "entry_price": entry if is_trade else None,
            "entry_basis_bar": "K1" if order_type == "突破单" else None,
            "entry_basis_extreme": "high" if order_type == "突破单" else None,
            "entry_rule": "突破 K1 高点" if order_type == "突破单" else None,
            "take_profit_price": tp1 if is_trade else None,
            "take_profit_price_2": tp2 if is_trade else None,
            "stop_loss_price": stop if is_trade else None,
            "reasoning": "阶段二交易决策",
            "diagnosis_confidence": 75,
            "diagnosis_confidence_reasoning": "沿用市场诊断置信度",
            "trade_confidence": trade_confidence,
            "trade_confidence_reasoning": "信号链完整",
            "estimated_win_rate": estimated_win_rate if is_trade else None,
            "estimated_win_rate_reasoning": "满足交易者方程" if is_trade else None,
            "key_factors": ["signal", "risk"],
            "watch_points": ["follow through"],
            "risk_assessment": "risk controlled",
            "invalidation_condition": None,
        },
        "diagnosis_summary": {
            "cycle_position": diagnosis.get("cycle_position", "normal_channel"),
            "direction": diagnosis.get("direction", "bullish"),
            "key_signals": list(diagnosis.get("key_signals") or []),
        },
        "decision_trace": [
            {
                "node_id": "9.0",
                "question": "是否存在可交易的入场信号？",
                "answer": "是" if is_trade else "等待",
                "reason": "测试入场信号",
                "branch": "signal" if is_trade else "wait",
                "section": "入场信号",
                "bar_range": "K1",
            },
            {
                "node_id": "10.1",
                "question": "是否有明确止损位？",
                "answer": "是" if is_trade else "否",
                "reason": "测试止损位",
                "branch": "stop" if is_trade else "wait",
                "section": "止损",
                "bar_range": "K1",
            },
            {
                "node_id": "10.2",
                "question": "是否有明确目标位？",
                "answer": "是" if is_trade else "否",
                "reason": "测试目标位",
                "branch": "target" if is_trade else "wait",
                "section": "目标",
                "bar_range": "K1",
            },
            {
                "node_id": "10.3",
                "question": "交易者方程是否通过？",
                "answer": "是" if is_trade else "否",
                "reason": "测试 trace",
                "branch": "trade" if is_trade else "wait",
                "section": "交易者方程",
                "bar_range": "K1",
            },
            {
                "node_id": "11.1",
                "question": "是否使用当前下单方式？",
                "answer": "是" if is_trade else "不适用",
                "reason": "测试下单方式",
                "branch": order_type if is_trade else None,
                "section": "下单方式",
                "bar_range": "K1",
            }
        ],
        "terminal": {
            "node_id": "11.1",
            "outcome": "trade" if is_trade else "wait",
            "label": "下单" if is_trade else "不下单",
        },
        "next_cycle_prediction": {
            "cycle": "normal_channel",
            "direction": "bullish",
            "probabilities": {
                "spike": 5,
                "micro_channel": 10,
                "tight_channel": 15,
                "normal_channel": 30,
                "broad_channel": 15,
                "trending_tr": 10,
                "trading_range": 10,
                "extreme_tr": 5,
            },
            "reasoning": "延续当前结构",
            "unpredictable": False,
            "features_used": ["stage1_diagnosis", "stage2_decision"],
        },
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
    def _features(self):
        df = _df_from_ohlc([(100.0, 104.0, 99.0, 103.0)] * 6)
        return build_price_action_features(
            df,
            symbol="BTC/USDT",
            timeframe="1h",
            market="crypto",
            window=6,
            warmup=0,
            now=pd.Timestamp("2026-06-01 14:30", tz="UTC"),
        )

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
        assert "1.1、1.2、1.3、2.1、2.2、2.3、2.4、2.5" in content
        assert '"market_state"' not in content
        assert '"signal_chain"' not in content

    def test_trade_decision_prompt_includes_decision_stance(self):
        features = self._features()
        messages = build_trade_decision_messages(
            features=features,
            diagnosis=_market_diagnosis(features),
            strategies=[],
            decision_stance="aggressive",
        )
        content = messages[1]["content"]

        assert "用户交易倾向" in content
        assert "aggressive" in content
        assert "交易倾向不能覆盖价格事实" in content

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
        features = self._features()
        diagnosis = _market_diagnosis(features)
        diagnosis["gate_trace"] = diagnosis["gate_trace"][:2]

        errors = validate_market_diagnosis(diagnosis, feature_rows=features.rows)

        assert "market_diagnosis_gate_trace_missing_proceed_nodes" in errors

    def test_market_diagnosis_validation_rejects_trace_answer_and_order_errors(self):
        features = self._features()
        diagnosis = _market_diagnosis(features)
        diagnosis["gate_trace"][0]["answer"] = "maybe"
        diagnosis["gate_trace"][2], diagnosis["gate_trace"][3] = (
            diagnosis["gate_trace"][3],
            diagnosis["gate_trace"][2],
        )

        errors = validate_market_diagnosis(diagnosis, feature_rows=features.rows)

        assert "market_diagnosis_gate_trace_answer_invalid" in errors
        assert "market_diagnosis_gate_trace_node_order_invalid" in errors

    def test_market_diagnosis_validation_rejects_trace_bar_range_out_of_frame(self):
        features = self._features()
        diagnosis = _market_diagnosis(features)
        diagnosis["gate_trace"][0]["bar_range"] = "K7-K1"

        errors = validate_market_diagnosis(diagnosis, feature_rows=features.rows)

        assert "market_diagnosis_gate_trace_bar_range_out_of_frame" in errors

    def test_market_diagnosis_validation_rejects_trace_branch_conflicts(self):
        features = self._features()
        diagnosis = _market_diagnosis(features)
        diagnosis["gate_trace"][1]["branch"] = "spike"
        diagnosis["gate_trace"][5]["branch"] = "bearish"

        errors = validate_market_diagnosis(diagnosis, feature_rows=features.rows)

        assert "market_diagnosis_gate_trace_cycle_branch_conflict" in errors
        assert "market_diagnosis_gate_trace_direction_branch_conflict" in errors

    def test_market_diagnosis_validation_rejects_schema_enum_errors(self):
        features = self._features()
        diagnosis = _market_diagnosis(features)
        diagnosis["support_levels"] = [100]
        diagnosis["market_phase"] = "transitioning"
        diagnosis["transition_risk"] = None
        diagnosis["bar_by_bar_summary"][0]["role"] = "summary"
        diagnosis["bar_by_bar_summary"][0]["follow_through"] = "later"

        errors = validate_market_diagnosis(diagnosis, feature_rows=features.rows)

        assert "market_diagnosis_support_levels_items_must_be_strings" in errors
        assert "market_diagnosis_transitioning_requires_transition_risk" in errors
        assert "market_diagnosis_bar_by_bar_role_invalid" in errors
        assert "market_diagnosis_bar_by_bar_follow_through_invalid" in errors


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
            "上涨通道分析识别.txt",
            "上涨通道交易策略.txt",
            "文件13-窄通道与宽通道策略.txt",
            "文件15-二次入场机会.txt",
            "文件19-H1H2-L1L2计数.txt",
        ]

    def test_routes_breakout_pullback_overlay_without_old_gate_shape(self):
        diagnosis = {
            "cycle_position": "normal_channel",
            "direction": "bullish",
            "detected_patterns": ["breakout_up"],
            "entry_setup": "breakout_pullback",
            "bar_analysis": {"entry_setup_type": "breakout_pullback"},
        }

        routed = route_strategies(diagnosis)

        assert [template.template_id for template in routed] == [
            "上涨通道分析识别.txt",
            "上涨通道交易策略.txt",
            "文件13-窄通道与宽通道策略.txt",
            "文件15-二次入场机会.txt",
            "文件19-H1H2-L1L2计数.txt",
            "文件18-突破失败与突破测试.txt",
        ]

    def test_routes_spike_ending_with_channel_playbooks(self):
        diagnosis = {
            "cycle_position": "spike",
            "direction": "bearish",
            "spike_stage": "ending",
            "detected_patterns": [],
        }

        routed = route_strategies(diagnosis)

        assert [template.template_id for template in routed] == [
            "极速下跌分析识别.txt",
            "极速下跌交易策略.txt",
            "下跌通道分析识别.txt",
            "下跌通道交易策略.txt",
            "文件13-窄通道与宽通道策略.txt",
        ]

    def test_routes_alternative_cycle_and_recent_spike_then_dedupes(self):
        diagnosis = {
            "cycle_position": "trading_range",
            "alternative_cycle_position": "normal_channel",
            "direction": "bullish",
            "trend_context": {"recent_spike": "bullish"},
            "detected_patterns": ["barbwire"],
        }

        routed = route_strategies(diagnosis)

        assert [template.template_id for template in routed] == [
            "震荡区间分析识别.txt",
            "震荡区间交易策略.txt",
            "极速上涨分析识别.txt",
            "极速上涨交易策略.txt",
            "上涨通道分析识别.txt",
            "上涨通道交易策略.txt",
            "文件13-窄通道与宽通道策略.txt",
            "文件21-铁丝网与无交易环境.txt",
        ]

    def test_routes_pattern_overlays_from_keywords(self):
        diagnosis = {
            "cycle_position": "unknown",
            "direction": "neutral",
            "detected_patterns": ["wedge", "mtr", "final_flag"],
            "key_signals": [
                "H2 计数入场后突破失败，形成上升三角形和双顶，"
                "Always In 背景转弱，信号失败后价格靠近磁力位",
            ],
        }

        routed = route_strategies(diagnosis)

        assert [template.template_id for template in routed] == [
            "文件14-楔形形态分析交易.txt",
            "文件15-二次入场机会.txt",
            "文件25-主要趋势反转MTR.txt",
            "文件24-最终旗形与趋势末端.txt",
            "文件19-H1H2-L1L2计数.txt",
            "文件18-突破失败与突破测试.txt",
            "文件20-AlwaysIn与20GB.txt",
            "文件22-信号失败后的磁力位.txt",
            "文件27-三角形与收敛形态.txt",
            "文件28-双重顶底与微型结构.txt",
        ]

    def test_routes_unknown_and_extreme_to_empty_when_no_overlay(self):
        assert route_strategies({"cycle_position": "unknown", "direction": "neutral"}) == []
        assert route_strategies({"cycle_position": "extreme_tr", "direction": "bearish"}) == []

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


class TestSyncGate23WithDirection:
    """validate_market_diagnosis 应就地归一化 node 2.3 的 branch/answer,
    避免因 LLM 把 branch 填成 signal_quality 术语等偏差直接判 invalid。
    参考 PA_Agent ``_sync_gate_23_answer_with_direction``。
    """

    def _diagnosis_with_2_3(
        self,
        *,
        direction="neutral",
        branch="no_valid_breakout",
        answer="否",
        reason="",
        skipped=False,
        cycle_position="trading_range",
    ):
        gate_trace = [
            {"node_id": "1.1", "question": "q", "answer": "是", "reason": "r",
             "branch": None, "section": "数据检查", "bar_range": "K5-K1"},
            {"node_id": "1.2", "question": "q", "answer": "是", "reason": "r",
             "branch": cycle_position, "section": "周期识别", "bar_range": "K4-K1"},
            {"node_id": "1.3", "question": "q", "answer": "否", "reason": "r",
             "branch": None, "section": "周期识别", "bar_range": "K3-K1"},
            {"node_id": "2.1", "question": "q", "answer": "否", "reason": "r",
             "branch": direction, "section": "方向判断", "bar_range": "K4-K2"},
            {"node_id": "2.2", "question": "q", "answer": "中性", "reason": "r",
             "branch": direction, "section": "背景判断", "bar_range": "K5-K1"},
            {"node_id": "2.3", "question": "q", "answer": answer, "reason": reason,
             "branch": branch, "section": "方向判断", "bar_range": "K3-K1",
             **({"skipped": True} if skipped else {})},
            {"node_id": "2.4", "question": "q", "answer": "否", "reason": "r",
             "branch": "low_quality", "section": "信号质量", "bar_range": "K2-K1"},
            {"node_id": "2.5", "question": "q", "answer": "中性", "reason": "r",
             "branch": direction, "section": "结构检查", "bar_range": "K8-K1"},
        ]
        return {
            "cycle_position": cycle_position,
            "alternative_cycle_position": None,
            "direction": direction,
            "diagnosis_confidence": 60,
            "spike_stage": None,
            "climax_risk": None,
            "market_phase": "stable",
            "transition_risk": None,
            "detected_patterns": ["trading_range"],
            "key_signals": [],
            "htf_context": "",
            "entry_setup": "none",
            "support_levels": [],
            "resistance_levels": [],
            "strategy_files_needed": [],
            "risk_warning": None,
            "bar_analysis": [],
            "bar_by_bar_summary": [],
            "gate_trace": gate_trace,
            "gate_result": "proceed",
        }

    def test_branch_no_valid_breakout_gets_corrected_to_top_direction(self):
        d = self._diagnosis_with_2_3(
            direction="neutral", branch="no_valid_breakout", answer="否"
        )
        errors = validate_market_diagnosis(d)
        assert "market_diagnosis_gate_trace_direction_branch_conflict" not in errors
        node_23 = next(i for i in d["gate_trace"] if i["node_id"] == "2.3")
        assert node_23["branch"] == "neutral"
        assert node_23["answer"] == "中性"

    def test_branch_missing_infers_bullish_from_reason(self):
        d = self._diagnosis_with_2_3(
            direction="bullish",
            branch=None,
            answer="中性",
            reason="多头 follow_through 强势,突破上沿",
        )
        errors = validate_market_diagnosis(d)
        assert "market_diagnosis_gate_trace_direction_branch_conflict" not in errors
        node_23 = next(i for i in d["gate_trace"] if i["node_id"] == "2.3")
        assert node_23["branch"] == "bullish"
        assert node_23["answer"] == "是"

    def test_bullish_branch_neutral_answer_corrected_to_yes(self):
        d = self._diagnosis_with_2_3(
            direction="bullish", branch="bullish", answer="中性"
        )
        validate_market_diagnosis(d)
        node_23 = next(i for i in d["gate_trace"] if i["node_id"] == "2.3")
        assert node_23["answer"] == "是"

    def test_neutral_branch_yes_answer_corrected_to_neutral(self):
        d = self._diagnosis_with_2_3(
            direction="neutral", branch="neutral", answer="是"
        )
        validate_market_diagnosis(d)
        node_23 = next(i for i in d["gate_trace"] if i["node_id"] == "2.3")
        assert node_23["answer"] == "中性"

    def test_no_change_when_already_consistent(self):
        d = self._diagnosis_with_2_3(
            direction="neutral", branch="neutral", answer="中性"
        )
        errors = validate_market_diagnosis(d)
        assert "market_diagnosis_gate_trace_direction_branch_conflict" not in errors

    def test_skipped_node_not_modified(self):
        d = self._diagnosis_with_2_3(
            direction="bullish", branch=None, answer="", skipped=True
        )
        validate_market_diagnosis(d)
        node_23 = next(i for i in d["gate_trace"] if i["node_id"] == "2.3")
        assert node_23.get("branch") is None
        assert node_23.get("skipped") is True


class TestSyncGate12WithCycle:
    """validate_market_diagnosis 应就地归一化 node 1.2 的 branch,
    避免因 LLM 把 branch 填成 signal 阶段术语(如 tr_identified)或 yes/no
    直接判 invalid。参考 PA_Agent ``_sync_gate_12_branch_with_cycle``。
    """

    def _diagnosis_with_1_2(
        self,
        *,
        cycle_position="trading_range",
        branch="tr_identified",
        answer="是",
        skipped=False,
        direction="neutral",
    ):
        gate_trace = [
            {"node_id": "1.1", "question": "q", "answer": "是", "reason": "r",
             "branch": None, "section": "数据检查", "bar_range": "K5-K1"},
            {"node_id": "1.2", "question": "q", "answer": answer, "reason": "r",
             "branch": branch, "section": "周期识别", "bar_range": "K4-K1",
             **({"skipped": True} if skipped else {})},
            {"node_id": "1.3", "question": "q", "answer": "否", "reason": "r",
             "branch": None, "section": "周期识别", "bar_range": "K3-K1"},
            {"node_id": "2.1", "question": "q", "answer": "否", "reason": "r",
             "branch": direction, "section": "方向判断", "bar_range": "K4-K2"},
            {"node_id": "2.2", "question": "q", "answer": "中性", "reason": "r",
             "branch": direction, "section": "背景判断", "bar_range": "K5-K1"},
            {"node_id": "2.3", "question": "q", "answer": "中性", "reason": "r",
             "branch": direction, "section": "方向判断", "bar_range": "K3-K1"},
            {"node_id": "2.4", "question": "q", "answer": "否", "reason": "r",
             "branch": "low_quality", "section": "信号质量", "bar_range": "K2-K1"},
            {"node_id": "2.5", "question": "q", "answer": "中性", "reason": "r",
             "branch": direction, "section": "结构检查", "bar_range": "K8-K1"},
        ]
        return {
            "cycle_position": cycle_position,
            "alternative_cycle_position": None,
            "direction": direction,
            "diagnosis_confidence": 60,
            "spike_stage": None,
            "climax_risk": None,
            "market_phase": "stable",
            "transition_risk": None,
            "detected_patterns": ["trading_range"],
            "key_signals": [],
            "htf_context": "",
            "entry_setup": "none",
            "support_levels": [],
            "resistance_levels": [],
            "strategy_files_needed": [],
            "risk_warning": None,
            "bar_analysis": [],
            "bar_by_bar_summary": [],
            "gate_trace": gate_trace,
            "gate_result": "proceed",
        }

    def test_branch_tr_identified_gets_corrected_to_cycle(self):
        """LLM 把 1.2 branch 填成 tr_identified(signal 术语) → 修正为顶层 cycle。"""
        d = self._diagnosis_with_1_2(
            cycle_position="trading_range", branch="tr_identified", answer="是"
        )
        errors = validate_market_diagnosis(d)
        assert "market_diagnosis_gate_trace_cycle_branch_conflict" not in errors
        node_12 = next(i for i in d["gate_trace"] if i["node_id"] == "1.2")
        assert node_12["branch"] == "trading_range"

    def test_branch_yes_gets_corrected_when_answer_is_yes(self):
        """branch=yes, answer=是 → branch 改成 cycle_position。"""
        d = self._diagnosis_with_1_2(
            cycle_position="spike", branch="yes", answer="是"
        )
        errors = validate_market_diagnosis(d)
        assert "market_diagnosis_gate_trace_cycle_branch_conflict" not in errors
        node_12 = next(i for i in d["gate_trace"] if i["node_id"] == "1.2")
        assert node_12["branch"] == "spike"

    def test_branch_unknown_when_answer_is_no(self):
        """answer=否 时,branch 应改为 unknown。"""
        d = self._diagnosis_with_1_2(
            cycle_position="trading_range", branch="tr_identified", answer="否"
        )
        errors = validate_market_diagnosis(d)
        assert "market_diagnosis_gate_trace_cycle_branch_conflict" not in errors
        node_12 = next(i for i in d["gate_trace"] if i["node_id"] == "1.2")
        assert node_12["branch"] == "unknown"

    def test_valid_branch_not_modified(self):
        """branch 已经是合法 cycle 名(如 spike) → 不动。"""
        d = self._diagnosis_with_1_2(
            cycle_position="spike", branch="spike", answer="是"
        )
        validate_market_diagnosis(d)
        node_12 = next(i for i in d["gate_trace"] if i["node_id"] == "1.2")
        assert node_12["branch"] == "spike"

    def test_chinese_cycle_alias_normalized(self):
        """branch=交易区间 → 归一化为 trading_range,不报冲突。"""
        d = self._diagnosis_with_1_2(
            cycle_position="trading_range", branch="交易区间", answer="是"
        )
        errors = validate_market_diagnosis(d)
        assert "market_diagnosis_gate_trace_cycle_branch_conflict" not in errors
        node_12 = next(i for i in d["gate_trace"] if i["node_id"] == "1.2")
        assert node_12["branch"] == "trading_range"

    def test_skipped_node_not_modified(self):
        d = self._diagnosis_with_1_2(
            cycle_position="trading_range", branch=None, answer="", skipped=True
        )
        validate_market_diagnosis(d)
        node_12 = next(i for i in d["gate_trace"] if i["node_id"] == "1.2")
        assert node_12.get("branch") is None
        assert node_12.get("skipped") is True


class TestRepairGateResult:
    """gate_result=wait/unknown 在无阻塞条件时自动改写为 proceed。

    参考 PA_Agent ``_repair_gate_result``(trace_normalize.py:778-811)。
    常见 LLM 偏差:周期/方向已识别但 2.x 节点出现"否",LLM 把 gate_result
    误设为 wait。这里测试 validate_market_diagnosis 内嵌的修正行为。
    """

    @staticmethod
    def _diagnosis(*, gate_result: str, node_12_answer: str, node_13_answer: str) -> dict:
        return {
            "cycle_position": "trading_range",
            "direction": "neutral",
            "diagnosis_confidence": 60,
            "spike_stage": None,
            "climax_risk": "none",
            "market_phase": "trending",
            "transition_risk": "low",
            "detected_patterns": ["trend_bull"],
            "key_signals": ["K1 trend_bull"],
            "htf_context": "ok",
            "entry_setup": "breakout_pullback",
            "support_levels": ["100.0"],
            "resistance_levels": ["110.0"],
            "strategy_files_needed": ["a.md"],
            "risk_warning": "n/a",
            "bar_analysis": {
                "always_in": "long",
                "last_closed_bar": "K1",
                "bar_type": "trend_bull",
                "signal_bar": {"bar": "K1", "quality": "medium", "pattern": "p", "reason": "r"},
                "entry_setup_type": "breakout_pullback",
                "follow_through": "pending",
            },
            "bar_by_bar_summary": [
                {"bar": "K5", "role": "noise", "bar_type": "doji", "context_effect": "neutral",
                 "follow_through": "no", "trapped_side": "none", "reason": "r"},
                {"bar": "K4", "role": "noise", "bar_type": "doji", "context_effect": "neutral",
                 "follow_through": "no", "trapped_side": "none", "reason": "r"},
                {"bar": "K3", "role": "noise", "bar_type": "doji", "context_effect": "neutral",
                 "follow_through": "no", "trapped_side": "none", "reason": "r"},
                {"bar": "K2", "role": "noise", "bar_type": "doji", "context_effect": "neutral",
                 "follow_through": "no", "trapped_side": "none", "reason": "r"},
                {"bar": "K1", "role": "signal", "bar_type": "trend_bull", "context_effect": "strengthens_bull",
                 "follow_through": "pending", "trapped_side": "bears", "reason": "r"},
            ],
            "gate_trace": [
                {"node_id": "1.1", "question": "q", "answer": "否", "branch": "trading_range",
                 "section": "周期识别", "bar_range": "K5-K1", "reason": "r"},
                {"node_id": "1.2", "question": "q", "answer": node_12_answer, "branch": "trading_range",
                 "section": "周期识别", "bar_range": "K8-K1", "reason": "r"},
                {"node_id": "1.3", "question": "q", "answer": node_13_answer, "branch": None,
                 "section": "周期识别", "bar_range": "K8-K1", "reason": "r"},
                {"node_id": "2.1", "question": "q", "answer": "是", "branch": "bullish",
                 "section": "方向与闸门", "bar_range": "K1", "reason": "r"},
                {"node_id": "2.2", "question": "q", "answer": "是", "branch": None,
                 "section": "方向与闸门", "bar_range": "K1", "reason": "r"},
                {"node_id": "2.3", "question": "q", "answer": "是", "branch": "bullish",
                 "section": "方向与闸门", "bar_range": "K1", "reason": "r"},
                {"node_id": "2.4", "question": "q", "answer": "否", "branch": None,
                 "section": "方向与闸门", "bar_range": "K1", "reason": "r"},
                {"node_id": "2.5", "question": "q", "answer": "是", "branch": None,
                 "section": "方向与闸门", "bar_range": "K8-K1", "reason": "r"},
            ],
            "gate_result": gate_result,
        }

    def test_wait_with_no_blocking_condition_repaired_to_proceed(self):
        """ETH 4h 那种场景:1.2=是、1.3=否,但 LLM 输出 wait + 末节点为是 → 自动改为 proceed,校验通过。"""
        d = self._diagnosis(gate_result="wait", node_12_answer="是", node_13_answer="否")
        errors = validate_market_diagnosis(d)
        assert d["gate_result"] == "proceed"
        assert "market_diagnosis_gate_wait_requires_negative_or_waiting_final_answer" not in errors

    def test_wait_with_node_12_block_is_preserved(self):
        """1.2=否 时 wait 应保留,但末节点必须 否/等待,否则报错。"""
        d = self._diagnosis(gate_result="wait", node_12_answer="否", node_13_answer="否")
        validate_market_diagnosis(d)
        assert d["gate_result"] == "wait"  # 1.2 阻塞条件成立,不改

    def test_wait_with_node_13_block_is_preserved(self):
        """1.3=是 时 wait 应保留。"""
        d = self._diagnosis(gate_result="wait", node_12_answer="是", node_13_answer="是")
        validate_market_diagnosis(d)
        assert d["gate_result"] == "wait"

    def test_unknown_with_no_blocking_condition_repaired_to_proceed(self):
        """gate_result=unknown 同样在无阻塞时改为 proceed。"""
        d = self._diagnosis(gate_result="unknown", node_12_answer="是", node_13_answer="否")
        errors = validate_market_diagnosis(d)
        assert d["gate_result"] == "proceed"
        assert "market_diagnosis_gate_wait_requires_negative_or_waiting_final_answer" not in errors

    def test_proceed_is_not_modified(self):
        """proceed 永远不改。"""
        d = self._diagnosis(gate_result="proceed", node_12_answer="是", node_13_answer="否")
        validate_market_diagnosis(d)
        assert d["gate_result"] == "proceed"


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
        assert outcome.decision["decision"]["order_type"] == "不下单"
        assert outcome.decision["terminal"]["outcome"] == "wait"
        assert outcome.decision["gate_shortcircuited"] is True
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
        decision = _trade_decision(order_type="不下单", diagnosis=diagnosis)
        llm = MagicMock()
        llm.model = "test-model"
        llm.base_url = "https://example.test"
        responses = [
            (
                "market_diagnosis",
                json.dumps(diagnosis, ensure_ascii=False),
                {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            ),
            (
                "trade_decision",
                json.dumps(decision, ensure_ascii=False),
                {
                    "prompt_tokens": 80,
                    "cached_prompt_tokens": 10,
                    "completion_tokens": 30,
                    "total_tokens": 110,
                },
            ),
        ]

        def complete_json(_messages, *, stage):
            expected_stage, content, usage = responses.pop(0)
            assert stage == expected_stage
            llm.last_response = {
                "stage": stage,
                "model": llm.model,
                "content": content,
                "usage": usage,
            }
            return content

        llm.complete_json.side_effect = complete_json
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
        assert '"order_direction"' in kwargs["trade_decision_messages"][1]["content"]
        assert '"terminal"' in kwargs["trade_decision_messages"][1]["content"]
        assert '"next_cycle_prediction"' in kwargs["trade_decision_messages"][1]["content"]
        assert kwargs["prompt_metadata"]["prompt_templates"]["market_diagnosis"][0]["sha256"]
        assert kwargs["raw_responses"]["market_diagnosis"]["content"] == json.dumps(
            diagnosis,
            ensure_ascii=False,
        )
        assert kwargs["raw_responses"]["trade_decision"]["content"] == json.dumps(
            decision,
            ensure_ascii=False,
        )
        assert kwargs["usage_total"] == {
            "prompt_tokens": 180,
            "cached_prompt_tokens": 10,
            "completion_tokens": 50,
            "total_tokens": 230,
        }
        assert kwargs["exception"] is None

    def test_actionable_trade_flows_to_save_and_telegram_notification(self):
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
        decision = _trade_decision(
            order_type="突破单",
            entry=105,
            stop=101,
            tp1=109,
            tp2=111,
            estimated_win_rate=60,
            diagnosis=diagnosis,
        )
        responses = [
            ("market_diagnosis", json.dumps(diagnosis, ensure_ascii=False)),
            ("trade_decision", json.dumps(decision, ensure_ascii=False)),
        ]
        llm = MagicMock()
        llm.model = "test-model"
        llm.base_url = "https://example.test"

        def complete_json(_messages, *, stage):
            expected_stage, content = responses.pop(0)
            assert stage == expected_stage
            llm.last_response = {
                "stage": stage,
                "model": llm.model,
                "content": content,
                "usage": {},
            }
            return content

        llm.complete_json.side_effect = complete_json
        repository = MagicMock()
        repository.get_previous_successful_analysis.return_value = None
        repository.query_experience.return_value = [{"id": 1, "title": "case"}]
        repository.save_analysis.return_value = True
        notifier = MagicMock()
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
            notifier=notifier,
            chart_generator=lambda *args, **kwargs: b"chart",
        )

        assert outcome.status == "success"
        assert outcome.decision["decision"]["order_type"] == "突破单"
        save_kwargs = repository.save_analysis.call_args.kwargs
        assert save_kwargs["status"] == "success"
        assert save_kwargs["validation_status"] == "valid"
        assert save_kwargs["trade_decision"] == outcome.decision
        assert save_kwargs["experience_cases"] == [{"id": 1, "title": "case"}]
        notifier.notify_decision.assert_called_once()
        notify_args = notifier.notify_decision.call_args
        assert notify_args.args[0] == "BTC/USDT"
        payload = notify_args.args[2]
        assert payload["decision"] == outcome.decision
        assert payload["validation"]["valid"] is True
        assert payload["diagnosis"] == diagnosis
        assert payload["experience_cases"] == [{"id": 1, "title": "case"}]

    def test_market_diagnosis_validation_retry_uses_feedback(self):
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
        bad_diagnosis = {
            "cycle_position": "normal_channel",
            "direction": "bullish",
            "gate_result": "proceed",
        }
        good_diagnosis = _market_diagnosis(features)
        decision = _trade_decision(order_type="不下单", diagnosis=good_diagnosis)
        responses = [
            ("market_diagnosis", json.dumps(bad_diagnosis, ensure_ascii=False)),
            ("market_diagnosis", json.dumps(good_diagnosis, ensure_ascii=False)),
            ("trade_decision", json.dumps(decision, ensure_ascii=False)),
        ]
        llm = MagicMock()
        llm.model = "test-model"
        llm.base_url = "https://example.test"
        seen_messages = []

        def complete_json(messages, *, stage):
            seen_messages.append(messages)
            expected_stage, content = responses.pop(0)
            assert stage == expected_stage
            llm.last_response = {
                "stage": stage,
                "model": llm.model,
                "content": content,
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
            return content

        llm.complete_json.side_effect = complete_json
        repository = MagicMock()
        repository.get_previous_successful_analysis.return_value = None
        repository.query_experience.return_value = []
        repository.save_analysis.return_value = True
        orchestrator = PriceActionOrchestrator(
            repository=repository,
            llm_client=llm,
            config={
                "pa_llm_window": 6,
                "pa_llm_warmup": 0,
                "pa_validation_retry_max": 1,
            },
        )

        outcome = orchestrator.analyze(
            symbol="BTC/USDT",
            dataframe=df,
            timeframe="1h",
            market="crypto",
        )

        assert outcome.status == "success"
        assert llm.complete_json.call_count == 3
        # Retry feedback is appended as the last user message. New structured
        # feedback uses a "## 校验未通过" header and translates error codes.
        feedback_msg = seen_messages[1][-1]
        assert feedback_msg["role"] == "user"
        assert "校验未通过" in feedback_msg["content"]
        # The previous (failed) assistant turn must be re-injected before the
        # feedback so the LLM can self-correct against its own prior output.
        assert seen_messages[1][-2]["role"] == "assistant"
        assert json.dumps(bad_diagnosis, ensure_ascii=False) in seen_messages[1][-2]["content"]
        raw = repository.save_analysis.call_args.kwargs["raw_responses"]
        assert len(raw["market_diagnosis"]["retry_attempts"]) == 1

    def test_trade_decision_validation_retry_uses_feedback(self):
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
        invalid_decision = _trade_decision(
            order_direction="做多",
            entry=100,
            stop=110,
            tp1=120,
            tp2=130,
            diagnosis=diagnosis,
        )
        valid_decision = _trade_decision(order_type="不下单", diagnosis=diagnosis)
        responses = [
            ("market_diagnosis", json.dumps(diagnosis, ensure_ascii=False)),
            ("trade_decision", json.dumps(invalid_decision, ensure_ascii=False)),
            ("trade_decision", json.dumps(valid_decision, ensure_ascii=False)),
        ]
        llm = MagicMock()
        llm.model = "test-model"
        llm.base_url = "https://example.test"
        seen_messages = []

        def complete_json(messages, *, stage):
            seen_messages.append(messages)
            expected_stage, content = responses.pop(0)
            assert stage == expected_stage
            llm.last_response = {
                "stage": stage,
                "model": llm.model,
                "content": content,
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
            return content

        llm.complete_json.side_effect = complete_json
        repository = MagicMock()
        repository.get_previous_successful_analysis.return_value = None
        repository.query_experience.return_value = []
        repository.save_analysis.return_value = True
        orchestrator = PriceActionOrchestrator(
            repository=repository,
            llm_client=llm,
            config={
                "pa_llm_window": 6,
                "pa_llm_warmup": 0,
                "pa_validation_retry_max": 1,
            },
        )

        outcome = orchestrator.analyze(
            symbol="BTC/USDT",
            dataframe=df,
            timeframe="1h",
            market="crypto",
        )

        assert outcome.status == "success"
        assert llm.complete_json.call_count == 3
        # New structured feedback header + previous assistant turn re-injected.
        feedback_msg = seen_messages[2][-1]
        assert feedback_msg["role"] == "user"
        assert "校验未通过" in feedback_msg["content"]
        assert seen_messages[2][-2]["role"] == "assistant"
        assert json.dumps(invalid_decision, ensure_ascii=False) in seen_messages[2][-2]["content"]
        raw = repository.save_analysis.call_args.kwargs["raw_responses"]
        assert len(raw["trade_decision"]["retry_attempts"]) == 1

    def test_incremental_stage1_prompt_uses_previous_successful_analysis(self):
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
        decision = _trade_decision(order_type="不下单", diagnosis=diagnosis)
        previous = {
            "candle_time": features.rows[1]["time"],
            "diagnosis": diagnosis,
            "decision": decision,
            "market_diagnosis_messages": [
                {"role": "system", "content": "previous system"},
                {"role": "user", "content": "previous stage1 user"},
            ],
            "raw_responses": {
                "market_diagnosis": {
                    "content": json.dumps(diagnosis, ensure_ascii=False),
                }
            },
        }
        responses = [
            ("market_diagnosis", json.dumps(diagnosis, ensure_ascii=False)),
            ("trade_decision", json.dumps(decision, ensure_ascii=False)),
        ]
        llm = MagicMock()
        llm.model = "test-model"
        llm.base_url = "https://example.test"
        seen_messages = []

        def complete_json(messages, *, stage):
            seen_messages.append(messages)
            expected_stage, content = responses.pop(0)
            assert stage == expected_stage
            llm.last_response = {
                "stage": stage,
                "model": llm.model,
                "content": content,
                "usage": {},
            }
            return content

        llm.complete_json.side_effect = complete_json
        repository = MagicMock()
        repository.get_previous_successful_analysis.return_value = previous
        repository.query_experience.return_value = []
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
        assert [message["role"] for message in seen_messages[0]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert "增量更新阶段一市场诊断" in seen_messages[0][-1]["content"]
        kwargs = repository.save_analysis.call_args.kwargs
        assert kwargs["prompt_metadata"]["market_diagnosis_mode"] == "incremental"
        assert kwargs["prompt_metadata"]["incremental_new_bar_count"] == 1

    def test_failed_market_diagnosis_persists_exception_and_partial_record(self):
        df = _df_from_ohlc([(100.0, 104.0, 99.0, 103.0)] * 6)
        bad_diagnosis = {
            "cycle_position": "normal_channel",
            "direction": "bullish",
            "gate_result": "proceed",
        }
        raw_diagnosis = json.dumps(bad_diagnosis, ensure_ascii=False)
        llm = MagicMock()
        llm.model = "test-model"
        llm.base_url = "https://example.test"
        llm.last_response = {
            "stage": "market_diagnosis",
            "model": llm.model,
            "content": raw_diagnosis,
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }
        llm.complete_json.return_value = raw_diagnosis
        repository = MagicMock()
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

        assert outcome.status == "failed"
        kwargs = repository.save_analysis.call_args.kwargs
        assert kwargs["status"] == "failed"
        assert kwargs["raw_responses"]["market_diagnosis"]["content"] == raw_diagnosis
        assert kwargs["usage_total"] == {
            "prompt_tokens": 50,
            "cached_prompt_tokens": 0,
            "completion_tokens": 10,
            "total_tokens": 60,
        }
        assert kwargs["exception"]["stage"] == "market_diagnosis"
        assert kwargs["exception"]["type"] == "ValueError"


class TestOpenAIJsonClient:
    def test_uses_json_object_response_format(self):
        completions = MagicMock()
        completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"ok": true}',
                        reasoning_content="reasoning",
                        role="assistant",
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            ),
            model="deepseek-chat",
            id="chatcmpl-test",
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
        assert "max_tokens" not in kwargs
        assert client.last_response == {
            "stage": "test",
            "model": "deepseek-chat",
            "content": '{"ok": true}',
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 3},
                "cached_prompt_tokens": 3,
            },
            "reasoning_content": "reasoning",
            "role": "assistant",
            "id": "chatcmpl-test",
        }

    def test_uses_configured_max_tokens(self):
        completions = MagicMock()
        completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage=SimpleNamespace(),
            model="glm-5-turbo",
        )
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        client = OpenAIJsonClient(
            base_url="https://example.test/v1",
            api_key="test",
            model="glm-5-turbo",
            max_tokens=2048,
            client=fake_client,
        )

        client.complete_json([{"role": "user", "content": "x"}], stage="test")

        assert completions.create.call_args.kwargs["max_tokens"] == 2048

    def test_from_config_reads_runtime_env_budget(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "sk-env",
                "PA_LLM_TIMEOUT": "180",
                "PA_LLM_MAX_TOKENS": "2048",
            },
            clear=False,
        ):
            client = OpenAIJsonClient.from_config({})

        assert client.api_key == "sk-env"
        assert client.timeout == 180
        assert client.max_tokens == 2048


class TestAnthropicJsonClient:
    def _fake_anthropic_response(
        self,
        *,
        content: str = '{"ok": true}',
        model: str = "glm-5-turbo",
        input_tokens: int = 12,
        output_tokens: int = 6,
        cache_read: int | None = 5,
        response_id: str = "msg-test",
    ):
        usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if cache_read is not None:
            usage.cache_read_input_tokens = cache_read
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=content)],
            model=model,
            id=response_id,
            role="assistant",
            usage=usage,
        )

    def test_extracts_system_message_and_forwards_rest(self):
        messages_endpoint = MagicMock()
        messages_endpoint.create.return_value = self._fake_anthropic_response()
        fake_client = SimpleNamespace(messages=messages_endpoint)
        client = AnthropicJsonClient(
            base_url="https://ca.f-free.site",
            api_key="test",
            model="glm-5-turbo",
            client=fake_client,
        )

        client.complete_json(
            [
                {"role": "system", "content": "you are json bot"},
                {"role": "user", "content": "hi"},
            ],
            stage="test",
        )

        kwargs = messages_endpoint.create.call_args.kwargs
        assert kwargs["system"] == "you are json bot"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert "response_format" not in kwargs

    def test_last_response_shape_maps_anthropic_usage(self):
        messages_endpoint = MagicMock()
        messages_endpoint.create.return_value = self._fake_anthropic_response(
            input_tokens=12,
            output_tokens=6,
            cache_read=5,
            model="glm-5-turbo",
            response_id="msg-1",
        )
        fake_client = SimpleNamespace(messages=messages_endpoint)
        client = AnthropicJsonClient(
            base_url="https://ca.f-free.site",
            api_key="test",
            model="glm-5-turbo",
            client=fake_client,
        )

        content = client.complete_json(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            stage="market_diagnosis",
        )

        assert json.loads(content) == {"ok": True}
        assert client.last_response == {
            "stage": "market_diagnosis",
            "model": "glm-5-turbo",
            "content": '{"ok": true}',
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 6,
                "total_tokens": 18,
                "cached_prompt_tokens": 5,
            },
            "id": "msg-1",
            "role": "assistant",
        }

    def test_no_system_message_passes_only_messages(self):
        messages_endpoint = MagicMock()
        messages_endpoint.create.return_value = self._fake_anthropic_response()
        fake_client = SimpleNamespace(messages=messages_endpoint)
        client = AnthropicJsonClient(
            base_url="https://ca.f-free.site",
            api_key="test",
            model="glm-5-turbo",
            client=fake_client,
        )

        client.complete_json(
            [{"role": "user", "content": "hi"}],
            stage="test",
        )

        kwargs = messages_endpoint.create.call_args.kwargs
        assert "system" not in kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_from_config_reads_runtime_env_budget(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "sk-env",
                "PA_LLM_BASE_URL": "https://ca.f-free.site",
                "PA_LLM_MODEL": "glm-5-turbo",
                "PA_LLM_TIMEOUT": "180",
                "PA_LLM_MAX_TOKENS": "2048",
            },
            clear=False,
        ):
            client = AnthropicJsonClient.from_config({})

        assert client.api_key == "sk-env"
        assert client.base_url == "https://ca.f-free.site"
        assert client.model == "glm-5-turbo"
        assert client.timeout == 180
        assert client.max_tokens == 2048

    def test_client_passes_timeout_to_sdk(self):
        """Regression: Anthropic client must forward `timeout` to SDK constructor.

        Previously `timeout` was stored on the wrapper but never passed to the
        underlying Anthropic SDK, so dead connections hung indefinitely.
        """
        import anthropic

        with patch.object(anthropic, "Anthropic") as mock_anthropic:
            client = AnthropicJsonClient(
                base_url="https://ca.f-free.site/v1",
                api_key="test",
                model="glm-5.2",
                timeout=180.0,
            )
            client._client()

        mock_anthropic.assert_called_once()
        kwargs = mock_anthropic.call_args.kwargs
        assert kwargs["timeout"] == 180.0
        assert "ca.f-free.site" in kwargs["base_url"]


class TestBuildLlmClient:
    def test_anthropic_protocol_routes_to_anthropic_client(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "sk-env",
                "PA_LLM_BASE_URL": "https://ca.f-free.site",
                "PA_LLM_MODEL": "glm-5-turbo",
                "PA_LLM_PROTOCOL": "anthropic",
            },
            clear=False,
        ):
            client = build_llm_client({})

        assert isinstance(client, AnthropicJsonClient)

    def test_claude_alias_routes_to_anthropic_client(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "sk-env",
                "PA_LLM_PROTOCOL": "claude",
            },
            clear=False,
        ):
            client = build_llm_client({})

        assert isinstance(client, AnthropicJsonClient)

    def test_openai_protocol_is_default(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "sk-env",
                "PA_LLM_PROTOCOL": "",
            },
            clear=False,
        ):
            client = build_llm_client({})

        assert isinstance(client, OpenAIJsonClient)

    def test_explicit_openai_protocol(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "sk-env",
                "PA_LLM_PROTOCOL": "openai",
            },
            clear=False,
        ):
            client = build_llm_client({"pa_llm_protocol": "openai"})

        assert isinstance(client, OpenAIJsonClient)

    def test_config_overrides_env(self):
        with patch.dict(
            os.environ,
            {"PA_LLM_PROTOCOL": "openai", "DEEPSEEK_API_KEY": "sk-env"},
            clear=False,
        ):
            client = build_llm_client({"pa_llm_protocol": "anthropic"})

        assert isinstance(client, AnthropicJsonClient)


class TestRuntimeConfig:
    def test_config_int_uses_env_when_config_missing(self):
        with patch.dict(os.environ, {"PA_LLM_WINDOW": "8"}, clear=False):
            assert _config_int({}, "pa_llm_window", "PA_LLM_WINDOW", 30) == 8

    def test_config_int_prefers_explicit_config(self):
        with patch.dict(os.environ, {"PA_LLM_WINDOW": "8"}, clear=False):
            assert _config_int({"pa_llm_window": 6}, "pa_llm_window", "PA_LLM_WINDOW", 30) == 6


class TestJsonSyntaxRepair:
    """Test JSON repair layer in parse_json_object.

    Ported from PA_Agent json_validator.py:144-372. Verifies that common
    glm-5.x LLM output errors (unbalanced brackets, truncated, control
    chars in strings) get repaired at parse time without consuming retry.
    """

    def test_unbalanced_brackets_repaired(self):
        """Missing closing brace gets appended."""
        result = parse_json_object('{"a": 1')
        assert result == {"a": 1}

    def test_truncated_array_repaired(self):
        """Missing closing bracket for array."""
        result = parse_json_object('{"k": [1, 2')
        assert result == {"k": [1, 2]}

    def test_control_char_in_string_repaired(self):
        """Raw newline inside string literal gets escaped to \\n."""
        # Raw newline in the middle of a string value
        raw = '{"r": "line1\nline2"}'
        result = parse_json_object(raw)
        assert result == {"r": "line1\nline2"}

    def test_truncated_in_gate_trace_gets_tail_injected(self):
        """Truncated at a comma before gate_trace → AUTO stub appended."""
        raw = '{"cycle_position": "trading_range", '
        result = parse_json_object(raw)
        assert result["cycle_position"] == "trading_range"
        assert "gate_trace" in result
        assert result["gate_result"] == "unknown"

    def test_already_valid_json_not_modified(self):
        """Valid JSON parses unchanged."""
        result = parse_json_object('{"a": 1}')
        assert result == {"a": 1}

    def test_garbage_text_falls_through_to_extract(self):
        """Pure prose without leading { → falls through to _extract_json_object_text.

        parse_json_object raises ValueError if no JSON object can be extracted.
        """
        with pytest.raises(ValueError):
            parse_json_object("hello world no json here")

    def test_fenced_json_still_works(self):
        """Markdown-fenced JSON still parsed after repair layer."""
        raw = '```json\n{"a": 1}\n```'
        assert parse_json_object(raw) == {"a": 1}


class TestCategoryAwareRetry:
    """Tests for _categorize_errors and _retry_limit_for_errors.

    Verifies that JSON syntax errors (a) and missing fields (b) get the
    base retry limit, while schema/semantic conflicts (c) are capped at
    the semantic limit (default 1).
    """

    def test_json_syntax_error_is_category_a(self):
        errors = ["ValueError:JSONDecodeError:Expecting ',' delimiter: line 1"]
        assert _categorize_errors(errors) == "a"

    def test_invalid_json_token_is_category_a(self):
        errors = ["invalid_json:Expecting value"]
        assert _categorize_errors(errors) == "a"

    def test_schema_conflict_is_category_c(self):
        errors = ["market_diagnosis_gate_trace_cycle_branch_conflict"]
        assert _categorize_errors(errors) == "c"

    def test_missing_field_is_category_b(self):
        errors = ["market_diagnosis_direction_required"]
        assert _categorize_errors(errors) == "b"

    def test_empty_errors_is_ok(self):
        assert _categorize_errors([]) == "ok"

    def test_retry_limit_for_syntax_errors_uses_base(self):
        errors = ["ValueError:JSONDecodeError:..."]
        # config base = 3
        assert _retry_limit_for_errors(errors, {"pa_validation_retry_max": 3}) == 3

    def test_retry_limit_for_schema_conflict_uses_semantic(self):
        errors = ["market_diagnosis_gate_trace_cycle_branch_conflict"]
        # min(base=3, semantic=1) = 1
        assert _retry_limit_for_errors(errors, {"pa_validation_retry_max": 3}) == 1

    def test_retry_limit_for_empty_errors_is_zero(self):
        assert _retry_limit_for_errors([], {"pa_validation_retry_max": 3}) == 0


class TestValidationRetryLimitBound:
    """Upper bound of _validation_retry_limit raised from 3 → 5."""

    def test_retry_limit_clamped_to_5(self):
        """Setting retry_max=99 returns 5 (was previously 3)."""
        assert _validation_retry_limit({"pa_validation_retry_max": 99}) == 5

    def test_retry_limit_default_zero(self):
        assert _validation_retry_limit({}) == 0

    def test_retry_limit_within_bound(self):
        assert _validation_retry_limit({"pa_validation_retry_max": 4}) == 4

    def test_semantic_retry_limit_default_one(self):
        assert _semantic_retry_limit({}) == 1

    def test_semantic_retry_limit_clamped_to_three(self):
        assert _semantic_retry_limit({"pa_validation_retry_semantic_max": 99}) == 3


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
    feature_rows = [
        {"k": "K1", "high": 105.0, "low": 95.0},
        {"k": "K2", "high": 103.0, "low": 94.0},
        {"k": "K3", "high": 102.0, "low": 93.0},
    ]

    def test_json_syntax_stops_first(self):
        parsed, result = DecisionValidator().validate(
            "{bad",
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

        assert parsed is None
        assert result.checks == ["json_syntax"]

    def test_markdown_fenced_json_is_accepted(self):
        decision = _trade_decision(diagnosis=self.diagnosis)
        raw = "```json\n" + json.dumps(decision, ensure_ascii=False) + "\n```"

        parsed, result = DecisionValidator().validate(
            raw,
            diagnosis=self.diagnosis,
            price_action_features=self.features,
            feature_rows=self.feature_rows,
        )

        assert parsed == decision
        assert result.checks[0] == "json_syntax"
        assert not any(error.startswith("invalid_json") for error in result.errors)

    def test_parse_json_object_extracts_wrapped_object(self):
        raw = "analysis follows\n" + json.dumps({"ok": True}) + "\nfinished"

        assert parse_json_object(raw) == {"ok": True}

    def test_stage_consistency_runs_before_semantic(self):
        decision = _trade_decision(diagnosis=self.diagnosis)
        decision["diagnosis_summary"]["direction"] = "bearish"
        raw = json.dumps(decision)

        _, result = DecisionValidator().validate(
            raw,
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

        assert result.checks == ["json_syntax", "stage_consistency"]
        assert "diagnosis_summary_direction_mismatch" in result.errors

    def test_semantic_runs_before_numeric(self):
        raw = json.dumps(
            _trade_decision(
                order_direction="做多",
                entry=100,
                stop=110,
                tp1=120,
                tp2=130,
                trade_confidence=200,
                diagnosis=self.diagnosis,
            )
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
            _trade_decision(
                order_direction="做多",
                entry=100,
                stop=95,
                tp1=110,
                tp2=120,
                trade_confidence=200,
                diagnosis=self.diagnosis,
            )
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
        assert "trade_confidence_must_be_0_to_100_int" in result.errors

    def test_no_trade_requires_null_order_fields(self):
        decision = _trade_decision(order_type="不下单", diagnosis=self.diagnosis)
        decision["decision"]["entry_price"] = 100

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

        assert result.checks[-1] == "semantic_reasonableness"
        assert "no_trade_fields_must_be_null" in result.errors

    def test_breakout_order_requires_entry_basis(self):
        decision = _trade_decision(order_type="突破单", diagnosis=self.diagnosis)
        decision["decision"]["entry_basis_bar"] = None

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

        assert result.checks[-1] == "semantic_reasonableness"
        assert "breakout_order_requires_entry_basis_bar" in result.errors

    def test_valid_breakout_uses_feature_rows_for_basis_extreme(self):
        decision = _trade_decision(
            order_type="突破单",
            entry=106,
            stop=102,
            tp1=110,
            tp2=112,
            estimated_win_rate=60,
            diagnosis=self.diagnosis,
        )

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=self.diagnosis,
            price_action_features=self.features,
            feature_rows=self.feature_rows,
        )

        assert result.valid is True

    def test_long_breakout_entry_must_be_above_basis_high(self):
        decision = _trade_decision(
            order_type="突破单",
            entry=105,
            stop=99,
            tp1=111,
            tp2=113,
            estimated_win_rate=60,
            diagnosis=self.diagnosis,
        )

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=self.diagnosis,
            price_action_features=self.features,
            feature_rows=self.feature_rows,
        )

        assert result.checks[-1] == "semantic_reasonableness"
        assert "long_breakout_entry_must_be_above_basis_high" in result.errors

    def test_short_breakout_entry_must_be_below_basis_low(self):
        diagnosis = {**self.diagnosis, "direction": "bearish"}
        decision = _trade_decision(
            order_direction="做空",
            order_type="突破单",
            entry=95,
            stop=101,
            tp1=89,
            tp2=87,
            estimated_win_rate=60,
            diagnosis=diagnosis,
        )
        decision["decision"]["entry_basis_extreme"] = "low"
        decision["decision"]["entry_rule"] = "跌破 K1 低点"
        decision["next_cycle_prediction"]["direction"] = "bearish"

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=diagnosis,
            price_action_features=self.features,
            feature_rows=self.feature_rows,
        )

        assert result.checks[-1] == "semantic_reasonableness"
        assert "short_breakout_entry_must_be_below_basis_low" in result.errors

    def test_actionable_trade_requires_trader_equation_trace_node(self):
        decision = _trade_decision(diagnosis=self.diagnosis)
        decision["decision_trace"] = [
            item for item in decision["decision_trace"] if item["node_id"] != "10.3"
        ]

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

        assert result.checks[-1] == "semantic_reasonableness"
        assert "actionable_decision_requires_trader_equation_node_10_3" in result.errors

    def test_decision_trace_chapter_order_is_enforced(self):
        decision = _trade_decision(diagnosis=self.diagnosis)
        decision["decision_trace"] = [
            item for item in decision["decision_trace"] if item["node_id"] != "11.1"
        ]
        decision["decision_trace"].insert(
            0,
            {
                "node_id": "11.1",
                "question": "是否使用当前下单方式？",
                "answer": "是",
                "reason": "过早选择下单方式",
                "branch": "突破单",
                "section": "下单方式",
                "bar_range": "K1",
            },
        )

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

        assert result.checks[-1] == "semantic_reasonableness"
        assert "order_method_nodes_must_follow_trader_equation" in result.errors

    def test_order_direction_cannot_reverse_stage1_without_node_2_3(self):
        diagnosis = {**self.diagnosis, "direction": "bearish"}
        decision = _trade_decision(
            order_direction="做多",
            entry=100,
            stop=95,
            tp1=105,
            tp2=107,
            estimated_win_rate=60,
            diagnosis=diagnosis,
        )
        decision["next_cycle_prediction"]["direction"] = "bearish"

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=diagnosis,
            price_action_features=self.features,
        )

        assert result.checks[-1] == "semantic_reasonableness"
        assert (
            "order_direction_conflicts_with_stage1_direction_without_node_2_3"
            in result.errors
        )

    def test_entry_basis_bar_must_exist_in_feature_rows(self):
        decision = _trade_decision(
            order_type="突破单",
            entry=106,
            stop=102,
            tp1=110,
            tp2=112,
            estimated_win_rate=60,
            diagnosis=self.diagnosis,
        )
        decision["decision"]["entry_basis_bar"] = "K4"

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=self.diagnosis,
            price_action_features=self.features,
            feature_rows=self.feature_rows,
        )

        assert result.checks[-1] == "semantic_reasonableness"
        assert "breakout_entry_basis_bar_out_of_frame" in result.errors

    def test_decision_trace_bar_range_must_exist_in_feature_rows(self):
        decision = _trade_decision(
            order_type="突破单",
            entry=106,
            stop=102,
            tp1=110,
            tp2=112,
            estimated_win_rate=60,
            diagnosis=self.diagnosis,
        )
        decision["decision_trace"][0]["bar_range"] = "K4"

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=self.diagnosis,
            price_action_features=self.features,
            feature_rows=self.feature_rows,
        )

        assert result.checks[-1] == "semantic_reasonableness"
        assert "decision_trace_bar_range_out_of_frame" in result.errors

    def test_next_cycle_prediction_probability_sum_and_argmax_are_checked(self):
        decision = _trade_decision(
            order_type="突破单",
            entry=106,
            stop=102,
            tp1=110,
            tp2=112,
            estimated_win_rate=60,
            diagnosis=self.diagnosis,
        )
        decision["next_cycle_prediction"]["probabilities"] = {
            "spike": 40,
            "micro_channel": 5,
            "tight_channel": 5,
            "normal_channel": 10,
            "broad_channel": 5,
            "trending_tr": 5,
            "trading_range": 5,
            "extreme_tr": 5,
        }

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

        assert result.checks[-1] == "numeric_range"
        assert "next_cycle_prediction_probabilities_sum_invalid" in result.errors
        assert "next_cycle_prediction_cycle_must_match_probability_argmax" in result.errors

    def test_risk_reward_and_trader_equation_must_pass(self):
        decision = _trade_decision(
            order_type="突破单",
            entry=100,
            stop=99,
            tp1=100.5,
            tp2=101,
            estimated_win_rate=55,
            diagnosis=self.diagnosis,
        )

        _, result = DecisionValidator().validate(
            json.dumps(decision),
            diagnosis=self.diagnosis,
            price_action_features=self.features,
        )

        assert result.checks[-1] == "semantic_reasonableness"
        assert "risk_reward_below_minimum" in result.errors
        assert "trader_equation_fails" in result.errors

    def test_atr_expansion_veto_is_semantic(self):
        features = {**self.features, "gate_break": "up", "atr_expand_ratio": 2.1}
        raw = json.dumps(_trade_decision(diagnosis=self.diagnosis))

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
        trade_decision = _trade_decision(diagnosis=market_diagnosis)
        raw_trade_decision = json.dumps(trade_decision, ensure_ascii=False)

        saved = repository.save_analysis(
            market="crypto",
            symbol="BTC/USDT",
            timeframe="1h",
            candle_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            status="success",
            market_diagnosis_messages=[{"role": "user", "content": "market diagnosis"}],
            trade_decision_messages=[{"role": "user", "content": "trade decision"}],
            market_diagnosis=market_diagnosis,
            trade_decision=trade_decision,
            raw_responses={
                "market_diagnosis": {
                    "stage": "market_diagnosis",
                    "content": json.dumps(market_diagnosis),
                    "usage": {"prompt_tokens": 10, "total_tokens": 12},
                },
                "trade_decision": {
                    "stage": "trade_decision",
                    "content": raw_trade_decision,
                    "usage": {"prompt_tokens": 20, "total_tokens": 25},
                },
            },
            usage_total={"prompt_tokens": 30, "total_tokens": 37},
            exception={"type": "validation_error", "stage": "trade_decision"},
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
        assert row.trade_decision == trade_decision
        assert row.raw_responses == {
            "market_diagnosis": {
                "stage": "market_diagnosis",
                "content": json.dumps(market_diagnosis),
                "usage": {"prompt_tokens": 10, "total_tokens": 12},
            },
            "trade_decision": {
                "stage": "trade_decision",
                "content": raw_trade_decision,
                "usage": {"prompt_tokens": 20, "total_tokens": 25},
            },
        }
        assert row.usage_total == {"prompt_tokens": 30, "total_tokens": 37}
        assert row.exception == {"type": "validation_error", "stage": "trade_decision"}
        session.commit.assert_called_once()

    def test_previous_successful_analysis_reads_semantic_analysis_fields(self):
        market_diagnosis = {
            "cycle_position": "normal_channel",
            "direction": "bullish",
            "gate_result": "proceed",
        }
        trade_decision = _trade_decision(diagnosis=market_diagnosis)
        row = SimpleNamespace(
            candle_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            trade_decision=trade_decision,
            market_diagnosis=market_diagnosis,
            validation_status="valid",
            usage_total={"total_tokens": 123},
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
            "decision": trade_decision,
            "diagnosis": market_diagnosis,
            "validation_status": "valid",
            "usage_total": {"total_tokens": 123},
        }

    def test_previous_analysis_excludes_messages_and_raw_responses(self):
        """Even if the DB row carries bloated messages/raw_responses, the
        returned previous-analysis dict must stay slim — those fields never
        belong in the next-round prompt (regression guard for prompt bloat).
        """
        row = SimpleNamespace(
            candle_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            trade_decision={"decision": {"order_direction": "做多"}},
            market_diagnosis={"cycle_position": "spike"},
            validation_status="valid",
            usage_total={"total_tokens": 100},
            market_diagnosis_messages=[{"role": "user", "content": "x" * 100000}],
            trade_decision_messages=[{"role": "user", "content": "y" * 100000}],
            price_action_features={"rows": [{"k": "K1"}]},
            raw_responses={"market_diagnosis": {"content": "z" * 100000}},
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

        for forbidden in (
            "market_diagnosis_messages",
            "trade_decision_messages",
            "price_action_features",
            "raw_responses",
        ):
            assert forbidden not in previous, f"{forbidden} must not leak into next-round prompt"

    def test_previous_analysis_includes_decision_summary_fields(self):
        row = SimpleNamespace(
            candle_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            trade_decision={"decision": {"order_direction": "做空"}},
            market_diagnosis={"cycle_position": "trending"},
            validation_status="valid",
            usage_total={"prompt_tokens": 5, "total_tokens": 7},
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

        for required in ("candle_time", "decision", "diagnosis", "validation_status", "usage_total"):
            assert required in previous, f"{required} must be present in previous-analysis summary"


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
        assert (
            orc._should_notify(
                {"decision": {"order_type": "突破单", "order_direction": "做多"}}
            )
            is True
        )
        assert (
            orc._should_notify(
                {"decision": {"order_type": "限价单", "order_direction": "做空"}}
            )
            is True
        )
        assert (
            orc._should_notify(
                {"decision": {"order_type": "不下单", "order_direction": None}}
            )
            is False
        )
        assert orc._should_notify(None) is False
        # pa_notify_wait=True surfaces no-trade decisions too
        orc_wait = PriceActionOrchestrator(
            repository=None, llm_client=MagicMock(), config={"pa_notify_wait": True}
        )
        assert orc_wait._should_notify({"decision": {"order_type": "不下单"}}) is True
