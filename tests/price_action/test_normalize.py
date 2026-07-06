"""Tests for the normalize layer (PR1: probability + answer alias)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parents[2] / "user_data" / "strategies"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from price_action.normalize import (  # noqa: E402
    _default_cycle_probs,
    _normalize_next_cycle_prediction,
    _resolve_trace_answer,
    _resolve_trace_answers,
    normalize_market_diagnosis,
    normalize_trade_decision,
)
from price_action.price_tick import (  # noqa: E402
    infer_price_tick_from_rows,
    normalize_breakout_basis_extreme,
    normalize_breakout_entry_price,
    round_to_tick,
)


# ── Probability normalization ──


class TestNormalizeProbabilities:
    """Verify _normalize_next_cycle_prediction float/clamp/rescale/argmax."""

    def test_float_decimals_converted_to_int(self):
        """LLM uses 0-1 floats (id=90/96 pattern) → int 0-100."""
        pred = {
            "cycle": "trading_range",
            "probabilities": {
                "spike": 0.08,
                "micro_channel": 0.05,
                "tight_channel": 0.07,
                "normal_channel": 0.1,
                "broad_channel": 0.2,
                "trending_tr": 0.1,
                "trading_range": 0.3,
                "extreme_tr": 0.1,
            },
        }
        _normalize_next_cycle_prediction(pred)
        probs = pred["probabilities"]
        for key, val in probs.items():
            assert isinstance(val, int), f"{key} should be int, got {type(val)}"
            assert 0 <= val <= 100
        assert sum(probs.values()) == 100

    def test_missing_keys_padded_to_zero(self):
        """LLM omits low-probability cycles → filled with 0."""
        pred = {
            "cycle": "trading_range",
            "probabilities": {"trading_range": 70, "broad_channel": 30},
        }
        _normalize_next_cycle_prediction(pred)
        probs = pred["probabilities"]
        assert set(probs.keys()) == {
            "spike", "micro_channel", "tight_channel", "normal_channel",
            "broad_channel", "trending_tr", "trading_range", "extreme_tr",
        }
        assert probs["spike"] == 0
        assert probs["micro_channel"] == 0
        assert sum(probs.values()) == 100

    def test_sum_rescaled_to_100(self):
        """Probabilities summing to !=100 → rescaled."""
        pred = {
            "cycle": "spike",
            "probabilities": {
                "spike": 50, "micro_channel": 10, "tight_channel": 5,
                "normal_channel": 5, "broad_channel": 5, "trending_tr": 5,
                "trading_range": 5, "extreme_tr": 5,
            },  # sum=90
        }
        _normalize_next_cycle_prediction(pred)
        assert sum(pred["probabilities"].values()) == 100

    def test_cycle_set_to_argmax(self):
        """LLM's cycle field ≠ argmax(probabilities) → corrected."""
        pred = {
            "cycle": "spike",
            "probabilities": {
                "spike": 10, "micro_channel": 5, "tight_channel": 5,
                "normal_channel": 5, "broad_channel": 5, "trending_tr": 5,
                "trading_range": 60, "extreme_tr": 5,
            },
        }
        _normalize_next_cycle_prediction(pred)
        assert pred["cycle"] == "trading_range"

    def test_cycle_kept_when_it_is_argmax(self):
        """Already-consistent cycle is preserved."""
        pred = {
            "cycle": "trading_range",
            "probabilities": {
                "spike": 10, "micro_channel": 5, "tight_channel": 5,
                "normal_channel": 5, "broad_channel": 5, "trending_tr": 5,
                "trading_range": 60, "extreme_tr": 5,
            },
        }
        _normalize_next_cycle_prediction(pred)
        assert pred["cycle"] == "trading_range"

    def test_all_zero_falls_back_to_default(self):
        """All-zero probabilities → default distribution."""
        pred = {
            "cycle": "trading_range",
            "probabilities": {k: 0 for k in (
                "spike", "micro_channel", "tight_channel", "normal_channel",
                "broad_channel", "trending_tr", "trading_range", "extreme_tr",
            )},
        }
        _normalize_next_cycle_prediction(pred)
        assert sum(pred["probabilities"].values()) == 100
        assert pred["cycle"] == "trading_range"

    def test_unpredictable_clears_prediction(self):
        """When unpredictable=true, cycle/direction/probabilities nullified."""
        pred = {
            "unpredictable": True,
            "cycle": "spike",
            "direction": "bullish",
            "probabilities": {"spike": 100},
        }
        _normalize_next_cycle_prediction(pred)
        assert pred["unpredictable"] is True
        assert pred["cycle"] is None
        assert pred["direction"] is None
        assert pred["probabilities"] is None

    def test_no_probabilities_synthesized_from_cycle(self):
        """Missing probabilities dict → synthesized from cycle field."""
        pred = {"cycle": "broad_channel"}
        _normalize_next_cycle_prediction(pred)
        probs = pred["probabilities"]
        assert isinstance(probs, dict)
        assert sum(probs.values()) == 100
        assert probs["broad_channel"] >= 50

    def test_idempotent(self):
        """Running normalize twice yields identical output."""
        pred = {
            "cycle": "trading_range",
            "probabilities": {
                "spike": 0.3, "trading_range": 0.4, "broad_channel": 0.3,
            },
        }
        _normalize_next_cycle_prediction(pred)
        snapshot = {
            "cycle": pred["cycle"],
            "probabilities": dict(pred["probabilities"]),
        }
        _normalize_next_cycle_prediction(pred)
        assert pred["cycle"] == snapshot["cycle"]
        assert pred["probabilities"] == snapshot["probabilities"]


class TestDefaultCycleProbs:
    def test_known_cycle_centers_mass(self):
        probs = _default_cycle_probs("spike")
        assert probs["spike"] >= 50
        assert sum(probs.values()) == 100

    def test_unknown_cycle_uses_baseline(self):
        probs = _default_cycle_probs("unknown_cycle")
        assert sum(probs.values()) == 100
        assert probs["broad_channel"] == 30


# ── Trace answer alias mapping ──


class TestResolveTraceAnswer:
    def test_no_order_mapped_to_no(self):
        """LLM writes "不下单" in terminal decision node (id=98 pattern)."""
        result = _resolve_trace_answer("10.3", "不下单")
        assert result is not None
        assert result[0] == "否"

    def test_long_mapped_to_yes(self):
        result = _resolve_trace_answer("10.1", "做多")
        assert result is not None
        assert result[0] == "是"

    def test_node_2_3_bullish_mapped_with_branch(self):
        result = _resolve_trace_answer("2.3", "多头")
        assert result == ("是", "bullish")

    def test_node_2_3_bearish_mapped_with_branch(self):
        result = _resolve_trace_answer("2.3", "做空")
        assert result == ("是", "bearish")

    def test_generic_yes_mapped(self):
        result = _resolve_trace_answer("9.0", "通过")
        assert result is not None
        assert result[0] == "是"

    def test_generic_no_mapped(self):
        result = _resolve_trace_answer("9.0", "不通过")
        assert result is not None
        assert result[0] == "否"

    def test_composite_answer_with_branch(self):
        """Composite '是（多头）' on node 2.3 → ("是", "bullish")."""
        result = _resolve_trace_answer("2.3", "是（多头）")
        assert result is not None
        assert result[0] == "是"
        assert result[1] == "bullish"

    def test_partial_answer_mapped_to_neutral(self):
        result = _resolve_trace_answer("2.2", "部分符合")
        assert result is not None
        assert result[0] == "中性"

    def test_valid_canonical_answer_unchanged(self):
        """Already-canonical answers return None (no mapping needed)."""
        # "是" without composite suffix and not in any per-node alias set.
        assert _resolve_trace_answer("9.0", "是") is None

    def test_empty_answer_returns_none(self):
        assert _resolve_trace_answer("2.3", "") is None

    def test_gate_result_token_mapped(self):
        """gate_result value 'proceed' written as answer → '是'."""
        result = _resolve_trace_answer("gate_end", "proceed")
        assert result is not None
        assert result[0] == "是"


class TestResolveTraceAnswersBatch:
    def test_decision_trace_terminal_no_order_corrected(self):
        """End-to-end: full decision_trace with terminal '不下单'."""
        trace = [
            {"node_id": "9.0", "answer": "通过", "reason": "ok"},
            {"node_id": "10.1", "answer": "不下单", "reason": "no signal"},
        ]
        _resolve_trace_answers(trace)
        assert trace[0]["answer"] == "是"
        assert trace[1]["answer"] == "否"

    def test_gate_trace_node_2_3_alias_corrected(self):
        trace = [
            {"node_id": "2.3", "answer": "多头", "branch": "", "reason": "bullish"},
        ]
        _resolve_trace_answers(trace)
        assert trace[0]["answer"] == "是"
        assert trace[0]["branch"] == "bullish"

    def test_non_dict_items_skipped(self):
        trace = [None, "string", 42, {"node_id": "2.3", "answer": "做多"}]
        _resolve_trace_answers(trace)
        assert trace[3]["answer"] == "是"

    def test_empty_branch_not_injected(self):
        """Branch is only overridden when item already has a branch field."""
        trace = [{"node_id": "2.3", "answer": "多头"}]  # no branch key
        _resolve_trace_answers(trace)
        assert trace[0]["answer"] == "是"
        assert "branch" not in trace[0]


# ── Public entry points ──


class TestNormalizeTradeDecision:
    def test_probability_and_answer_fixed_together(self):
        """Both probability floats and decision_trace answer alias in one call."""
        decision = {
            "decision": {"order_type": "不下单", "order_direction": None},
            "diagnosis_summary": {"cycle_position": "trading_range", "direction": "neutral"},
            "decision_trace": [
                {"node_id": "10.1", "answer": "不下单", "reason": "no setup"},
            ],
            "next_cycle_prediction": {
                "cycle": "trading_range",
                "probabilities": {"trading_range": 0.5, "broad_channel": 0.5},
            },
        }
        result = normalize_trade_decision(decision)
        assert result["decision_trace"][0]["answer"] == "否"
        probs = result["next_cycle_prediction"]["probabilities"]
        for v in probs.values():
            assert isinstance(v, int)
        assert sum(probs.values()) == 100

    def test_does_not_mutate_input(self):
        """normalize should return a new dict, leaving input unchanged."""
        decision = {
            "decision_trace": [{"node_id": "10.1", "answer": "不下单"}],
            "next_cycle_prediction": {
                "cycle": "spike",
                "probabilities": {"spike": 0.7},
            },
        }
        original_answer = decision["decision_trace"][0]["answer"]
        original_probs = dict(decision["next_cycle_prediction"]["probabilities"])
        _ = normalize_trade_decision(decision)
        # Input is untouched (deep-copy).
        assert decision["decision_trace"][0]["answer"] == original_answer
        assert decision["next_cycle_prediction"]["probabilities"] == original_probs

    def test_missing_next_cycle_prediction_passes_through(self):
        decision = {"decision": {"order_type": "不下单"}}
        result = normalize_trade_decision(decision)
        # No exception, no prediction added.
        assert "next_cycle_prediction" not in result


class TestNormalizeMarketDiagnosis:
    def test_gate_trace_answer_alias_corrected(self):
        diag = {
            "cycle_position": "trading_range",
            "direction": "neutral",
            "gate_trace": [
                {"node_id": "2.3", "answer": "多头", "branch": "", "reason": "bull"},
            ],
        }
        result = normalize_market_diagnosis(diag)
        assert result["gate_trace"][0]["answer"] == "是"
        assert result["gate_trace"][0]["branch"] == "bullish"

    def test_does_not_mutate_input(self):
        diag = {
            "gate_trace": [{"node_id": "2.3", "answer": "多头", "branch": ""}],
        }
        original = diag["gate_trace"][0]["answer"]
        _ = normalize_market_diagnosis(diag)
        assert diag["gate_trace"][0]["answer"] == original

    def test_idempotent(self):
        diag = {
            "gate_trace": [
                {"node_id": "2.3", "answer": "多头", "branch": ""},
                {"node_id": "1.1", "answer": "通过"},
            ],
        }
        once = normalize_market_diagnosis(diag)
        twice = normalize_market_diagnosis(once)
        assert once == twice


# ── Breakout price normalization (ported from PA_Agent price_tick) ──


def _feature_rows(high: float = 104.0, low: float = 99.0) -> list[dict]:
    return [{"k": "K1", "open": 100.0, "high": high, "low": low, "close": 103.0}]


class TestBreakoutPriceNormalize:
    def test_infer_tick_from_three_decimal_prices(self):
        rows = [{"k": "K1", "open": 0.0, "high": 4556.595, "low": 99.0, "close": 0.0}]
        assert infer_price_tick_from_rows(rows) == 0.001

    def test_infer_tick_returns_none_for_empty_rows(self):
        assert infer_price_tick_from_rows(None) is None
        assert infer_price_tick_from_rows([]) is None

    def test_infer_tick_returns_one_for_integer_prices(self):
        rows = [{"k": "K1", "open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0}]
        assert infer_price_tick_from_rows(rows) == 1.0

    def test_round_to_tick_basic(self):
        assert round_to_tick(4556.5951, 0.001) == 4556.595
        # Python's round uses banker's rounding (round-half-to-even), so
        # 100.5 -> 100, not 101. This is shared behavior with upstream.
        assert round_to_tick(100.5, 1.0) == 100.0
        assert round_to_tick(100.4, 1.0) == 100.0
        assert round_to_tick(100.6, 1.0) == 101.0

    def test_round_to_tick_zero_tick_is_noop(self):
        assert round_to_tick(123.456, 0.0) == 123.456

    def test_normalize_breakout_entry_at_high_bumps_up(self):
        """entry == K1.high → pushed to high + tick (mirrors upstream behavior)."""
        rows = _feature_rows(high=4556.595, low=4500.0)
        decision = {
            "order_type": "突破单",
            "order_direction": "做多",
            "entry_basis_bar": "K1",
            "entry_basis_extreme": "high",
            "entry_price": 4556.595,
        }
        assert normalize_breakout_entry_price(decision, feature_rows=rows) is True
        assert decision["entry_price"] == round_to_tick(4556.595 + 0.001, 0.001)

    def test_normalize_breakout_entry_at_low_pushes_down(self):
        """entry == K1.low → pushed to low - tick."""
        rows = _feature_rows(high=100.0, low=95.0)
        decision = {
            "order_type": "突破单",
            "order_direction": "做空",
            "entry_basis_bar": "K1",
            "entry_basis_extreme": "low",
            "entry_price": 95.0,
        }
        assert normalize_breakout_entry_price(decision, feature_rows=rows) is True
        assert decision["entry_price"] == 94.0  # tick=1.0 inferred from integer prices

    def test_normalize_short_breakout_extreme_high_to_low(self):
        decision = {
            "order_type": "突破单",
            "order_direction": "做空",
            "entry_basis_extreme": "high",
            "entry_basis_bar": "K3",
            "entry_price": 3.42,
        }
        assert normalize_breakout_basis_extreme(decision)
        assert decision["entry_basis_extreme"] == "low"

    def test_normalize_breakout_skips_non_breakout(self):
        decision = {
            "order_type": "限价单",
            "order_direction": "做多",
            "entry_basis_bar": "K1",
            "entry_basis_extreme": "high",
            "entry_price": 100.0,
        }
        assert normalize_breakout_entry_price(decision, feature_rows=_feature_rows()) is False
        assert normalize_breakout_basis_extreme(decision) is False
        assert decision["entry_price"] == 100.0

    def test_normalize_breakout_skips_missing_basis_bar(self):
        decision = {
            "order_type": "突破单",
            "order_direction": "做多",
            "entry_basis_extreme": "high",
            "entry_price": 100.0,
        }
        assert normalize_breakout_entry_price(decision, feature_rows=_feature_rows()) is False

    def test_normalize_breakout_skips_missing_feature_rows(self):
        decision = {
            "order_type": "突破单",
            "order_direction": "做多",
            "entry_basis_bar": "K1",
            "entry_basis_extreme": "high",
            "entry_price": 100.0,
        }
        assert normalize_breakout_entry_price(decision, feature_rows=None) is False
        assert decision["entry_price"] == 100.0

    def test_normalize_breakout_skips_when_basis_row_absent(self):
        decision = {
            "order_type": "突破单",
            "order_direction": "做多",
            "entry_basis_bar": "K5",
            "entry_basis_extreme": "high",
            "entry_price": 100.0,
        }
        assert normalize_breakout_entry_price(decision, feature_rows=_feature_rows()) is False

    def test_normalize_breakout_idempotent(self):
        rows = _feature_rows(high=4556.595, low=4500.0)
        decision = {
            "order_type": "突破单",
            "order_direction": "做多",
            "entry_basis_bar": "K1",
            "entry_basis_extreme": "high",
            "entry_price": 4556.595,
        }
        first = normalize_breakout_entry_price(decision, feature_rows=rows)
        second = normalize_breakout_entry_price(decision, feature_rows=rows)
        assert first is True
        assert second is False  # already at target, no further change

    def test_normalize_trade_decision_snaps_breakout_entry(self):
        """End-to-end: normalize_trade_decision pushes entry_price to high + tick."""
        decision_json = {
            "decision": {
                "order_type": "突破单",
                "order_direction": "做多",
                "entry_basis_bar": "K1",
                "entry_basis_extreme": "high",
                "entry_price": 4556.595,
                "stop_loss_price": 99.0,
                "take_profit_price": 4600.0,
                "take_profit_price_2": 4700.0,
            },
        }
        out = normalize_trade_decision(
            decision_json,
            feature_rows=[{"k": "K1", "open": 100.0, "high": 4556.595, "low": 99.0, "close": 103.0}],
        )
        assert out["decision"]["entry_price"] == round_to_tick(4556.596, 0.001)

    def test_normalize_trade_decision_does_not_mutate_input(self):
        decision_json = {
            "decision": {
                "order_type": "突破单",
                "order_direction": "做多",
                "entry_basis_bar": "K1",
                "entry_basis_extreme": "high",
                "entry_price": 4556.595,
            },
        }
        original_entry = decision_json["decision"]["entry_price"]
        _ = normalize_trade_decision(
            decision_json,
            feature_rows=[{"k": "K1", "open": 100.0, "high": 4556.595, "low": 99.0, "close": 103.0}],
        )
        assert decision_json["decision"]["entry_price"] == original_entry
