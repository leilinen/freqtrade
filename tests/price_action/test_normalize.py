"""Tests for the normalize layer (PR1: probability + answer alias)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parents[2] / "user_data" / "strategies"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from price_action.normalize import (  # noqa: E402
    _clear_decision_to_no_order,
    _coerce_decision_no_order,
    _default_cycle_probs,
    _normalize_next_cycle_prediction,
    _normalize_order_type_aliases,
    _resolve_trace_answer,
    _resolve_trace_answers,
    _section14_violated,
    _trace_node_answer,
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


def _trade_decision_payload(order_type: str = "突破单") -> dict:
    """Build a minimal valid trade-decision payload (for coerce tests)."""
    return {
        "decision": {
            "order_direction": "做多",
            "order_type": order_type,
            "entry_price": 10.88,
            "entry_basis_bar": "K1",
            "entry_basis_extreme": "high",
            "entry_rule": "K1 高点上方 1 跳动",
            "take_profit_price": 10.94,
            "take_profit_price_2": 11.00,
            "stop_loss_price": 10.81,
            "reasoning": "方程不通过但仍写突破单",
            "diagnosis_confidence": 58,
            "diagnosis_confidence_reasoning": "t",
            "trade_confidence": 30,
            "trade_confidence_reasoning": "t",
            "estimated_win_rate": 45,
            "estimated_win_rate_reasoning": "t",
            "key_factors": [],
            "watch_points": [],
            "risk_assessment": "t",
            "invalidation_condition": "t",
        },
        "decision_trace": [],
        "terminal": {"node_id": "10.3", "outcome": "trade", "label": "方程通过"},
    }


class TestNormalizeOrderTypeAliases:
    def test_no_order_alias_mapped_to_zh(self):
        d = {"order_type": "no_order"}
        assert _normalize_order_type_aliases(d) is True
        assert d["order_type"] == "不下单"

    def test_breakout_alias_mapped(self):
        d = {"order_type": "Breakout"}
        assert _normalize_order_type_aliases(d) is True
        assert d["order_type"] == "突破单"

    def test_already_zh_noop(self):
        d = {"order_type": "限价单"}
        assert _normalize_order_type_aliases(d) is False
        assert d["order_type"] == "限价单"

    def test_empty_or_unknown_noop(self):
        assert _normalize_order_type_aliases({"order_type": ""}) is False
        assert _normalize_order_type_aliases({"order_type": "随便"}) is False


class TestTraceNodeAnswer:
    def test_returns_answer_for_matching_node(self):
        trace = [{"node_id": "10.3", "answer": "否"}]
        assert _trace_node_answer(trace, "10.3") == "否"

    def test_returns_none_when_missing(self):
        assert _trace_node_answer([], "10.3") is None
        assert _trace_node_answer(None, "10.3") is None
        assert _trace_node_answer([{"node_id": "9.0"}], "10.3") is None

    def test_trims_whitespace(self):
        trace = [{"node_id": " 10.3 ", "answer": " 是 "}]
        assert _trace_node_answer(trace, "10.3") == "是"


class TestSection14Violated:
    def test_yes_answer_is_violation(self):
        trace = [{"node_id": "14.0", "answer": "是", "reason": "方程不通过仍强行交易"}]
        assert _section14_violated(trace) is True

    def test_no_answer_is_not_violation(self):
        trace = [{"node_id": "14.0", "answer": "否", "reason": "未触犯"}]
        assert _section14_violated(trace) is False

    def test_yes_with_denial_phrase_is_not_violation(self):
        """Models that write answer=是 to mean 'scan done' must not trigger coerce."""
        trace = [{"node_id": "14.0", "answer": "是", "reason": "扫描通过，未触犯禁止行为"}]
        assert _section14_violated(trace) is False

    def test_empty_or_no_section14(self):
        assert _section14_violated([]) is False
        assert _section14_violated(None) is False


class TestClearDecisionToNoOrder:
    def test_clears_prices_and_estimated_win_rate(self):
        d = {
            "order_type": "突破单",
            "order_direction": "做多",
            "entry_price": 100.0,
            "take_profit_price": 110.0,
            "stop_loss_price": 95.0,
            "estimated_win_rate": 55,
            "estimated_win_rate_reasoning": "t",
            "trade_confidence": 30,
        }
        _clear_decision_to_no_order(d)
        assert d["order_type"] == "不下单"
        assert d["order_direction"] is None
        assert d["entry_price"] is None
        assert d["take_profit_price"] is None
        assert d["stop_loss_price"] is None
        assert d["estimated_win_rate"] is None
        assert d["estimated_win_rate_reasoning"] is None

    def test_fills_default_trade_confidence_when_missing(self):
        d = {"order_type": "突破单", "trade_confidence": None, "trade_confidence_reasoning": ""}
        _clear_decision_to_no_order(d)
        assert d["trade_confidence"] == 0
        assert d["trade_confidence_reasoning"] == "无入场计划，不存在交易信心"

    def test_preserves_existing_trade_confidence(self):
        """If the model already supplied trade_confidence, do not overwrite."""
        d = {"order_type": "突破单", "trade_confidence": 25, "trade_confidence_reasoning": "低信心"}
        _clear_decision_to_no_order(d)
        assert d["trade_confidence"] == 25
        assert d["trade_confidence_reasoning"] == "低信心"


class TestCoerceDecisionNoOrder:
    def test_10_3_no_coerces_to_no_order(self):
        """When trader_equation node says 否, force 不下单 and clear prices."""
        out = _trade_decision_payload()
        out["decision_trace"] = [
            {"node_id": "10.3", "answer": "否", "reason": "RR 0.86:1 方程不通过"}
        ]
        assert _coerce_decision_no_order(out) is True
        d = out["decision"]
        assert d["order_type"] == "不下单"
        assert d["entry_price"] is None
        assert d["take_profit_price"] is None
        assert d["stop_loss_price"] is None
        assert d["estimated_win_rate"] is None

    def test_terminal_wait_coerces_to_no_order(self):
        out = _trade_decision_payload()
        out["terminal"] = {"node_id": "10.3", "outcome": "wait", "label": "等待"}
        assert _coerce_decision_no_order(out) is True
        assert out["decision"]["order_type"] == "不下单"

    def test_terminal_reject_coerces_to_no_order(self):
        out = _trade_decision_payload()
        out["terminal"] = {"node_id": "14.0", "outcome": "reject", "label": "禁止"}
        assert _coerce_decision_no_order(out) is True
        assert out["decision"]["order_type"] == "不下单"

    def test_section14_violated_coerces_to_no_order(self):
        out = _trade_decision_payload()
        out["decision_trace"] = [
            {"node_id": "10.3", "answer": "是"},
            {"node_id": "14.0", "answer": "是", "reason": "违反反转规则"},
        ]
        assert _coerce_decision_no_order(out) is True
        assert out["decision"]["order_type"] == "不下单"

    def test_section14_yes_with_denial_phrase_does_not_coerce(self):
        out = _trade_decision_payload()
        out["decision_trace"] = [
            {"node_id": "10.3", "answer": "是"},
            {"node_id": "14.0", "answer": "是", "reason": "扫描通过，未触犯"},
        ]
        assert _coerce_decision_no_order(out) is False
        assert out["decision"]["order_type"] == "突破单"

    def test_no_triggers_leaves_trade_intact(self):
        out = _trade_decision_payload()
        out["decision_trace"] = [{"node_id": "10.3", "answer": "是"}]
        out["terminal"] = {"node_id": "10.3", "outcome": "trade"}
        assert _coerce_decision_no_order(out) is False
        assert out["decision"]["order_type"] == "突破单"
        assert out["decision"]["entry_price"] == 10.88

    def test_already_no_order_with_trade_terminal_is_noop(self):
        out = _trade_decision_payload(order_type="不下单")
        out["terminal"] = {"node_id": "10.3", "outcome": "trade"}
        assert _coerce_decision_no_order(out) is False
        assert out["decision"]["order_type"] == "不下单"

    def test_english_no_order_alias_with_wait_terminal_is_idempotent(self):
        """Regression: order_type=no_order + terminal=wait must not error."""
        out = _trade_decision_payload(order_type="no_order")
        out["terminal"] = {"node_id": "0.1", "outcome": "wait"}
        # First call should normalize alias and recognize 不下单; no further coerce
        # because order_type is already 不下单 after alias mapping — but we still
        # don't trigger since 不下单 is not in _TRADE_ORDER_TYPES and the wait/reject
        # branch only fires when current order_type != 不下单.
        assert _coerce_decision_no_order(out) is False
        assert out["decision"]["order_type"] == "不下单"

    def test_idempotent(self):
        """Running coerce twice yields the same result."""
        out = _trade_decision_payload()
        out["terminal"] = {"node_id": "14.0", "outcome": "reject"}
        first = _coerce_decision_no_order(out)
        snapshot = copy.deepcopy(out)
        second = _coerce_decision_no_order(out)
        assert first is True
        assert second is False
        assert out == snapshot

    def test_missing_decision_returns_false(self):
        assert _coerce_decision_no_order({}) is False
        assert _coerce_decision_no_order({"decision": "garbage"}) is False


class TestNormalizeTradeDecisionCoerceIntegration:
    def test_normalize_clears_trade_when_terminal_rejects(self):
        """End-to-end: normalize_trade_decision coerces to 不下单 before validator runs."""
        decision_json = _trade_decision_payload()
        out = normalize_trade_decision(decision_json)
        # No rejection signals → still a trade
        assert out["decision"]["order_type"] == "突破单"

        decision_json = _trade_decision_payload()
        decision_json["terminal"] = {"node_id": "14.0", "outcome": "reject"}
        out = normalize_trade_decision(decision_json)
        assert out["decision"]["order_type"] == "不下单"
        assert out["decision"]["entry_price"] is None

    def test_normalize_does_not_mutate_input_when_coercing(self):
        decision_json = _trade_decision_payload()
        decision_json["terminal"] = {"node_id": "14.0", "outcome": "reject"}
        original_type = decision_json["decision"]["order_type"]
        _ = normalize_trade_decision(decision_json)
        assert decision_json["decision"]["order_type"] == original_type
