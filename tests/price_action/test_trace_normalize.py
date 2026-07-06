"""Tests for trace_normalize (Batch A: bar_range repair).

Mirrors upstream ``tests/unit/test_trace_normalize.py`` for the bar_range
subset. Verifies canonicalization, placeholder inference, reason-citation
expansion, prior-bar_range fallback, and integration with the higher-level
normalize_market_diagnosis / normalize_trade_decision entry points.
"""
from __future__ import annotations

import sys
from pathlib import Path

_STRATEGY_DIR = Path(__file__).resolve().parents[2] / "user_data" / "strategies"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

import pytest  # noqa: E402

from price_action.normalize import (  # noqa: E402
    normalize_market_diagnosis,
    normalize_trade_decision,
)
from price_action.trace_normalize import (  # noqa: E402
    _bar_range_from_reason,
    _bar_range_is_canonical,
    _bar_seqs_from_range_text,
    _bar_seqs_from_reason_text,
    _comma_separated_bar_range,
    _expand_bar_range_for_reason_citations,
    _is_nullish,
    ensure_trace_string_fields,
    fix_bar_range_string,
    infer_max_bar_seq_from_trace,
    normalize_trace_item_bar_range,
    normalize_trace_list_bar_range,
)


class TestIsNullish:
    def test_none(self):
        assert _is_nullish(None) is True

    def test_empty_string(self):
        assert _is_nullish("") is True

    def test_whitespace(self):
        assert _is_nullish("   ") is True

    def test_null_string(self):
        assert _is_nullish("null") is True
        assert _is_nullish("NULL") is True

    def test_normal_string(self):
        assert _is_nullish("K1") is False


class TestEnsureTraceStringFields:
    def test_fills_missing_keys(self):
        item = {}
        ensure_trace_string_fields(item)
        assert item["node_id"] == ""
        assert item["question"] == "—"
        assert item["answer"] == "否"
        assert item["reason"] == "—"

    def test_null_values_filled(self):
        item = {"node_id": None, "question": None, "answer": None, "reason": None}
        ensure_trace_string_fields(item)
        assert item["node_id"] == ""
        assert item["question"] == "—"
        assert item["answer"] == "否"
        assert item["reason"] == "—"

    def test_skipped_node_gets_bu_shi_yong(self):
        item = {"answer": None, "skipped": True}
        ensure_trace_string_fields(item)
        assert item["answer"] == "不适用"

    def test_existing_values_preserved(self):
        item = {
            "node_id": "10.3",
            "question": "方程是否通过？",
            "answer": "是",
            "reason": "方程通过",
        }
        ensure_trace_string_fields(item)
        assert item == {
            "node_id": "10.3",
            "question": "方程是否通过？",
            "answer": "是",
            "reason": "方程通过",
        }


class TestInferMaxBarSeq:
    def test_from_bar_range(self):
        trace = [{"bar_range": "K5-K1"}, {"bar_range": "K3-K2"}]
        assert infer_max_bar_seq_from_trace(trace) == 5

    def test_from_reason(self):
        trace = [{"reason": "K7 收盘强于 K3"}]
        assert infer_max_bar_seq_from_trace(trace) == 7

    def test_returns_none_for_empty(self):
        assert infer_max_bar_seq_from_trace([]) is None
        assert infer_max_bar_seq_from_trace([{}]) is None

    def test_ignores_non_dict(self):
        assert infer_max_bar_seq_from_trace([None, "garbage"]) is None


class TestCommaSeparatedBarRange:
    def test_chinese_comma(self):
        assert _comma_separated_bar_range("K7，K1") == "K7-K1"

    def test_ideographic_comma(self):
        assert _comma_separated_bar_range("K1、K7") == "K7-K1"

    def test_ascii_comma(self):
        assert _comma_separated_bar_range("K3,K5") == "K5-K3"

    def test_single_returns_none(self):
        assert _comma_separated_bar_range("K3,garbage") is None

    def test_no_comma_returns_none(self):
        assert _comma_separated_bar_range("K3-K1") is None


class TestBarRangeIsCanonical:
    @pytest.mark.parametrize(
        "text",
        ["K1", "K8-K1", "K1-K8", "不适用", "—", "-", ""],
    )
    def test_canonical(self, text):
        assert _bar_range_is_canonical(text) is True

    @pytest.mark.parametrize(
        "text",
        ["Kx", "global", "all", "pending", "K1-K2-K3", "garbage"],
    )
    def test_not_canonical(self, text):
        assert _bar_range_is_canonical(text) is False


class TestBarSeqsFromRangeText:
    def test_single(self):
        assert _bar_seqs_from_range_text("K3") == {3}

    def test_range(self):
        assert _bar_seqs_from_range_text("K5-K1") == {1, 2, 3, 4, 5}

    def test_reversed_range(self):
        assert _bar_seqs_from_range_text("K1-K5") == {1, 2, 3, 4, 5}

    def test_bu_shi_yong(self):
        assert _bar_seqs_from_range_text("不适用") == set()

    def test_global(self):
        assert _bar_seqs_from_range_text("全局") == set()


class TestBarSeqsFromReasonText:
    def test_extracts_all_k_refs(self):
        assert _bar_seqs_from_reason_text("K1 强于 K3 和 K7") == {1, 3, 7}

    def test_handles_space_between(self):
        assert _bar_seqs_from_reason_text("K 1 / K  2") == {1, 2}

    def test_empty(self):
        assert _bar_seqs_from_reason_text("") == set()
        assert _bar_seqs_from_reason_text("无 K 引用") == set()


class TestBarRangeFromReason:
    def test_infers_from_reason_citations(self):
        item = {"reason": "K7 突破 K3 高点"}
        assert _bar_range_from_reason(item) == "K7-K3"

    def test_uses_default_max_seq_to_clip(self):
        item = {"reason": "K1 K2 K9"}
        assert _bar_range_from_reason(item, default_max_seq=8) == "K2-K1"

    def test_returns_none_when_no_citations(self):
        item = {"reason": "无 K 引用"}
        assert _bar_range_from_reason(item) is None

    def test_single_bar(self):
        item = {"reason": "K3 是信号 bar"}
        assert _bar_range_from_reason(item) == "K3"


class TestFixBarRangeString:
    def test_reversed_range_auto_fixed(self):
        assert fix_bar_range_string("K1-K8") == "K8-K1"

    def test_already_canonical(self):
        assert fix_bar_range_string("K8-K1") == "K8-K1"

    def test_single_bar(self):
        assert fix_bar_range_string("K3") == "K3"

    def test_chinese_alias_with_default(self):
        assert fix_bar_range_string("全局", default_max_seq=8) == "K8-K1"

    def test_chinese_alias_without_default(self):
        assert fix_bar_range_string("全局") == "不适用"

    def test_english_alias_all(self):
        assert fix_bar_range_string("all", default_max_seq=5) == "K5-K1"

    def test_pending_placeholder_returns_empty(self):
        assert fix_bar_range_string("pending") == ""
        assert fix_bar_range_string("等待") == ""

    def test_comma_shorthand(self):
        assert fix_bar_range_string("K1, K5") == "K5-K1"

    def test_cap_with_default(self):
        assert fix_bar_range_string("K10-K1", default_max_seq=8) == "K8-K1"

    def test_unknown_returns_raw(self):
        assert fix_bar_range_string("garbage") == "garbage"

    def test_cap_merges_to_single_when_max_collides(self):
        # K3 with default_max_seq=2 → cap to K2
        assert fix_bar_range_string("K3", default_max_seq=2) == "K2"


class TestExpandBarRangeForReasonCitations:
    def test_widens_when_reason_cites_outside_bar_range(self):
        item = {"node_id": "9.0", "bar_range": "K3-K1", "reason": "K5 提供背景"}
        _expand_bar_range_for_reason_citations(item)
        assert item["bar_range"] == "K5-K1"

    def test_no_change_when_cited_within_range(self):
        item = {"node_id": "9.0", "bar_range": "K5-K1", "reason": "K3 信号"}
        _expand_bar_range_for_reason_citations(item)
        assert item["bar_range"] == "K5-K1"

    def test_no_change_for_bu_shi_yong(self):
        item = {"node_id": "9.0", "bar_range": "不适用", "reason": "K3 信号"}
        _expand_bar_range_for_reason_citations(item)
        assert item["bar_range"] == "不适用"

    def test_clips_to_default_when_known(self):
        """Reason citations are always kept even outside frame; non-cited parts clip."""
        item = {"node_id": "9.0", "bar_range": "K3-K1", "reason": "K9 引用"}
        _expand_bar_range_for_reason_citations(item, default_max_seq=5)
        # Cited K9 is preserved; non-cited K3,K1 clipped (but K3,K1 are ≤5 so kept)
        # merged = {1,2,3} | {9} (after clip applied to non-cited only)
        assert item["bar_range"] == "K9-K1"


class TestNormalizeTraceItemBarRange:
    def test_fixes_reversed_range(self):
        item = {"node_id": "1.1", "answer": "是", "reason": "K 线足够", "bar_range": "K1-K5"}
        normalize_trace_item_bar_range(item)
        assert item["bar_range"] == "K5-K1"

    def test_infers_null_bar_range_from_reason(self):
        item = {"node_id": "9.0", "answer": "是", "reason": "K3 与 K5 形态确认", "bar_range": None}
        normalize_trace_item_bar_range(item)
        assert item["bar_range"] == "K5-K3"

    def test_infers_pending_bar_range_from_reason(self):
        item = {"node_id": "9.0", "answer": "是", "reason": "K2 K4", "bar_range": "pending"}
        normalize_trace_item_bar_range(item)
        assert item["bar_range"] == "K4-K2"

    def test_skipped_node_gets_bu_shi_yong(self):
        item = {"node_id": "9.0", "answer": "", "reason": "—", "bar_range": None, "skipped": True}
        normalize_trace_item_bar_range(item)
        assert item["bar_range"] == "不适用"
        assert item["answer"] == "不适用"

    def test_falls_back_to_default_max_seq(self):
        item = {"node_id": "9.0", "answer": "是", "reason": "无引用", "bar_range": None}
        normalize_trace_item_bar_range(item, default_max_seq=8)
        assert item["bar_range"] == "K8-K1"

    def test_falls_back_to_bu_shi_yong_when_no_context(self):
        item = {"node_id": "9.0", "answer": "是", "reason": "无引用", "bar_range": None}
        normalize_trace_item_bar_range(item)
        assert item["bar_range"] == "不适用"


class TestNormalizeTraceListBarRange:
    def test_walks_list_with_prior_carrier(self):
        trace = [
            {"node_id": "1.1", "answer": "是", "reason": "K5 数据", "bar_range": "K5-K1"},
            {"node_id": "1.2", "answer": "是", "reason": "通道", "bar_range": None},
        ]
        normalize_trace_list_bar_range(trace, default_max_seq=5)
        # 1.2 falls back to default_max_seq (K5-K1) since reason has no K ref
        assert trace[1]["bar_range"] == "K5-K1"

    def test_default_max_seq_inferred_from_trace(self):
        trace = [
            {"node_id": "1.1", "answer": "是", "reason": "K8 数据", "bar_range": "K8-K1"},
            {"node_id": "1.2", "answer": "是", "reason": "无引用", "bar_range": None},
        ]
        normalize_trace_list_bar_range(trace)
        # max_seq inferred as 8 from K8 mention
        assert trace[1]["bar_range"] == "K8-K1"

    def test_none_trace_is_noop(self):
        assert normalize_trace_list_bar_range(None) is None

    def test_non_dict_items_skipped(self):
        trace = [None, "garbage", {"node_id": "1.1", "answer": "是", "reason": "K1", "bar_range": "K1"}]
        # Should not raise
        normalize_trace_list_bar_range(trace)
        assert trace[2]["bar_range"] == "K1"


class TestNormalizeMarketDiagnosisIntegration:
    def test_canonicalizes_gate_trace_bar_range(self):
        diagnosis = {
            "gate_trace": [
                {"node_id": "1.1", "answer": "是", "reason": "K 线足够", "bar_range": "K1-K5"},
            ],
        }
        out = normalize_market_diagnosis(diagnosis, feature_rows=[{"k": f"K{i}"} for i in range(1, 6)])
        assert out["gate_trace"][0]["bar_range"] == "K5-K1"
        # Original not mutated
        assert diagnosis["gate_trace"][0]["bar_range"] == "K1-K5"

    def test_infers_null_bar_range_from_reason_for_gate_trace(self):
        diagnosis = {
            "gate_trace": [
                {"node_id": "1.1", "answer": "是", "reason": "K3 与 K5 形态", "bar_range": None},
            ],
        }
        out = normalize_market_diagnosis(diagnosis)
        assert out["gate_trace"][0]["bar_range"] == "K5-K3"


class TestNormalizeTradeDecisionIntegration:
    def test_canonicalizes_decision_trace_bar_range(self):
        decision = {
            "decision_trace": [
                {"node_id": "9.0", "answer": "是", "reason": "K 线突破", "bar_range": "K1-K3"},
            ],
            "terminal": {"node_id": "10.3", "outcome": "trade"},
        }
        out = normalize_trade_decision(decision, feature_rows=[{"k": f"K{i}"} for i in range(1, 4)])
        assert out["decision_trace"][0]["bar_range"] == "K3-K1"

    def test_decision_trace_infers_bar_range_from_reason(self):
        decision = {
            "decision_trace": [
                {"node_id": "9.0", "answer": "是", "reason": "K2 K4 形态确认", "bar_range": None},
            ],
            "terminal": {"node_id": "10.3", "outcome": "trade"},
        }
        out = normalize_trade_decision(decision)
        assert out["decision_trace"][0]["bar_range"] == "K4-K2"

    def test_does_not_mutate_input(self):
        decision = {
            "decision_trace": [
                {"node_id": "9.0", "answer": "是", "reason": "K 线", "bar_range": "K1-K3"},
            ],
            "terminal": {"node_id": "10.3", "outcome": "trade"},
        }
        original = decision["decision_trace"][0]["bar_range"]
        _ = normalize_trade_decision(decision)
        assert decision["decision_trace"][0]["bar_range"] == original
