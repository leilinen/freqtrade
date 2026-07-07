"""Tests for decision_tree loader and canonical question repair."""
from __future__ import annotations

import sys
from pathlib import Path

_STRATEGY_DIR = Path(__file__).resolve().parents[2] / "user_data" / "strategies"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

import pytest  # noqa: E402

from price_action.decision_tree import (  # noqa: E402
    canonical_tree_questions,
    load_decision_tree,
)
from price_action.trace_normalize import (  # noqa: E402
    repair_stage1_gate_trace_questions,
    repair_stage2_decision_trace_questions,
)


class TestLoadDecisionTree:
    def test_returns_dict(self):
        tree = load_decision_tree()
        assert isinstance(tree, dict)
        assert "sections" in tree
        assert "node_index" in tree

    def test_has_sections(self):
        tree = load_decision_tree()
        assert len(tree["sections"]) > 0

    def test_node_index_populated(self):
        tree = load_decision_tree()
        index = tree["node_index"]
        assert len(index) > 0
        # Spot-check known node_ids
        assert "1.1" in index
        assert "10.3" in index

    def test_node_has_question(self):
        tree = load_decision_tree()
        node = tree["node_index"]["1.1"]
        assert isinstance(node["question"], str)
        assert node["question"].strip()

    def test_cached(self):
        a = load_decision_tree()
        b = load_decision_tree()
        assert a is b  # lru_cache

    def test_source_field(self):
        tree = load_decision_tree()
        assert tree["source"]


class TestCanonicalTreeQuestions:
    def test_returns_dict(self):
        qs = canonical_tree_questions()
        assert isinstance(qs, dict)
        assert len(qs) > 0

    def test_spot_check_1_1(self):
        qs = canonical_tree_questions()
        assert "1.1" in qs
        assert "数据是否足够" in qs["1.1"]

    def test_all_values_non_empty_strings(self):
        qs = canonical_tree_questions()
        for nid, q in qs.items():
            assert isinstance(nid, str)
            assert isinstance(q, str)
            assert q.strip()


class TestRepairStage1GateTraceQuestions:
    def test_rewrites_paraphrased_question(self):
        gate = [{"node_id": "1.1", "question": "K线数据是否充足？", "answer": "是"}]
        assert repair_stage1_gate_trace_questions(gate) is True
        assert gate[0]["question"] == canonical_tree_questions()["1.1"]

    def test_no_change_when_already_canonical(self):
        canonical = canonical_tree_questions()["1.1"]
        gate = [{"node_id": "1.1", "question": canonical, "answer": "是"}]
        assert repair_stage1_gate_trace_questions(gate) is False

    def test_unknown_node_left_untouched(self):
        gate = [{"node_id": "99.9", "question": "随机问题？", "answer": "是"}]
        assert repair_stage1_gate_trace_questions(gate) is False
        assert gate[0]["question"] == "随机问题？"

    def test_non_dict_items_skipped(self):
        gate = [None, "string", {"node_id": "1.1", "question": "x"}]
        assert repair_stage1_gate_trace_questions(gate) is True
        assert gate[2]["question"] == canonical_tree_questions()["1.1"]

    def test_empty_list_no_op(self):
        assert repair_stage1_gate_trace_questions([]) is False


class TestRepairStage2DecisionTraceQuestions:
    def test_rewrites_paraphrased_question(self):
        trace = [{"node_id": "10.3", "question": "随便写的", "answer": "否"}]
        assert repair_stage2_decision_trace_questions(trace) is True
        assert trace[0]["question"] == canonical_tree_questions()["10.3"]

    def test_no_change_when_already_canonical(self):
        canonical = canonical_tree_questions()["10.3"]
        trace = [{"node_id": "10.3", "question": canonical, "answer": "否"}]
        assert repair_stage2_decision_trace_questions(trace) is False

    def test_unknown_node_left_untouched(self):
        trace = [{"node_id": "abc", "question": "x"}]
        assert repair_stage2_decision_trace_questions(trace) is False

    def test_non_dict_items_skipped(self):
        trace = [42, {"node_id": "10.3", "question": "x"}]
        assert repair_stage2_decision_trace_questions(trace) is True

    def test_empty_list_no_op(self):
        assert repair_stage2_decision_trace_questions([]) is False
