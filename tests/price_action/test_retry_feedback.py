"""Tests for the structured retry-feedback builder (P0-2).

Verifies the ported ``_build_retry_feedback`` / ``_append_retry_feedback``
produce actionable, human-readable hints instead of bare internal error codes,
and that the previous assistant turn is re-injected for in-context correction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parents[2] / "user_data" / "strategies"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from price_action.orchestrator import (  # noqa: E402
    _append_retry_feedback,
    _build_retry_feedback,
    _categorize_errors,
    _lookup_hint,
)


# ── _build_retry_feedback content ──


class TestBuildRetryFeedback:
    def test_gate_trace_direction_branch_conflict_translated(self):
        """The id=232 case: bare error code must become actionable Chinese."""
        fb = _build_retry_feedback(
            stage="market_diagnosis",
            errors=["market_diagnosis_gate_trace_direction_branch_conflict"],
            attempt=1,
            retry_limit=1,
        )
        # Category label
        assert "字段值/一致性不符合规则" in fb
        # Human-readable explanation, not just the code
        assert "direction" in fb and "branch" in fb
        assert "long/short/neutral" in fb
        # Attempt counter
        assert "1/1" in fb
        # Must NOT be just the raw code dump
        assert fb.count("market_diagnosis_gate_trace") == 1  # only in the [code] tag

    def test_bar_by_bar_summary_translated(self):
        fb = _build_retry_feedback(
            stage="market_diagnosis",
            errors=["market_diagnosis_bar_by_bar_summary_must_cover_k5_to_k1"],
            attempt=2,
            retry_limit=3,
        )
        assert "K5" in fb and "K1" in fb
        assert "缺一不可" in fb
        assert "2/3" in fb

    def test_enum_hint_appended_for_gate_trace_answer(self):
        """When errors mention gate_trace + answer, an enum block is appended."""
        fb = _build_retry_feedback(
            stage="market_diagnosis",
            errors=["market_diagnosis_gate_trace_answer_invalid"],
            attempt=1,
            retry_limit=2,
        )
        assert "是/否/中性/等待/不适用" in fb

    def test_enum_hint_appended_for_bar_by_bar_role(self):
        fb = _build_retry_feedback(
            stage="market_diagnosis",
            errors=[
                "market_diagnosis_bar_by_bar_role_invalid",
                "market_diagnosis_bar_by_bar_trapped_side_invalid",
            ],
            attempt=1,
            retry_limit=2,
        )
        assert "structure/signal/entry" in fb
        assert "bulls/bears/both/none" in fb

    def test_trade_decision_stage_uses_correct_label(self):
        fb = _build_retry_feedback(
            stage="trade_decision",
            errors=["risk_reward_below_minimum"],
            attempt=1,
            retry_limit=2,
        )
        assert "交易决策(stage2)" in fb
        assert "盈亏比" in fb

    def test_forbidden_section_present_for_diagnosis(self):
        fb = _build_retry_feedback(
            stage="market_diagnosis",
            errors=["market_diagnosis_direction_invalid"],
            attempt=1,
            retry_limit=2,
        )
        assert "禁止为通过校验而修改" in fb
        assert "顶层 direction" in fb

    def test_forbidden_section_present_for_decision(self):
        fb = _build_retry_feedback(
            stage="trade_decision",
            errors=["risk_reward_below_minimum"],
            attempt=1,
            retry_limit=2,
        )
        assert "禁止为通过校验而修改" in fb
        assert "不下单" in fb

    def test_too_many_errors_truncated(self):
        """When >8 errors, the rest are summarized not listed."""
        errors = [f"market_diagnosis_missing_field_{i}" for i in range(12)]
        fb = _build_retry_feedback(
            stage="market_diagnosis",
            errors=errors,
            attempt=1,
            retry_limit=2,
        )
        assert "另有 4 条错误" in fb


# ── _lookup_hint edge cases ──


class TestLookupHint:
    def test_known_code_returns_hint(self):
        hint = _lookup_hint("market_diagnosis_direction_invalid")
        assert "long/short/neutral" in hint

    def test_missing_field_pattern(self):
        hint = _lookup_hint("market_diagnosis_missing_some_field")
        assert "缺少必填字段" in hint and "some_field" in hint

    def test_json_decode_error(self):
        hint = _lookup_hint("invalid_json:Expecting ':' delimiter")
        assert "JSON 语法错误" in hint

    def test_unknown_code_falls_back_to_raw(self):
        hint = _lookup_hint("some_unknown_error_code_xyz")
        assert "some_unknown_error_code_xyz" in hint


# ── _append_retry_feedback message assembly ──


class TestAppendRetryFeedback:
    def test_previous_assistant_turn_re_injected(self):
        """The previous LLM output must be re-added so it can self-correct."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        result = _append_retry_feedback(
            msgs,
            stage="market_diagnosis",
            errors=["market_diagnosis_direction_invalid"],
            attempt=2,
            retry_limit=3,
            previous_raw='{"direction": "neutral"}',
        )
        roles = [m["role"] for m in result]
        # assistant turn appears before the new user feedback
        assert roles == ["system", "user", "assistant", "user"]
        assert result[-2]["role"] == "assistant"
        assert result[-2]["content"] == '{"direction": "neutral"}'
        assert result[-1]["role"] == "user"
        assert "校验未通过" in result[-1]["content"]

    def test_no_previous_raw_omits_assistant_turn(self):
        msgs = [{"role": "user", "content": "q"}]
        result = _append_retry_feedback(
            msgs,
            stage="market_diagnosis",
            errors=["market_diagnosis_direction_invalid"],
            previous_raw=None,
        )
        roles = [m["role"] for m in result]
        assert roles == ["user", "user"]

    def test_empty_previous_raw_omits_assistant_turn(self):
        msgs = [{"role": "user", "content": "q"}]
        result = _append_retry_feedback(
            msgs,
            stage="market_diagnosis",
            errors=["market_diagnosis_direction_invalid"],
            previous_raw="   ",
        )
        roles = [m["role"] for m in result]
        assert roles == ["user", "user"]

    def test_original_messages_not_mutated(self):
        msgs = [{"role": "user", "content": "q"}]
        original_len = len(msgs)
        _append_retry_feedback(
            msgs,
            stage="market_diagnosis",
            errors=["market_diagnosis_direction_invalid"],
            previous_raw="x",
        )
        assert len(msgs) == original_len
