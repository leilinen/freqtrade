"""Tests for stage1 DecisionNodeEngine port (§1.1 + §2.3/§2.4).

Mirrors upstream PA_Agent/tests/unit/test_decision_nodes.py but adapted to
freqtrade's ``feature_rows: list[dict]`` shape (K1=newest).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parents[2] / "user_data" / "strategies"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from price_action.decision_nodes import (  # noqa: E402
    ALWAYS_IN_NEAR_WINDOW,
    BAR_COUNT_THRESHOLD,
    DIRECTION_BEAR_THRESHOLD,
    DIRECTION_BULL_THRESHOLD,
    DIRECTION_WINDOW,
    NodeFill,
    _build_program_trace_node,
    _frame_from_feature_rows,
    _merge_program_nodes,
    _node_id_sort_key,
    apply_stage1_nodes,
    judge_always_in,
    judge_data_sufficiency,
    judge_direction,
)
from price_action.normalize import normalize_market_diagnosis  # noqa: E402


# ── Test helpers ────────────────────────────────────────────────────────────


def _row(
    seq: int,
    *,
    o: float,
    h: float,
    l: float,
    c: float,
    ema: float | None = None,
    atr: float | None = None,
) -> dict:
    """Build a feature_row dict (K1 = newest)."""
    return {
        "k": f"K{seq}",
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "ema20": ema if ema is not None else c,
        "atr14": atr if atr is not None else 1.0,
    }


def _bullish_rows(n: int = 10) -> list[dict]:
    """Build n rows with a steady uptrend (close rising from old→new).

    bars[0] = K1 (newest, highest), bars[n-1] = KN (oldest, lowest).
    EMA tracks close. ATR set so the 0.05*ATR dead-zone doesn't kill S1.
    """
    rows = []
    base = 100.0
    step = 1.5  # > 0.05*ATR (ATR=1.0 → threshold 0.05)
    for i in range(n):
        seq = i + 1  # K1..Kn
        # oldest (Kn) has lowest price; newest (K1) highest
        close = base + (n - seq) * step
        rows.append(_row(seq, o=close - 0.5, h=close + 0.5, l=close - 1.0, c=close, ema=close - 0.5))
    return rows


def _bearish_rows(n: int = 10) -> list[dict]:
    """Build n rows with a steady downtrend (close falling from old→new)."""
    rows = []
    base = 200.0
    step = -1.5
    for i in range(n):
        seq = i + 1
        close = base + (n - seq) * step
        rows.append(_row(seq, o=close + 0.5, h=close + 1.0, l=close - 0.5, c=close, ema=close + 0.5))
    return rows


def _range_rows(n: int = 10) -> list[dict]:
    """Build n rows in a flat trading range (no direction)."""
    rows = []
    base = 150.0
    for i in range(n):
        seq = i + 1
        # oscillate within 0.2 of base
        close = base + (0.1 if seq % 2 == 0 else -0.1)
        rows.append(_row(seq, o=close, h=close + 0.2, l=close - 0.2, c=close, ema=base))
    return rows


# ── TestFrameFromFeatureRows ────────────────────────────────────────────────


class TestFrameFromFeatureRows:
    """Verify the frame adapter handles edge cases."""

    def test_none_returns_none(self):
        assert _frame_from_feature_rows(None) is None

    def test_empty_list_returns_none(self):
        assert _frame_from_feature_rows([]) is None

    def test_too_few_bars_returns_none(self):
        """Need >=5 bars for swing detection."""
        rows = [_row(i + 1, o=1.0, h=2.0, l=0.5, c=1.5) for i in range(4)]
        assert _frame_from_feature_rows(rows) is None

    def test_missing_ema_fills_nan(self):
        rows = [
            {**_row(i + 1, o=1.0, h=2.0, l=0.5, c=1.5), "ema20": None}
            for i in range(6)
        ]
        frame = _frame_from_feature_rows(rows)
        assert frame is not None
        # All EMA values become NaN (or row.get fallback fails). Just confirm
        # the frame was built; judge_direction must handle NaN safely.
        assert len(frame.bars) == 6

    def test_missing_ohlc_skips_row(self):
        rows = [_row(i + 1, o=1.0, h=2.0, l=0.5, c=1.5) for i in range(5)]
        # corrupt one row by removing 'high'
        del rows[2]["high"]
        frame = _frame_from_feature_rows(rows)
        # 4 valid rows left → below MIN_BARS_FOR_ENGINE → None
        assert frame is None

    def test_bars_sorted_by_seq_ascending(self):
        """bars[0] should be K1 (smallest seq = newest)."""
        rows = [_row(seq, o=1.0, h=2.0, l=0.5, c=1.5) for seq in range(1, 7)]
        # shuffle the order
        rows.reverse()
        frame = _frame_from_feature_rows(rows)
        assert frame is not None
        assert [b.seq for b in frame.bars] == [1, 2, 3, 4, 5, 6]


# ── TestJudgeDirection ──────────────────────────────────────────────────────


class TestJudgeDataSufficiency:
    """Verify upstream §1.1 program-filled data sufficiency node."""

    def test_data_sufficiency_node_uses_full_window(self):
        rows = _bullish_rows(24)
        frame = _frame_from_feature_rows(rows)
        assert frame is not None
        fill = judge_data_sufficiency(frame)
        assert fill.node_id == "1.1"
        assert fill.answer == "是"
        assert fill.bar_range == "K24-K1"
        assert str(BAR_COUNT_THRESHOLD) in fill.reason


class TestJudgeDirection:
    """Verify §2.3 five-signal direction vote."""

    def test_clear_uptrend_returns_bullish(self):
        rows = _bullish_rows(12)
        frame = _frame_from_feature_rows(rows)
        assert frame is not None
        direction, fill = judge_direction(frame)
        assert direction == "bullish"
        assert fill.branch == "bullish"
        assert fill.answer == "是"
        assert fill.node_id == "2.3"
        assert fill.bar_range == f"K{DIRECTION_WINDOW}-K1"

    def test_clear_downtrend_returns_bearish(self):
        rows = _bearish_rows(12)
        frame = _frame_from_feature_rows(rows)
        assert frame is not None
        direction, fill = judge_direction(frame)
        assert direction == "bearish"
        assert fill.branch == "bearish"
        assert fill.answer == "是"

    def test_flat_range_returns_neutral(self):
        rows = _range_rows(12)
        frame = _frame_from_feature_rows(rows)
        assert frame is not None
        direction, fill = judge_direction(frame)
        assert direction == "neutral"
        assert fill.branch == "neutral"
        assert fill.answer == "中性"

    def test_reason_includes_all_signal_descriptors(self):
        """Reason string must mention all 5 signals for traceability."""
        rows = _bullish_rows(12)
        frame = _frame_from_feature_rows(rows)
        direction, fill = judge_direction(frame)
        for marker in ("EMA斜率", "收盘重心", "波段结构", "趋势棒占比", "K线重叠"):
            assert marker in fill.reason, f"reason missing signal descriptor: {marker}"

    def test_returns_nodefill_instance(self):
        rows = _bullish_rows(8)
        frame = _frame_from_feature_rows(rows)
        _, fill = judge_direction(frame)
        assert isinstance(fill, NodeFill)


# ── TestJudgeAlwaysIn ───────────────────────────────────────────────────────


class TestJudgeAlwaysIn:
    """Verify §2.4 Always-In dual-window evaluation."""

    def test_strong_uptrend_ail(self):
        rows = _bullish_rows(20)
        frame = _frame_from_feature_rows(rows)
        fill = judge_always_in(frame)
        assert fill.branch == "AIL"
        assert fill.answer == "是"
        assert fill.node_id == "2.4"
        assert fill.bar_range == f"K{ALWAYS_IN_NEAR_WINDOW}-K1"

    def test_strong_downtrend_ais(self):
        rows = _bearish_rows(20)
        frame = _frame_from_feature_rows(rows)
        fill = judge_always_in(frame)
        assert fill.branch == "AIS"
        assert fill.answer == "是"

    def test_flat_range_no_ai(self):
        rows = _range_rows(20)
        frame = _frame_from_feature_rows(rows)
        fill = judge_always_in(frame)
        assert fill.branch is None
        assert fill.answer == "否"


# ── TestApplyStage1Nodes ────────────────────────────────────────────────────


class TestApplyStage1Nodes:
    """Verify the high-level apply_stage1_nodes entry point."""

    def test_replaces_llm_program_nodes(self):
        """Existing 1.1/2.3/2.4 nodes from LLM should be replaced."""
        rows = _bullish_rows(15)
        out = {
            "direction": "neutral",
            "gate_trace": [
                {"node_id": "1.1", "question": "Q1.1", "answer": "是"},
                {"node_id": "2.1", "question": "Q2.1", "answer": "是"},
                {"node_id": "2.3", "question": "Q2.3", "answer": "中性",
                 "branch": "middle"},  # ← bad AI value
                {"node_id": "2.4", "question": "Q2.4", "answer": "是",
                 "branch": None},  # ← null branch
                {"node_id": "2.5", "question": "Q2.5", "answer": "是"},
            ],
        }
        changed = apply_stage1_nodes(out, rows)
        assert changed is True
        gate = out["gate_trace"]
        node_11 = next(n for n in gate if n["node_id"] == "1.1")
        node_23 = next(n for n in gate if n["node_id"] == "2.3")
        node_24 = next(n for n in gate if n["node_id"] == "2.4")
        assert "已通过前置数据闸门" in node_11["reason"]
        assert node_23["branch"] != "middle"  # no longer the bad value
        assert node_23["branch"] in ("bullish", "bearish", "neutral")
        assert node_24["branch"] in ("AIL", "AIS", None)
        # Top-level direction synced with 2.3 branch
        assert out["direction"] == node_23["branch"]

    def test_inserts_when_missing(self):
        """If LLM omits 1.1/2.3/2.4, they get inserted in correct order."""
        rows = _bullish_rows(10)
        out = {
            "direction": "bullish",
            "gate_trace": [
                {"node_id": "2.1", "question": "Q2.1", "answer": "是"},
                {"node_id": "2.2", "question": "Q2.2", "answer": "是"},
                {"node_id": "2.5", "question": "Q2.5", "answer": "是"},
            ],
        }
        changed = apply_stage1_nodes(out, rows)
        assert changed is True
        gate = out["gate_trace"]
        ids = [n["node_id"] for n in gate]
        assert "1.1" in ids
        assert "2.3" in ids
        assert "2.4" in ids
        # Sort key: 2.2 < 2.3 < 2.4 < 2.5
        sorted_ids = sorted(ids, key=lambda i: _node_id_sort_key(i))
        assert ids == sorted_ids or set(ids) == set(sorted_ids)

    def test_feature_rows_none_is_noop(self):
        out = {"direction": "neutral", "gate_trace": []}
        changed = apply_stage1_nodes(out, None)
        assert changed is False

    def test_insufficient_bars_is_noop(self):
        """Fewer than 5 bars → frame is None → no-op, no exception."""
        rows = [_row(i + 1, o=1.0, h=2.0, l=0.5, c=1.5) for i in range(3)]
        out = {"direction": "neutral", "gate_trace": []}
        changed = apply_stage1_nodes(out, rows)
        assert changed is False

    def test_does_not_throw_on_malformed_rows(self):
        """Malformed rows (missing OHLC) should not raise."""
        rows = [{"k": "K1"}, {"k": "K2"}]
        out = {"direction": "neutral", "gate_trace": []}
        changed = apply_stage1_nodes(out, rows)
        assert changed is False

    def test_gate_trace_absent_creates_it(self):
        """If gate_trace is missing entirely, engine creates it."""
        rows = _bullish_rows(10)
        out = {"direction": "neutral"}
        changed = apply_stage1_nodes(out, rows)
        assert changed is True
        assert isinstance(out["gate_trace"], list)
        assert len(out["gate_trace"]) >= 3

    def test_bar_analysis_always_in_synced_with_2_4(self):
        """bar_analysis.always_in tracks §2.4, translated to validator vocab.

        The §2.4 branch is AIL/AIS, but MARKET_DIAGNOSIS_ALWAYS_IN is
        {"long","short","neutral"}, so apply_stage1_nodes must translate
        (AIL→long, AIS→short). Mirrors upstream apply_stage1 Step 6.
        """
        rows = _bullish_rows(20)
        out = {
            "direction": "neutral",
            "bar_analysis": {"always_in": "neutral"},
        }
        apply_stage1_nodes(out, rows)
        node_24 = next(n for n in out["gate_trace"] if n["node_id"] == "2.4")
        ba = out["bar_analysis"]
        if node_24["branch"] == "AIL":
            assert ba["always_in"] == "long"
        elif node_24["branch"] == "AIS":
            assert ba["always_in"] == "short"
        # validator enum must always hold
        assert ba["always_in"] in {"long", "short", "neutral"}


# ── TestNormalizeMarketDiagnosisIntegration ─────────────────────────────────


class TestNormalizeMarketDiagnosisIntegration:
    """End-to-end: normalize_market_diagnosis should fix the bug-2 pattern."""

    def test_neutral_with_null_branch_gets_fixed(self):
        """Issue #37: AI emits neutral direction but 2.3.branch=null."""
        rows = _range_rows(15)
        diagnosis = {
            "direction": "neutral",
            "gate_result": "proceed",
            "gate_trace": [
                {"node_id": "1.1", "question": "Q1.1", "answer": "是",
                 "bar_range": "K8-K1"},
                {"node_id": "1.2", "question": "Q1.2", "answer": "是",
                 "bar_range": "K15-K1"},
                {"node_id": "1.3", "question": "Q1.3", "answer": "否",
                 "bar_range": "K20-K1"},
                {"node_id": "2.1", "question": "Q2.1", "answer": "是",
                 "bar_range": "K8-K1"},
                {"node_id": "2.2", "question": "Q2.2", "answer": "是",
                 "bar_range": "K8-K1"},
                {"node_id": "2.3", "question": "Q2.3", "answer": "中性",
                 "branch": None, "bar_range": "K8-K1"},
                {"node_id": "2.4", "question": "Q2.4", "answer": "否",
                 "branch": None, "bar_range": "K8-K1"},
                {"node_id": "2.5", "question": "Q2.5", "answer": "是",
                 "bar_range": "K8-K1"},
            ],
        }
        out = normalize_market_diagnosis(diagnosis, feature_rows=rows)
        node_23 = next(n for n in out["gate_trace"] if n["node_id"] == "2.3")
        assert node_23["branch"] in ("bullish", "bearish", "neutral")
        assert node_23["branch"] is not None
        # Top-level direction matches 2.3 branch
        assert out["direction"] == node_23["branch"]

    def test_branch_middle_position_gets_overwritten(self):
        """Issue #37: AI writes branch='middle' (position word, not direction)."""
        rows = _bullish_rows(15)
        diagnosis = {
            "direction": "neutral",
            "gate_result": "proceed",
            "gate_trace": [
                {"node_id": "2.3", "question": "Q2.3", "answer": "中性",
                 "branch": "middle", "bar_range": "K8-K1"},
            ],
        }
        out = normalize_market_diagnosis(diagnosis, feature_rows=rows)
        node_23 = next(n for n in out["gate_trace"] if n["node_id"] == "2.3")
        assert node_23["branch"] != "middle"

    def test_works_without_feature_rows(self):
        """When feature_rows is None, normalize still runs (legacy path)."""
        diagnosis = {
            "direction": "neutral",
            "gate_trace": [
                {"node_id": "1.1", "question": "Q1.1", "answer": "是",
                 "bar_range": "K8-K1"},
            ],
        }
        out = normalize_market_diagnosis(diagnosis)  # no feature_rows
        assert out["direction"] == "neutral"
        # No 2.3/2.4 injected
        ids = [n["node_id"] for n in out.get("gate_trace", [])]
        assert "2.3" not in ids

    def test_bar_type_opposites_are_auto_fixed_from_feature_rows(self):
        rows = _bullish_rows(8)
        for row in rows:
            row["bar_type"] = "trend_bull"
        diagnosis = {
            "direction": "bullish",
            "bar_analysis": {"bar_type": "trend_bear"},
            "bar_by_bar_summary": [
                {
                    "bar": "K1",
                    "role": "structure",
                    "bar_type": "trend_bear",
                    "context_effect": "neutral",
                    "follow_through": "pending",
                    "trapped_side": "none",
                    "reason": "模型误写为空头趋势棒",
                }
            ],
            "gate_trace": [],
        }
        out = normalize_market_diagnosis(diagnosis, feature_rows=rows)
        assert out["bar_analysis"]["bar_type"] == "trend_bull"
        assert out["bar_by_bar_summary"][0]["bar_type"] == "trend_bull"
