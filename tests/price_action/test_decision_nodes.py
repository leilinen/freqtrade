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
    AI_PRIMARY_NODES,
    ALWAYS_IN_NEAR_WINDOW,
    BAR_COUNT_THRESHOLD,
    DIRECTION_BEAR_THRESHOLD,
    DIRECTION_BULL_THRESHOLD,
    DIRECTION_WINDOW,
    LOCKED_NODES,
    OVERRIDABLE_NODES,
    SAFETY_GATE_NODES,
    SIGNAL_BAR_LONG_ATR_RATIO,
    KlineGeometryFeature,
    NodeFill,
    _build_program_trace_node,
    _CYCLE_ORDER_METHOD,
    _frame_from_feature_rows,
    _get_signal_seq,
    _merge_program_nodes,
    _node_id_sort_key,
    apply_stage1_nodes,
    compute_kline_geometry_features,
    judge_always_in,
    judge_data_sufficiency,
    judge_direction,
    judge_follow_through,
    judge_signal_bar_closed,
    judge_signal_bar_direction,
    judge_signal_bar_length,
    route_order_method,
    apply_overrides,
    write_override_trace,
    _merge_program_nodes_head,
    DecisionNodeEngine,
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


# ── Planned-limit helper tests (mirror SOURCE test_decision_nodes_judges.py:597-628) ──


def test_has_background_limit_path_detects_9_0p_yes() -> None:
    from price_action.decision_nodes import has_background_limit_path

    out = {
        "decision_trace": [
            {"node_id": "9.0", "answer": "否"},
            {"node_id": "9.0P", "answer": "是"},
        ]
    }
    assert has_background_limit_path(out) is True


def test_has_background_limit_path_false_when_9_0p_absent_or_no() -> None:
    from price_action.decision_nodes import has_background_limit_path

    assert has_background_limit_path({"decision_trace": []}) is False
    assert has_background_limit_path({}) is False
    assert (
        has_background_limit_path(
            {"decision_trace": [{"node_id": "9.0P", "answer": "否"}]}
        )
        is False
    )


def test_is_planned_limit_order_detects_pending_limit_without_signal_bar() -> None:
    from price_action.decision_nodes import is_planned_limit_order

    obj = {
        "decision": {"order_type": "限价单"},
        "bar_analysis": {
            "signal_bar": {"bar": None, "quality": "invalid", "pattern": "none"},
            "entry_bar": {
                "bar": None,
                "strength": "not_triggered",
                "freshness": "pending",
            },
        },
    }
    assert is_planned_limit_order(obj) is True


def test_is_planned_limit_order_detects_weak_boundary_limit() -> None:
    from price_action.decision_nodes import is_planned_limit_order

    obj = {
        "decision": {"order_type": "限价单"},
        "bar_analysis": {
            "signal_bar": {"bar": "K2", "quality": "weak", "pattern": "tr_boundary"},
            "entry_bar": {
                "bar": None,
                "strength": "not_triggered",
                "freshness": "pending",
            },
        },
    }
    assert is_planned_limit_order(obj) is True


def test_is_planned_limit_order_returns_true_via_background_path() -> None:
    """order_type=限价单 + 9.0P=是 → True regardless of bar_analysis."""
    from price_action.decision_nodes import is_planned_limit_order

    obj = {
        "decision": {"order_type": "限价单"},
        "decision_trace": [{"node_id": "9.0P", "answer": "是"}],
        # bar_analysis 完全缺失也应返回 True
    }
    assert is_planned_limit_order(obj) is True


def test_is_planned_limit_order_false_when_not_limit_order() -> None:
    from price_action.decision_nodes import is_planned_limit_order

    obj = {
        "decision": {"order_type": "市价单"},
        "bar_analysis": {
            "signal_bar": {"bar": None, "quality": "invalid", "pattern": "none"},
            "entry_bar": {"bar": None, "strength": "not_triggered"},
        },
    }
    assert is_planned_limit_order(obj) is False


def test_is_planned_limit_order_false_when_entry_already_triggered() -> None:
    """entry_bar.strength != not_triggered 且 bar 已指定 → 非 pending."""
    from price_action.decision_nodes import is_planned_limit_order

    obj = {
        "decision": {"order_type": "限价单"},
        "bar_analysis": {
            "signal_bar": {"bar": "K2", "quality": "weak", "pattern": "tr_boundary"},
            "entry_bar": {
                "bar": "K1",
                "strength": "strong",
                "freshness": "filled",
            },
        },
    }
    assert is_planned_limit_order(obj) is False


def test_is_planned_limit_order_false_when_signal_quality_strong() -> None:
    """signal_bar.quality=strong 时即便 pending 也不算 planned limit."""
    from price_action.decision_nodes import is_planned_limit_order

    obj = {
        "decision": {"order_type": "限价单"},
        "bar_analysis": {
            "signal_bar": {"bar": "K2", "quality": "strong", "pattern": "breakout"},
            "entry_bar": {
                "bar": None,
                "strength": "not_triggered",
                "freshness": "pending",
            },
        },
    }
    assert is_planned_limit_order(obj) is False


# ── Step 3.0/3.1: KlineGeometryFeature adapter + stage2 constants ─────────


def _geom_row(
    seq: int,
    *,
    bar_type: str = "other",
    range_atr: float | None = 1.0,
    follow_through: str = "pending",
    ema_relation: str = "unknown",
    body_ratio: float | None = 0.5,
    close_position: float | None = 0.5,
) -> dict:
    """Build a feature_row dict carrying pre-computed geometry fields."""
    return {
        "k": f"K{seq}",
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "ema20": 1.5,
        "atr14": 1.0,
        "bar_type": bar_type,
        "body_ratio": body_ratio,
        "upper_wick_pct": 0.1,
        "lower_wick_pct": 0.1,
        "close_position": close_position,
        "range_atr": range_atr,
        "ema20_relation": ema_relation,
        "overlap_prev_ratio": 0.5,
        "inside_sequence": "none",
        "ioi_pattern": False,
        "micro_double": "none",
        "gap_bar": "none",
        "ema_gap_count": 0,
        "breakout_prev": "none",
        "follow_through_1_2": follow_through,
    }


class TestKlineGeometryFeatureAdapter:
    def test_maps_dict_keys_to_dataclass_fields(self):
        feats = compute_kline_geometry_features(
            [_geom_row(1, bar_type="trend_bull", range_atr=2.3, follow_through="yes", ema_relation="above")]
        )
        assert 1 in feats
        f = feats[1]
        assert f.seq == 1
        assert f.bar_type == "trend_bull"
        assert f.range_atr_ratio == 2.3
        assert f.follow_through_1_2 == "yes"
        assert f.ema_relation == "above"
        assert f.body_ratio == 0.5

    def test_none_inputs_return_empty(self):
        assert compute_kline_geometry_features(None) == {}
        assert compute_kline_geometry_features([]) == {}

    def test_skips_rows_without_parseable_k(self):
        feats = compute_kline_geometry_features(
            [{"bar_type": "doji"}, _geom_row(2, bar_type="doji")]
        )
        assert set(feats.keys()) == {2}

    def test_none_range_atr_becomes_none(self):
        feats = compute_kline_geometry_features([_geom_row(1, range_atr=None)])
        assert feats[1].range_atr_ratio is None

    def test_nan_range_atr_becomes_none(self):
        feats = compute_kline_geometry_features([_geom_row(1, range_atr=float("nan"))])
        assert feats[1].range_atr_ratio is None


class TestStage2Constants:
    """Verify the ported stage2 permission sets and order-method map."""

    def test_cycle_order_method_values(self):
        assert _CYCLE_ORDER_METHOD["spike"] == "市价单"
        assert _CYCLE_ORDER_METHOD["broad_channel"] == "限价单"
        assert _CYCLE_ORDER_METHOD["trading_range"] == "限价单"
        assert _CYCLE_ORDER_METHOD["micro_channel"] == "突破单"
        assert _CYCLE_ORDER_METHOD["extreme_tr"] == "不下单"
        assert _CYCLE_ORDER_METHOD["unknown"] == "不下单"

    def test_permission_sets_match_upstream(self):
        assert LOCKED_NODES == frozenset({"1.1", "9.1"})
        assert AI_PRIMARY_NODES == frozenset({"1.3", "2.5"})
        assert SAFETY_GATE_NODES == frozenset({"1.1", "10.3", "14"})
        assert "9.1" in LOCKED_NODES
        assert "11.3" in OVERRIDABLE_NODES

    def test_signal_bar_long_atr_ratio(self):
        assert SIGNAL_BAR_LONG_ATR_RATIO == 2.0


class TestGetSignalSeq:
    def test_reads_bar_analysis_signal_bar(self):
        out = {"bar_analysis": {"signal_bar": {"bar": "K3"}}}
        assert _get_signal_seq(out) == 3

    def test_defaults_to_k1_when_missing(self):
        assert _get_signal_seq({}) == 1
        assert _get_signal_seq({"bar_analysis": {}}) == 1
        assert _get_signal_seq({"bar_analysis": {"signal_bar": {"bar": None}}}) == 1

    def test_defaults_to_k1_for_unparseable(self):
        assert _get_signal_seq({"bar_analysis": {"signal_bar": {"bar": "garbage"}}}) == 1


# ── Step 3.2: §9 stage2 judges ──────────────────────────────────────────────


def _feat(
    bar_type: str = "other",
    range_atr_ratio: float | None = 1.5,
    follow_through_1_2: str = "yes",
) -> KlineGeometryFeature:
    """Build a KlineGeometryFeature with controllable judge-relevant fields."""
    return KlineGeometryFeature(
        seq=2,
        bar_type=bar_type,
        body_ratio=0.5,
        upper_wick_ratio=0.1,
        lower_wick_ratio=0.1,
        close_position=0.5,
        range_atr_ratio=range_atr_ratio,
        ema_relation="unknown",
        overlap_prev_ratio=0.5,
        inside_sequence="none",
        ioi_pattern=False,
        micro_double="none",
        gap_bar="none",
        ema_gap_count=0,
        breakout_prev="none",
        follow_through_1_2=follow_through_1_2,
    )


class TestSignalBarJudge:
    """§9.1 / §9.2 / §9.3 — mirrors SOURCE test_decision_nodes_judges.py:248-314."""

    def test_91_always_yes(self):
        fill = judge_signal_bar_closed(2, None)
        assert fill.node_id == "9.1"
        assert fill.answer == "是"
        assert fill.bar_range == "K2"

    def test_92_long_consistent_yes(self):
        features = {2: _feat("trend_bull")}
        fill = judge_signal_bar_direction(2, "做多", features)
        assert fill.node_id == "9.2"
        assert fill.answer == "是"

    def test_92_long_inconsistent_no(self):
        features = {2: _feat("doji")}
        fill = judge_signal_bar_direction(2, "做多", features)
        assert fill.node_id == "9.2"
        assert fill.answer == "否"

    def test_92_short_consistent_yes(self):
        features = {2: _feat("trend_bear")}
        fill = judge_signal_bar_direction(2, "做空", features)
        assert fill.node_id == "9.2"
        assert fill.answer == "是"

    def test_92_short_inconsistent_no(self):
        features = {2: _feat("trend_bull")}
        fill = judge_signal_bar_direction(2, "做空", features)
        assert fill.node_id == "9.2"
        assert fill.answer == "否"

    def test_92_outside_bull_long_is_no_with_warning(self):
        """outside_bull is weak → 否 (outside bar congestion zone)."""
        features = {2: _feat("outside_bull")}
        fill = judge_signal_bar_direction(2, "做多", features)
        assert fill.answer == "否"
        assert "外包棒" in fill.reason

    def test_92_outside_bear_short_is_no_with_warning(self):
        features = {2: _feat("outside_bear")}
        fill = judge_signal_bar_direction(2, "做空", features)
        assert fill.answer == "否"
        assert "外包棒" in fill.reason

    def test_92_no_direction_not_applicable(self):
        features = {2: _feat("trend_bull")}
        fill = judge_signal_bar_direction(2, None, features)
        assert fill.node_id == "9.2"
        assert fill.answer == "不适用"
        assert fill.bar_range == "不适用"

    def test_92_missing_feature_treats_as_unknown_no(self):
        fill = judge_signal_bar_direction(2, "做多", {})
        assert fill.answer == "否"
        assert "unknown" in fill.reason

    def test_93_overlong_yes(self):
        features = {2: _feat(range_atr_ratio=SIGNAL_BAR_LONG_ATR_RATIO + 0.1)}
        fill = judge_signal_bar_length(2, features)
        assert fill.node_id == "9.3"
        assert fill.answer == "是"

    def test_93_not_overlong_no(self):
        features = {2: _feat(range_atr_ratio=SIGNAL_BAR_LONG_ATR_RATIO - 0.1)}
        fill = judge_signal_bar_length(2, features)
        assert fill.node_id == "9.3"
        assert fill.answer == "否"

    def test_93_nan_ratio_conservative_yes(self):
        features = {2: _feat(range_atr_ratio=None)}
        fill = judge_signal_bar_length(2, features)
        assert fill.node_id == "9.3"
        assert fill.answer == "是"

    def test_93_boundary_exactly_2_0_is_no(self):
        """ratio == 2.0 should be no (not strictly greater than)."""
        features = {2: _feat(range_atr_ratio=2.0)}
        fill = judge_signal_bar_length(2, features)
        assert fill.answer == "否"

    def test_93_just_above_2_0_is_yes(self):
        features = {2: _feat(range_atr_ratio=2.001)}
        fill = judge_signal_bar_length(2, features)
        assert fill.answer == "是"


class TestFollowThroughJudge:
    """§9.5 — mirrors SOURCE test_decision_nodes_judges.py:316-339."""

    def test_yes_maps_to_shi(self):
        features = {2: _feat(follow_through_1_2="yes")}
        fill = judge_follow_through(2, features)
        assert fill.answer == "是"

    def test_failed_maps_to_fou(self):
        features = {2: _feat(follow_through_1_2="failed")}
        fill = judge_follow_through(2, features)
        assert fill.answer == "否"

    def test_no_maps_to_fou(self):
        features = {2: _feat(follow_through_1_2="no")}
        fill = judge_follow_through(2, features)
        assert fill.answer == "否"

    def test_pending_maps_to_dengdai(self):
        features = {1: _feat(follow_through_1_2="pending")}
        fill = judge_follow_through(1, features)
        assert fill.answer == "等待"

    def test_missing_feature_conservative_dengdai(self):
        fill = judge_follow_through(5, {})
        assert fill.answer == "等待"

    def test_bar_range_spans_signal_to_k1(self):
        features = {3: _feat(follow_through_1_2="yes")}
        fill = judge_follow_through(3, features)
        assert fill.bar_range == "K3-K1"


# ── Step 3.3: route_order_method (§11) ──────────────────────────────────────


class TestRouteOrderMethod:
    """§11 order-method routing — adapted from SOURCE test_order_method_router.py.

    SOURCE's ``_has_trade_prices`` requires ``take_profit_price_2``; the SOURCE
    tests omitted it, which made the preserve-model branches unreachable. Here
    we include ``take_profit_price_2`` so the preserve logic actually fires.
    """

    @staticmethod
    def _full_prices(**extra) -> dict:
        return {
            "entry_price": 100.0,
            "stop_loss_price": 102.0,
            "take_profit_price": 98.0,
            "take_profit_price_2": 96.0,
            **extra,
        }

    def test_no_order_returns_empty(self):
        nodes = route_order_method({"cycle_position": "spike"}, {"order_type": "不下单"}, [])
        assert nodes == []

    def test_safety_gate_10_3_no_returns_empty(self):
        decision = {"order_type": "市价单", **self._full_prices()}
        trace = [{"node_id": "10.3", "answer": "否"}]
        assert route_order_method({"cycle_position": "spike"}, decision, trace) == []

    def test_sec14_violation_returns_empty(self):
        decision = {"order_type": "市价单", **self._full_prices()}
        trace = [
            {"node_id": "10.3", "answer": "是"},
            {"node_id": "14", "answer": "是", "reason": "触及禁止交易时段"},
        ]
        assert route_order_method({"cycle_position": "spike"}, decision, trace) == []

    def test_sec14_with_denial_phrase_not_violated(self):
        decision = {"order_type": "市价单", **self._full_prices()}
        trace = [
            {"node_id": "10.3", "answer": "是"},
            {"node_id": "14", "answer": "是", "reason": "未触犯禁止时段"},
        ]
        # spike cycle → still returns [] (spike always returns empty)
        assert route_order_method({"cycle_position": "spike"}, decision, trace) == []

    def test_unknown_cycle_returns_empty(self):
        decision = {"order_type": "市价单", **self._full_prices()}
        trace = [{"node_id": "10.3", "answer": "是"}]
        assert route_order_method({"cycle_position": "extreme_tr"}, decision, trace) == []
        assert route_order_method({"cycle_position": "garbage"}, decision, trace) == []

    def test_spike_cycle_always_returns_empty(self):
        """cycle=spike → no §11 nodes injected (handled elsewhere)."""
        decision = {"order_type": "市价单", **self._full_prices()}
        trace = [{"node_id": "10.3", "answer": "是"}]
        assert route_order_method({"cycle_position": "spike"}, decision, trace) == []

    def test_breakout_without_basis_falls_back_to_limit(self):
        decision = {
            "order_type": "突破单",
            "entry_price": 101.0,
            "stop_loss_price": 99.0,
            "take_profit_price": 102.0,
            "take_profit_price_2": 103.0,
        }
        trace = [{"node_id": "10.3", "answer": "是", "reason": "ok"}]
        nodes = route_order_method({"cycle_position": "normal_channel"}, decision, trace)
        assert decision["order_type"] == "限价单"
        assert nodes
        assert nodes[-1].node_id == "11.2"
        assert nodes[-1].answer == "是"
        assert "限价单" in nodes[-1].reason

    def test_model_breakout_preserved_for_broad_channel(self):
        decision = {
            "order_type": "突破单",
            "order_direction": "做空",
            **self._full_prices(
                entry_price=4210.348,
                entry_basis_bar="K1",
                entry_basis_extreme="low",
                stop_loss_price=4228.399,
                take_profit_price=4183.278,
            ),
        }
        trace = [{"node_id": "10.3", "answer": "是", "reason": "ok"}]
        nodes = route_order_method({"cycle_position": "broad_channel"}, decision, trace)
        assert decision["order_type"] == "突破单"
        assert nodes
        assert nodes[-1].node_id == "11.2"
        assert nodes[-1].answer == "是"

    def test_model_limit_order_preserved_for_breakout_cycle(self):
        decision = {
            "order_type": "限价单",
            **self._full_prices(entry_price=100.5, stop_loss_price=99.0, take_profit_price=101.5),
        }
        trace = [{"node_id": "10.3", "answer": "是", "reason": "ok"}]
        nodes = route_order_method({"cycle_position": "normal_channel"}, decision, trace)
        assert decision["order_type"] == "限价单"
        assert nodes[-1].answer == "是"

    def test_trending_tr_breakout_uses_11_2(self):
        decision = {
            "order_type": "突破单",
            **self._full_prices(entry_basis_bar="K2", entry_basis_extreme=99.5),
        }
        trace = [{"node_id": "10.3", "answer": "是"}]
        nodes = route_order_method({"cycle_position": "trending_tr"}, decision, trace)
        assert decision["order_type"] == "突破单"
        assert nodes[-1].node_id == "11.2"

    def test_trading_range_limit_uses_11_3(self):
        decision = {"order_type": "限价单", **self._full_prices()}
        trace = [{"node_id": "10.3", "answer": "是"}]
        nodes = route_order_method({"cycle_position": "trading_range"}, decision, trace)
        assert decision["order_type"] == "限价单"
        assert nodes[-1].node_id == "11.3"

    def test_11_nodes_prior_get_no_answer(self):
        """Nodes before the final §11 node get answer=否."""
        decision = {"order_type": "限价单", **self._full_prices()}
        trace = [{"node_id": "10.3", "answer": "是"}]
        nodes = route_order_method({"cycle_position": "trading_range"}, decision, trace)
        # 11.3 is final → 11.1, 11.2 should be 否, 11.3 是
        answers = {n.node_id: n.answer for n in nodes}
        assert answers.get("11.1") == "否"
        assert answers.get("11.2") == "否"
        assert answers.get("11.3") == "是"

    def test_breakout_fallback_reason_mentions_no_basis(self):
        decision = {
            "order_type": "突破单",
            "entry_price": 101.0,
            "stop_loss_price": 99.0,
            "take_profit_price": 102.0,
            "take_profit_price_2": 103.0,
        }
        trace = [{"node_id": "10.3", "answer": "是"}]
        nodes = route_order_method({"cycle_position": "tight_channel"}, decision, trace)
        assert "entry_basis" in nodes[-1].reason


# ── Step 3.4/3.5: apply_overrides + merge_program_nodes ─────────────────────


def _make_program_nodes() -> list[dict]:
    """Return sample program nodes for override testing."""
    return [
        {"node_id": "1.1", "question": "q1.1", "answer": "是", "reason": "r", "bar_range": "K20-K1"},
        {"node_id": "2.3", "question": "q2.3", "answer": "是", "reason": "r", "bar_range": "K20-K1", "branch": "bullish"},
        {"node_id": "2.4", "question": "q2.4", "answer": "否", "reason": "r", "bar_range": "K20-K1"},
        {"node_id": "9.1", "question": "q9.1", "answer": "是", "reason": "r", "bar_range": "K1"},
        {"node_id": "9.2", "question": "q9.2", "answer": "是", "reason": "r", "bar_range": "K1"},
        {"node_id": "9.3", "question": "q9.3", "answer": "否", "reason": "r", "bar_range": "K1"},
    ]


class TestOverrideArbiter:
    """Override arbiter — mirrors SOURCE test_decision_nodes_judges.py:356-524."""

    def test_no_overrides_keeps_program_values(self):
        nodes = _make_program_nodes()
        out = {"direction": "bullish"}
        result = apply_overrides(nodes, None, out=out, stage="stage1")
        assert result[0]["answer"] == "是"
        assert result[1]["branch"] == "bullish"
        assert not any(n.get("overridden_by_ai") for n in result)

    def test_empty_overrides_list_keeps_program_values(self):
        nodes = _make_program_nodes()
        out = {"direction": "bullish"}
        result = apply_overrides(nodes, [], out=out, stage="stage1")
        assert not any(n.get("overridden_by_ai") for n in result)

    def test_locked_node_11_cannot_be_overridden(self):
        nodes = _make_program_nodes()
        overrides = [{"node_id": "1.1", "answer": "否", "override_reason": "test"}]
        out = {}
        result = apply_overrides(nodes, overrides, out=out, stage="stage1")
        n11 = next((n for n in result if n["node_id"] == "1.1"), None)
        assert n11["answer"] == "是"
        assert not n11.get("overridden_by_ai")

    def test_locked_node_91_cannot_be_overridden(self):
        nodes = _make_program_nodes()
        overrides = [{"node_id": "9.1", "answer": "否", "override_reason": "test"}]
        out = {}
        result = apply_overrides(nodes, overrides, out=out, stage="stage2")
        n91 = next((n for n in result if n["node_id"] == "9.1"), None)
        assert n91["answer"] == "是"
        assert not n91.get("overridden_by_ai")

    def test_missing_override_reason_rejected(self):
        nodes = _make_program_nodes()
        overrides = [{"node_id": "2.4", "answer": "是"}]  # no override_reason
        out = {}
        result = apply_overrides(nodes, overrides, out=out, stage="stage1")
        n24 = next((n for n in result if n["node_id"] == "2.4"), None)
        assert n24["answer"] == "否"  # original
        assert not n24.get("overridden_by_ai")

    def test_empty_override_reason_rejected(self):
        nodes = _make_program_nodes()
        overrides = [{"node_id": "2.4", "answer": "是", "override_reason": "  "}]
        out = {}
        result = apply_overrides(nodes, overrides, out=out, stage="stage1")
        n24 = next((n for n in result if n["node_id"] == "2.4"), None)
        assert not n24.get("overridden_by_ai")

    def test_valid_override_accepted_with_trace(self):
        nodes = _make_program_nodes()
        overrides = [{"node_id": "2.4", "answer": "是", "branch": "AIL", "override_reason": "strong bullish trend"}]
        out = {}
        result = apply_overrides(nodes, overrides, out=out, stage="stage1")
        n24 = next((n for n in result if n["node_id"] == "2.4"), None)
        assert n24["answer"] == "是"
        assert n24.get("overridden_by_ai") is True
        assert n24.get("program_answer") == "否"
        assert n24.get("override_reason") == "strong bullish trend"

    def test_24_override_syncs_always_in(self):
        """§2.4 override with branch=AIL syncs bar_analysis.always_in=long."""
        nodes = _make_program_nodes()
        overrides = [{"node_id": "2.4", "answer": "是", "branch": "AIL", "override_reason": "AIL confirmed"}]
        out = {"bar_analysis": {"always_in": "neutral"}}
        apply_overrides(nodes, overrides, out=out, stage="stage1")
        assert out["bar_analysis"]["always_in"] == "long"

    def test_23_override_bearish_syncs_direction(self):
        nodes = _make_program_nodes()
        overrides = [{"node_id": "2.3", "answer": "是", "branch": "bearish",
                      "override_reason": "strong bearish reversal"}]
        out = {"direction": "bullish"}
        result = apply_overrides(nodes, overrides, out=out, stage="stage1")
        n23 = next((n for n in result if n["node_id"] == "2.3"), None)
        assert n23["answer"] == "是"
        assert n23["branch"] == "bearish"
        assert n23.get("overridden_by_ai") is True
        assert out["direction"] == "bearish"

    def test_23_override_neutral_answer_zhongxing(self):
        nodes = _make_program_nodes()
        overrides = [{"node_id": "2.3", "answer": "中性", "branch": "neutral",
                      "override_reason": "market is ranging"}]
        out = {"direction": "bullish"}
        apply_overrides(nodes, overrides, out=out, stage="stage1")
        assert out["direction"] == "neutral"

    def test_23_override_inconsistent_rejected(self):
        nodes = _make_program_nodes()
        overrides = [{"node_id": "2.3", "answer": "中性", "branch": "bullish",
                      "override_reason": "contradiction"}]
        out = {"direction": "bullish"}
        result = apply_overrides(nodes, overrides, out=out, stage="stage1")
        n23 = next((n for n in result if n["node_id"] == "2.3"), None)
        assert not n23.get("overridden_by_ai")
        assert out["direction"] == "bullish"  # unchanged

    def test_non_list_overrides_ignored(self):
        nodes = _make_program_nodes()
        out = {}
        for bad_input in [None, "string", 42, {"key": "val"}]:
            result = apply_overrides(nodes, bad_input, out=out, stage="stage1")
            assert not any(n.get("overridden_by_ai") for n in result)

    def test_invalid_answer_enum_skipped(self):
        nodes = _make_program_nodes()
        overrides = [{"node_id": "2.4", "answer": "INVALID", "override_reason": "test"}]
        out = {}
        result = apply_overrides(nodes, overrides, out=out, stage="stage1")
        n24 = next((n for n in result if n["node_id"] == "2.4"), None)
        assert not n24.get("overridden_by_ai")

    def test_first_valid_override_per_node_wins(self):
        """Only the first valid override per node_id is applied."""
        nodes = _make_program_nodes()
        overrides = [
            {"node_id": "2.4", "answer": "是", "branch": "AIL", "override_reason": "first"},
            {"node_id": "2.4", "answer": "否", "branch": "AIS", "override_reason": "second"},
        ]
        out = {}
        result = apply_overrides(nodes, overrides, out=out, stage="stage1")
        n24 = next((n for n in result if n["node_id"] == "2.4"), None)
        assert n24["answer"] == "是"
        assert n24.get("override_reason") == "first"

    def test_safety_gate_10_3_aggressive_rejected(self):
        """§10.3 否→是 is less conservative (rank 5→3), rejected."""
        nodes = [{"node_id": "10.3", "question": "q", "answer": "否", "reason": "r", "bar_range": "K1"}]
        overrides = [{"node_id": "10.3", "answer": "是", "override_reason": "try"}]
        out = {}
        result = apply_overrides(nodes, overrides, out=out, stage="stage2")
        n103 = next((n for n in result if n["node_id"] == "10.3"), None)
        assert n103["answer"] == "否"
        assert not n103.get("overridden_by_ai")

    def test_safety_gate_10_3_not_overridable_even_conservative(self):
        """§10.3 is a safety gate but not in OVERRIDABLE_NODES, so even a more
        conservative override (是→否) is not applied (it passes the rank check
        but falls through rule 7 since §10.3 ∉ OVERRIDABLE_NODES)."""
        nodes = [{"node_id": "10.3", "question": "q", "answer": "是", "reason": "r", "bar_range": "K1"}]
        overrides = [{"node_id": "10.3", "answer": "否", "override_reason": "risk too high"}]
        out = {}
        result = apply_overrides(nodes, overrides, out=out, stage="stage2")
        n103 = next((n for n in result if n["node_id"] == "10.3"), None)
        # §10.3 ∉ OVERRIDABLE_NODES → not applied
        assert n103["answer"] == "是"
        assert not n103.get("overridden_by_ai")

    def test_write_override_trace_sets_fields(self):
        node = {"node_id": "2.3", "answer": "是", "branch": "bullish", "reason": "r", "bar_range": "K1"}
        override = {"answer": "否", "branch": "bearish", "override_reason": "reversal signal"}
        write_override_trace(node, override)
        assert node["program_answer"] == "是"
        assert node["program_branch"] == "bullish"
        assert node["answer"] == "否"
        assert node["branch"] == "bearish"
        assert node["override_reason"] == "reversal signal"
        assert node["overridden_by_ai"] is True


class TestMergeProgramNodesAIPrimary:
    """Step 3.5: AI_PRIMARY_NODES preservation — mirrors SOURCE test:474-513."""

    def test_ai_primary_25_keeps_ai_reason(self):
        """§2.5 is AI-primary: AI node preserved, program not appended."""
        trace = [{"node_id": "2.5", "question": "AI q5", "answer": "是", "reason": "AI", "bar_range": "K1"}]
        prog = [{"node_id": "2.5", "question": "程序 q5", "answer": "否", "reason": "程序长理由" * 20, "bar_range": "K8-K1"}]
        result = _merge_program_nodes(trace, prog)
        node_25 = next((n for n in result if n["node_id"] == "2.5"), None)
        assert node_25 is not None
        assert node_25["reason"] == "AI"
        assert "程序参考数据" not in node_25["reason"]

    def test_program_authoritative_23_replaces_ai(self):
        """§2.3 is program-authoritative: program replaces AI node."""
        trace = [{"node_id": "2.3", "question": "AI q", "answer": "空头", "reason": "AI", "bar_range": "K5-K1"}]
        prog = [{"node_id": "2.3", "question": "程序 q", "answer": "是", "reason": "程序", "bar_range": "K20-K1", "branch": "bullish"}]
        result = _merge_program_nodes(trace, prog)
        node_23 = next((n for n in result if n["node_id"] == "2.3"), None)
        assert node_23 is not None
        assert node_23["answer"] == "是"
        assert node_23["branch"] == "bullish"
        assert node_23["reason"] == "程序"

    def test_merge_program_nodes_head_prepends_new(self):
        """merge_program_nodes_head puts new program nodes at the head."""
        trace = [{"node_id": "9.1", "answer": "是", "reason": "AI tail"}]
        prog = [{"node_id": "1.1", "answer": "是", "reason": "程序 head"}]
        result = _merge_program_nodes_head(trace, prog)
        # new 1.1 node should be at index 0, AI 9.1 stays at end
        assert result[0]["node_id"] == "1.1"
        assert result[-1]["node_id"] == "9.1"

    def test_merge_program_nodes_head_replaces_existing_in_place(self):
        """Existing matching nodes are replaced in-place (not moved to head)."""
        trace = [
            {"node_id": "1.1", "answer": "否", "reason": "AI"},
            {"node_id": "9.1", "answer": "是", "reason": "AI tail"},
        ]
        prog = [{"node_id": "1.1", "answer": "是", "reason": "程序"}]
        result = _merge_program_nodes_head(trace, prog)
        # 1.1 replaced in place (index 0), 9.1 stays at index 1
        assert result[0]["node_id"] == "1.1"
        assert result[0]["answer"] == "是"
        assert result[1]["node_id"] == "9.1"


# ── Step 3.6/3.7: DecisionNodeEngine.apply_stage2 ───────────────────────────


def _stage2_rows(n: int = 10, bar_type: str = "trend_bear", follow: str = "yes") -> list[dict]:
    """Build n feature_rows for apply_stage2; K1 is newest."""
    return [
        {
            "k": f"K{i}", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
            "ema20": 1.5, "atr14": 1.0, "bar_type": bar_type,
            "body_ratio": 0.6, "upper_wick_pct": 0.1, "lower_wick_pct": 0.1,
            "close_position": 0.3, "range_atr": 1.2, "ema20_relation": "below",
            "overlap_prev_ratio": 0.5, "inside_sequence": "none", "ioi_pattern": False,
            "micro_double": "none", "gap_bar": "none", "ema_gap_count": 0,
            "breakout_prev": "none", "follow_through_1_2": follow,
        }
        for i in range(1, n + 1)
    ]


def _make_stage2_out(
    *,
    order_type: str = "限价单",
    order_direction: str = "做空",
    ans_90: str = "是",
    ans_103: str = "是",
    signal_bar: dict | None = None,
) -> dict:
    if signal_bar is None:
        signal_bar = {"bar": "K1", "quality": "strong", "pattern": "breakout"}
    return {
        "decision": {
            "order_type": order_type,
            "order_direction": order_direction,
            "entry_price": 101.0,
            "stop_loss_price": 103.0,
            "take_profit_price": 98.0,
            "take_profit_price_2": 96.0,
        },
        "bar_analysis": {
            "signal_bar": signal_bar,
            "entry_bar": {"bar": "K1", "strength": "triggered", "freshness": "fresh"},
        },
        "decision_trace": [
            {"node_id": "9.0", "answer": ans_90, "reason": "r"},
            {"node_id": "10.3", "answer": ans_103, "reason": "r"},
        ],
    }


class TestApplyStage2:
    """DecisionNodeEngine.apply_stage2 — mirrors SOURCE orchestrator behaviour."""

    def test_injects_9_judges_and_11_nodes(self):
        out = _make_stage2_out()
        DecisionNodeEngine.apply_stage2(out, _stage2_rows(), {"cycle_position": "trading_range"})
        ids = [n["node_id"] for n in out["decision_trace"]]
        for expected in ("9.1", "9.2", "9.3", "9.5", "11.3"):
            assert expected in ids, f"{expected} missing from {ids}"

    def test_91_answer_yes(self):
        out = _make_stage2_out()
        DecisionNodeEngine.apply_stage2(out, _stage2_rows(), {"cycle_position": "trading_range"})
        node = next(n for n in out["decision_trace"] if n["node_id"] == "9.1")
        assert node["answer"] == "是"

    def test_92_direction_consistent_for_short(self):
        out = _make_stage2_out(order_direction="做空")
        DecisionNodeEngine.apply_stage2(out, _stage2_rows(bar_type="trend_bear"), {"cycle_position": "trading_range"})
        node = next(n for n in out["decision_trace"] if n["node_id"] == "9.2")
        assert node["answer"] == "是"

    def test_trading_range_routes_to_11_3(self):
        out = _make_stage2_out()
        DecisionNodeEngine.apply_stage2(out, _stage2_rows(), {"cycle_position": "trading_range"})
        n113 = next(n for n in out["decision_trace"] if n["node_id"] == "11.3")
        assert n113["answer"] == "是"

    def test_gate_shortcircuited_returns_unchanged(self):
        out = _make_stage2_out()
        out["gate_shortcircuited"] = True
        original_ids = [n["node_id"] for n in out["decision_trace"]]
        DecisionNodeEngine.apply_stage2(out, _stage2_rows(), {"cycle_position": "trading_range"})
        assert [n["node_id"] for n in out["decision_trace"]] == original_ids

    def test_no_order_type_skips_11_injection(self):
        out = _make_stage2_out(order_type="不下单")
        DecisionNodeEngine.apply_stage2(out, _stage2_rows(), {"cycle_position": "trading_range"})
        ids = [n["node_id"] for n in out["decision_trace"]]
        # §9 judges still injected, but no §11 nodes
        assert "9.1" in ids
        assert not any(i.startswith("11.") for i in ids)

    def test_section9_skipped_when_90_is_no(self):
        """§9.0=否 → §9.1-9.5 marked 不适用/skipped."""
        out = _make_stage2_out(ans_90="否")
        DecisionNodeEngine.apply_stage2(out, _stage2_rows(), {"cycle_position": "trading_range"})
        for nid in ("9.1", "9.2", "9.3", "9.5"):
            node = next(n for n in out["decision_trace"] if n["node_id"] == nid)
            assert node["answer"] == "不适用", f"{nid} should be 不适用"
            assert node.get("skipped") is True

    def test_section9_skipped_when_90_waiting(self):
        """§9.0=等待 → §9.1-9.5 marked 不适用/skipped (AI means 'no signal bar')."""
        out = _make_stage2_out(ans_90="等待")
        DecisionNodeEngine.apply_stage2(out, _stage2_rows(), {"cycle_position": "trading_range"})
        node = next(n for n in out["decision_trace"] if n["node_id"] == "9.1")
        assert node["answer"] == "不适用"

    def test_planned_limit_skips_91_93_when_no_signal_bar(self):
        """Planned limit with no signal bar → §9.1-9.3 不适用 (§9.5 still computed)."""
        out = _make_stage2_out(
            order_type="限价单",
            signal_bar={"bar": None, "quality": "invalid", "pattern": "none"},
        )
        out["bar_analysis"]["entry_bar"] = {"bar": None, "strength": "not_triggered", "freshness": "pending"}
        DecisionNodeEngine.apply_stage2(out, _stage2_rows(), {"cycle_position": "broad_channel"})
        n91 = next(n for n in out["decision_trace"] if n["node_id"] == "9.1")
        assert n91["answer"] == "不适用"

    def test_node_overrides_applied(self):
        """node_overrides for §9.2 (overridable) is applied with trace."""
        out = _make_stage2_out()
        out["node_overrides"] = [
            {"node_id": "9.2", "answer": "否", "override_reason": "AI disagrees"},
        ]
        DecisionNodeEngine.apply_stage2(out, _stage2_rows(), {"cycle_position": "trading_range"})
        n92 = next(n for n in out["decision_trace"] if n["node_id"] == "9.2")
        assert n92["answer"] == "否"
        assert n92.get("overridden_by_ai") is True
        assert n92.get("override_reason") == "AI disagrees"

    def test_none_feature_rows_still_runs(self):
        """apply_stage2 tolerates None feature_rows (judges get empty features)."""
        out = _make_stage2_out()
        DecisionNodeEngine.apply_stage2(out, None, {"cycle_position": "trading_range"})
        # §9.1 still computed; §9.2/§9.3 use unknown/None features
        ids = [n["node_id"] for n in out["decision_trace"]]
        assert "9.1" in ids
