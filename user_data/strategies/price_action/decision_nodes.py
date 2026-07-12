"""DecisionNodeEngine port from upstream PA_Agent (stage1 §1.1/§2.3/§2.4 + stage2 §9.x/§11.x).

The upstream ``PA_Agent/pa_agent/ai/decision_nodes.py`` is 3041 lines and
covers stage1 (1.1/1.3/2.3/2.4/2.5) plus stage2 (9.x/11.x). This port ships:

Stage1 (ported earlier):
  * §1.1 data sufficiency, §2.3 direction, §2.4 Always-In.

Stage2 (ported in this revision — mirrors upstream apply_stage2):
  * §9.1 signal-bar-closed, §9.2 signal-bar-direction, §9.3 signal-bar-length,
    §9.5 follow-through judges.
  * §11 order-method routing (``route_order_method``) keyed by cycle_position.
  * ``apply_overrides`` (OverrideArbiter) with conservativeness-rank safety
    gates + §2.3/§2.4/§11 field-sync helpers.
  * ``_merge_program_nodes`` / ``_merge_program_nodes_head`` with the
    ``AI_PRIMARY_NODES`` (§1.3/§2.5) preservation distinction.
  * ``DecisionNodeEngine.apply_stage2`` orchestrator, wired into
    ``normalize_trade_decision``.
  * ``KlineGeometryFeature`` adapter: maps ``feature_rows`` dicts to the
    geometry dataclass the §9 judges consume.

Not ported (future issues):
  * §1.3 ``judge_market_chaos`` / §2.5 ``judge_momentum_strength`` — the
    AI-primary judges behind ``AI_PRIMARY_NODES``; freqtrade lets the LLM fill
    these today. The merge logic already preserves AI nodes for these ids.

The frame adapter (``_Frame`` / ``_frame_from_feature_rows``) bridges
freqtrade's ``feature_rows: list[dict]`` (with ``k``/``open``/``high``/
``low``/``close``/``ema20``/``atr14`` keys) to the bar/indicators shape
expected by the stage1 functions. Stage2 judges read pre-computed geometry
(``bar_type``/``range_atr``/``follow_through_1_2``) directly from the
``feature_rows`` dicts via ``compute_kline_geometry_features``.
"""
# ruff: noqa: RUF001 C901
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from .decision_tree import canonical_tree_questions


logger = logging.getLogger(__name__)


# ── Coercion helpers (mirror upstream decision_nodes.py:35-44) ──────────────


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Return *value* when it is a dict; otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def _coerce_trace_list(trace: Any) -> list[dict[str, Any]]:
    """Keep only dict trace nodes; tolerate non-list or string elements from AI JSON."""
    if not isinstance(trace, list):
        return []
    return [item for item in trace if isinstance(item, dict)]


# ── Threshold constants (mirror upstream decision_nodes.py:51-81) ─────────────

DIRECTION_WINDOW: int = 8              # §2.3 short voting window
DIRECTION_WINDOW_MED: int = 20         # §2.3 medium confirmation window
DIRECTION_STRONG_SHORT_SCORE: int = 4  # |score|≥此值时忽略中窗口冲突
BAR_COUNT_THRESHOLD: int = 20          # §1.1 data sufficiency threshold

ALWAYS_IN_NEAR_WINDOW: int = 8         # §2.4 近端主判窗口
ALWAYS_IN_WINDOW: int = 20             # §2.4 背景参考窗口
ALWAYS_IN_NEAR_SAME_SIDE_RATIO: float = 0.65
ALWAYS_IN_SAME_SIDE_RATIO: float = 0.70
ALWAYS_IN_PULLBACK_ATR_RATIO: float = 1.5

EMA_SLOPE_LOOKBACK: int = 10

DIRECTION_BULL_THRESHOLD: int = 3
DIRECTION_BEAR_THRESHOLD: int = -3
TREND_BAR_DOMINANCE_RATIO: float = 1.5
OVERLAP_LOW_THRESHOLD: float = 0.45
OVERLAP_HIGH_THRESHOLD: float = 0.65

# ── §9 stage2 thresholds (mirror upstream decision_nodes.py:64) ─────────────
SIGNAL_BAR_LONG_ATR_RATIO: float = 2.0  # §9.3 overlong threshold

# Valid trace answers (mirror upstream decision_tree.py:27 TRACE_ANSWERS).
TRACE_ANSWERS: frozenset[str] = frozenset({"是", "否", "中性", "等待", "不适用"})

# ── Override permission sets (mirror upstream decision_nodes.py:85-119) ─────
LOCKED_NODES: frozenset[str] = frozenset({"1.1", "9.1"})
OVERRIDABLE_NODES: frozenset[str] = frozenset(
    {"1.3", "2.3", "2.4", "2.5", "9.2", "9.3", "11.1", "11.2", "11.3", "11.4"}
)
# Nodes where the AI is the primary judge; program does not replace AI when AI
# wrote the node. The program node is used only when the AI omitted it entirely.
AI_PRIMARY_NODES: frozenset[str] = frozenset({"1.3", "2.5"})
# AI-primary nodes that receive appended program metrics in reason (none yet).
AI_PRIMARY_SUPPLEMENT_NODES: frozenset[str] = frozenset()
SAFETY_GATE_NODES: frozenset[str] = frozenset({"1.1", "10.3", "14"})

# §1.3 extreme chaos thresholds
CHAOS_OVERLAP_THRESHOLD: float = 0.70
CHAOS_EMA_FLAT_ATR_RATIO: float = 0.05
CHAOS_DIRECTION_SCORE_MAX: int = 1

# §2.5 momentum strength thresholds
MOMENTUM_OVERLAP_WEAK: float = 0.50
MOMENTUM_TREND_RATIO_STRONG: float = 1.5
MOMENTUM_PULLBACK_DEEP_ATR: float = 3.0
MOMENTUM_TREND_BAR_MIN_RATIO: float = 0.50


# ── cycle_position → candidate order method (mirror upstream:1789-1811) ──────
_CYCLE_ORDER_METHOD: dict[str, str] = {
    "spike": "市价单",
    "micro_channel": "突破单",
    "tight_channel": "突破单",
    "normal_channel": "突破单",
    "broad_channel": "限价单",
    "trading_range": "限价单",
    "trending_tr": "突破单",
    "extreme_tr": "不下单",
    "unknown": "不下单",
}


# ── NodeFill dataclass (mirror upstream NodeFill) ────────────────────────────


@dataclass(frozen=True)
class NodeFill:
    """Intermediate representation of a program-filled trace node."""

    node_id: str
    answer: str        # ∈ {是, 否, 中性, 等待, 不适用}
    reason: str
    bar_range: str     # like "K8-K1"
    branch: str | None = None
    section: str | None = None


# ── Frame adapter: feature_rows (list[dict]) → frame-like object ─────────────


_K_SEQ_RE = re.compile(r"K\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class _Bar:
    seq: int           # 1=newest
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class _Indicators:
    ema20: tuple[float, ...]
    atr14: tuple[float, ...]


@dataclass(frozen=True)
class _Frame:
    bars: tuple[_Bar, ...]
    indicators: _Indicators


_MIN_BARS_FOR_ENGINE = 5


def _frame_from_feature_rows(
    feature_rows: list[dict[str, Any]] | None,
) -> _Frame | None:
    """Build a frame from freqtrade feature_rows.

    Returns ``None`` when there are not enough valid bars
    (engine needs >=5 to compute swings).
    """
    if not feature_rows:
        return None
    bars: list[_Bar] = []
    ema20: list[float] = []
    atr14: list[float] = []
    for row in feature_rows:
        m = _K_SEQ_RE.search(str(row.get("k", "")))
        if not m:
            continue
        try:
            seq = int(m.group(1))
            bars.append(
                _Bar(
                    seq=seq,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
        try:
            ema20.append(float(row.get("ema20")))
        except (TypeError, ValueError):
            ema20.append(float("nan"))
        try:
            atr14.append(float(row.get("atr14")))
        except (TypeError, ValueError):
            atr14.append(float("nan"))
    if len(bars) < _MIN_BARS_FOR_ENGINE:
        return None
    bars.sort(key=lambda b: b.seq)  # ascending seq => bars[0]=K1=newest
    return _Frame(
        bars=tuple(bars),
        indicators=_Indicators(tuple(ema20), tuple(atr14)),
    )


# ── KlineGeometryFeature adapter (mirror upstream kline_features.py) ────────
# Upstream computes geometry from KlineFrame/KlineBar (attribute access). In
# freqtrade the feature_rows dicts already carry these values (computed by
# ``features.py``), so this adapter maps the dict keys onto the frozen dataclass
# the §9 judges consume (``features[sig].bar_type`` / ``.range_atr_ratio`` /
# ``.follow_through_1_2``). Dict key names differ slightly from upstream
# (``range_atr`` vs ``range_atr_ratio``, ``ema20_relation`` vs ``ema_relation``).


@dataclass(frozen=True)
class KlineGeometryFeature:
    """Single-bar geometry, newest bar keeps its original ``seq``."""

    seq: int
    bar_type: str
    body_ratio: float | None
    upper_wick_ratio: float | None
    lower_wick_ratio: float | None
    close_position: float | None
    range_atr_ratio: float | None
    ema_relation: str
    overlap_prev_ratio: float | None
    inside_sequence: str
    ioi_pattern: bool
    micro_double: str
    gap_bar: str
    ema_gap_count: int
    breakout_prev: str
    follow_through_1_2: str


def _seq_from_k(value: Any) -> int | None:
    """Parse the seq number from a ``k`` field like ``K1``/``K 12``."""
    m = _K_SEQ_RE.search(str(value or ""))
    return int(m.group(1)) if m else None


def _finite_float(value: Any) -> float | None:
    """Coerce to a finite float, else None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def compute_kline_geometry_features(
    feature_rows: list[dict[str, Any]] | None,
) -> dict[int, KlineGeometryFeature]:
    """Build a ``{seq: KlineGeometryFeature}`` map from freqtrade feature_rows.

    Upstream returns ``list[KlineGeometryFeature]`` and the stage2 engine indexes
    by ``features[sig]``. Here we return a dict keyed by seq for direct lookup.
    Rows missing a parseable ``k`` seq are skipped.
    """
    result: dict[int, KlineGeometryFeature] = {}
    if not feature_rows:
        return result
    for row in feature_rows:
        if not isinstance(row, dict):
            continue
        seq = _seq_from_k(row.get("k"))
        if seq is None:
            continue
        result[seq] = KlineGeometryFeature(
            seq=seq,
            bar_type=str(row.get("bar_type", "other")),
            body_ratio=_finite_float(row.get("body_ratio")),
            upper_wick_ratio=_finite_float(row.get("upper_wick_pct")),
            lower_wick_ratio=_finite_float(row.get("lower_wick_pct")),
            close_position=_finite_float(row.get("close_position")),
            range_atr_ratio=_finite_float(row.get("range_atr")),
            ema_relation=str(row.get("ema20_relation", "unknown")),
            overlap_prev_ratio=_finite_float(row.get("overlap_prev_ratio")),
            inside_sequence=str(row.get("inside_sequence", "none")),
            ioi_pattern=bool(row.get("ioi_pattern", False)),
            micro_double=str(row.get("micro_double", "none")),
            gap_bar=str(row.get("gap_bar", "none")),
            ema_gap_count=int(row.get("ema_gap_count", 0) or 0),
            breakout_prev=str(row.get("breakout_prev", "none")),
            follow_through_1_2=str(row.get("follow_through_1_2", "pending")),
        )
    return result


# ── Helper functions (verbatim from upstream) ────────────────────────────────


def judge_data_sufficiency(frame: Any) -> NodeFill:
    """Fill §1.1=是 (data already sufficient, PreflightDataGate already passed).

    Ported from ``PA_Agent/pa_agent/ai/decision_nodes.py``.
    """
    bars = getattr(frame, "bars", ()) or ()
    try:
        n = max(int(getattr(b, "seq", 0)) for b in bars)
    except (TypeError, ValueError):
        n = len(bars)
    return NodeFill(
        node_id="1.1",
        answer="是",
        reason=(
            f"已收盘K线 {n} 根 >= {BAR_COUNT_THRESHOLD} 根阈值"
            "（已通过前置数据闸门），数据量满足分析要求。"
        ),
        bar_range=f"K{n}-K1",
    )


def _count_trend_bars(bars: Any, W: int) -> tuple[int, int]:
    """Count bull-trend and bear-trend bars in the first W bars.

    A bull-trend bar: close > open AND close_position >= 0.65 AND body_ratio > 0.25.
    A bear-trend bar: close < open AND close_position <= 0.35 AND body_ratio > 0.25.
    """
    bull = 0
    bear = 0
    for bar in list(bars)[:W]:
        try:
            high = max(float(bar.high), float(bar.low))
            low = min(float(bar.high), float(bar.low))
            open_ = float(bar.open)
            close = float(bar.close)
            full_range = high - low
            if full_range <= 0:
                continue
            body = abs(close - open_)
            body_ratio = body / full_range
            close_pos = max(0.0, min(1.0, (close - low) / full_range))
            if body_ratio <= 0.25:
                continue  # doji — not a trend bar
            if close > open_ and close_pos >= 0.65:
                bull += 1
            elif close < open_ and close_pos <= 0.35:
                bear += 1
        except (TypeError, ValueError, AttributeError):
            continue
    return bull, bear


def _mean_overlap_ratio(bars: Any, W: int) -> float | None:
    """Mean IoU-style overlap between adjacent bar pairs in window."""
    window = list(bars)[:W]
    ratios: list[float] = []
    for i in range(len(window) - 1):
        try:
            cur = window[i]
            prv = window[i + 1]
            cur_h = max(float(cur.high), float(cur.low))
            cur_l = min(float(cur.high), float(cur.low))
            prv_h = max(float(prv.high), float(prv.low))
            prv_l = min(float(prv.high), float(prv.low))
            overlap = max(0.0, min(cur_h, prv_h) - max(cur_l, prv_l))
            union = max(cur_h, prv_h) - min(cur_l, prv_l)
            if union > 0:
                ratios.append(overlap / union)
        except (TypeError, ValueError, AttributeError):
            continue
    if len(ratios) < 2:
        return None
    return sum(ratios) / len(ratios)


def _find_swings(bars: Any, W: int) -> tuple[list[float], list[float]]:
    """Find swing highs and lows using left/right 2-bar pivot detection."""
    window = list(bars[:W])
    if len(window) < 5:
        return [], []
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(2, len(window) - 2):
        h = float(window[i].high)
        if (
            float(window[i - 1].high) < h
            and float(window[i - 2].high) < h
            and float(window[i + 1].high) < h
            and float(window[i + 2].high) < h
        ):
            swing_highs.append(h)
        lo = float(window[i].low)
        if (
            float(window[i - 1].low) > lo
            and float(window[i - 2].low) > lo
            and float(window[i + 1].low) > lo
            and float(window[i + 2].low) > lo
        ):
            swing_lows.append(lo)
    return swing_highs, swing_lows


def _weighted_ema_side_weights(
    bars: Any, N: int, ema20: tuple,
) -> tuple[float, float]:
    """Linear-decay weighted counts of closes above/below EMA in first N bars."""
    w_above = 0.0
    w_below = 0.0
    for i, bar in enumerate(list(bars)[:N]):
        if i >= len(ema20):
            break
        try:
            ema_val = float(ema20[i])
            close_val = float(bar.close)
        except (TypeError, ValueError, AttributeError):
            continue
        if math.isnan(ema_val):
            continue
        weight = float(N - i)
        if close_val > ema_val:
            w_above += weight
        elif close_val < ema_val:
            w_below += weight
    return w_above, w_below


def _max_pullback_atr(bars: Any, N: int, ema20: tuple, atr14: tuple) -> float | None:
    """Max close-to-close range in window normalized by ATR."""
    try:
        if not atr14 or math.isnan(float(atr14[0])) or float(atr14[0]) <= 0:
            return None
        atr_val = float(atr14[0])
        closes = []
        for bar in list(bars)[:N]:
            try:
                closes.append(float(bar.close))
            except (TypeError, ValueError, AttributeError):
                pass
        if len(closes) < 2:
            return None
        max_range = max(closes) - min(closes)
        return max_range / atr_val
    except (TypeError, ValueError):
        return None


# ── §2.3 direction judge (mirror upstream judge_direction) ───────────────────


def judge_direction(frame: Any) -> tuple[str, NodeFill]:
    """Five-signal vote to determine direction and fill §2.3 node.

    Signals (each contributes -1, 0, or +1 to the score):
      S1: EMA slope (10-bar lookback, ATR dead-zone filter)
      S2: Closing center of gravity (short window, near half vs far half)
      S3: Swing structure HH+HL vs LL+LH (2-bar pivot detection)
      S4: Trend-bar dominance (bull vs bear trend-bar count ratio)
      S5: K-line overlap ratio (low overlap → trending)

    Medium-window confirmation reduces |score| by 1 when it contradicts
    the short-window result.

    Returns (direction, NodeFill) where direction ∈ {bullish, bearish, neutral}.
    """
    bars = getattr(frame, "bars", ()) or ()
    indicators = getattr(frame, "indicators", None)
    ema20 = tuple(getattr(indicators, "ema20", ()) or ())
    atr14 = tuple(getattr(indicators, "atr14", ()) or ())

    n = 0
    try:
        n = max(int(getattr(b, "seq", 0)) for b in bars)
    except (TypeError, ValueError):
        n = len(bars)

    W = min(DIRECTION_WINDOW, n)
    W_med = min(DIRECTION_WINDOW_MED, n)

    close_prices: list[float] = []
    for bar in list(bars)[:W]:
        try:
            close_prices.append(float(bar.close))
        except (TypeError, ValueError, AttributeError):
            close_prices.append(float("nan"))

    # ── Signal 1: EMA slope ──
    s1 = 0
    s1_desc = "EMA斜率:0"
    try:
        if ema20 and len(ema20) >= 1 and not math.isnan(float(ema20[0])):
            k = min(EMA_SLOPE_LOOKBACK, n - 1)
            if k >= 1 and len(ema20) > k and not math.isnan(float(ema20[k])):
                d = float(ema20[0]) - float(ema20[k])
                thr = 0.0
                if atr14 and len(atr14) >= 1 and not math.isnan(float(atr14[0])):
                    thr = 0.05 * float(atr14[0])
                if d > thr:
                    s1 = 1
                    s1_desc = f"EMA斜率:+1(d={d:.4f}>thr={thr:.4f})"
                elif d < -thr:
                    s1 = -1
                    s1_desc = f"EMA斜率:-1(d={d:.4f}<-thr={-thr:.4f})"
                else:
                    s1_desc = f"EMA斜率:0(d={d:.4f},死区±{thr:.4f})"
    except (TypeError, ValueError):
        pass

    # ── Signal 2: Weighted closing center of gravity (short window) ──
    s2 = 0
    s2_desc = "收盘重心:0"
    try:
        h = W // 2
        if h >= 1 and len(close_prices) >= 2 * h:
            def _weighted_avg(vals: list[float], start_idx: int) -> float:
                total_w = 0.0
                total_wv = 0.0
                for local_i, v in enumerate(vals):
                    if math.isnan(v):
                        continue
                    w = W - (start_idx + local_i)
                    total_w += w
                    total_wv += w * v
                return total_wv / total_w if total_w > 0 else float("nan")

            near_vals = close_prices[:h]
            far_vals = close_prices[h:2 * h]
            near = _weighted_avg(near_vals, 0)
            far = _weighted_avg(far_vals, h)
            if not math.isnan(near) and not math.isnan(far):
                diff = near - far
                thr2 = 0.0
                if atr14 and len(atr14) >= 1 and not math.isnan(float(atr14[0])):
                    thr2 = 0.1 * float(atr14[0])
                if diff > thr2:
                    s2 = 1
                    s2_desc = f"收盘重心(加权):+1(diff={diff:.4f}>thr={thr2:.4f})"
                elif diff < -thr2:
                    s2 = -1
                    s2_desc = f"收盘重心(加权):-1(diff={diff:.4f}<-thr={-thr2:.4f})"
                else:
                    s2_desc = f"收盘重心(加权):0(diff={diff:.4f},死区±{thr2:.4f})"
    except (TypeError, ValueError):
        pass

    # ── Signal 3: Swing structure HH/HL vs LL/LH ──
    s3 = 0
    s3_desc = "波段结构:0"
    try:
        swing_highs, swing_lows = _find_swings(bars, W)
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[0] > swing_highs[1]
            hl = swing_lows[0] > swing_lows[1]
            ll = swing_lows[0] < swing_lows[1]
            lh = swing_highs[0] < swing_highs[1]
            if hh and hl:
                s3 = 1
                s3_desc = "波段结构:+1(HH+HL)"
            elif ll and lh:
                s3 = -1
                s3_desc = "波段结构:-1(LL+LH)"
            else:
                s3_desc = f"波段结构:0(HH={hh},HL={hl},LL={ll},LH={lh})"
        else:
            s3_desc = (
                f"波段结构:0(枢轴不足,highs={len(swing_highs)},lows={len(swing_lows)})"
            )
    except (TypeError, ValueError, IndexError):
        pass

    # ── Signal 4: Trend-bar dominance ──
    s4 = 0
    s4_desc = "趋势棒占比:0"
    try:
        bull_tb, bear_tb = _count_trend_bars(bars, W)
        if bull_tb + bear_tb > 0:
            if bull_tb > 0 and bear_tb == 0:
                s4 = 1
                s4_desc = f"趋势棒占比:+1(多头趋势棒{bull_tb}根,空头0根)"
            elif bear_tb > 0 and bull_tb == 0:
                s4 = -1
                s4_desc = f"趋势棒占比:-1(空头趋势棒{bear_tb}根,多头0根)"
            elif bull_tb >= bear_tb * TREND_BAR_DOMINANCE_RATIO:
                s4 = 1
                s4_desc = (
                    f"趋势棒占比:+1(多{bull_tb}/空{bear_tb}"
                    f"≥{TREND_BAR_DOMINANCE_RATIO:.1f}×)"
                )
            elif bear_tb >= bull_tb * TREND_BAR_DOMINANCE_RATIO:
                s4 = -1
                s4_desc = (
                    f"趋势棒占比:-1(空{bear_tb}/多{bull_tb}"
                    f"≥{TREND_BAR_DOMINANCE_RATIO:.1f}×)"
                )
            else:
                s4_desc = f"趋势棒占比:0(多{bull_tb}/空{bear_tb},无明显优势)"
        else:
            s4_desc = "趋势棒占比:0(窗口内无趋势棒)"
    except (TypeError, ValueError):
        pass

    # ── Signal 5: K-line overlap ratio ──
    s5 = 0
    s5_desc = "K线重叠:0"
    try:
        mean_overlap = _mean_overlap_ratio(bars, W)
        if mean_overlap is not None:
            if mean_overlap < OVERLAP_LOW_THRESHOLD:
                if s1 > 0:
                    s5 = 1
                    s5_desc = (
                        f"K线重叠:+1(均值重叠{mean_overlap:.3f}<{OVERLAP_LOW_THRESHOLD},"
                        "低重叠强化多头方向)"
                    )
                elif s1 < 0:
                    s5 = -1
                    s5_desc = (
                        f"K线重叠:-1(均值重叠{mean_overlap:.3f}<{OVERLAP_LOW_THRESHOLD},"
                        "低重叠强化空头方向)"
                    )
                else:
                    s5_desc = (
                        f"K线重叠:0(均值重叠{mean_overlap:.3f}<{OVERLAP_LOW_THRESHOLD},"
                        "EMA斜率中性,重叠信号不明)"
                    )
            elif mean_overlap > OVERLAP_HIGH_THRESHOLD:
                s5_desc = (
                    f"K线重叠:0(均值重叠{mean_overlap:.3f}>{OVERLAP_HIGH_THRESHOLD},"
                    "高重叠→区间,无方向贡献)"
                )
            else:
                s5_desc = f"K线重叠:0(均值重叠{mean_overlap:.3f},中等重叠)"
    except (TypeError, ValueError):
        pass

    score = s1 + s2 + s3 + s4 + s5

    # ── Medium-window confirmation filter ──
    med_confirm = 0
    med_confirm_desc = "中窗口重心:0"
    try:
        close_prices_med: list[float] = []
        for bar in list(bars)[:W_med]:
            try:
                close_prices_med.append(float(bar.close))
            except (TypeError, ValueError, AttributeError):
                close_prices_med.append(float("nan"))
        hm = W_med // 2
        if hm >= 1 and len(close_prices_med) >= 2 * hm:
            def _weighted_avg_med(vals: list[float], start_idx: int) -> float:
                total_w = 0.0
                total_wv = 0.0
                for local_i, v in enumerate(vals):
                    if math.isnan(v):
                        continue
                    w = W_med - (start_idx + local_i)
                    total_w += w
                    total_wv += w * v
                return total_wv / total_w if total_w > 0 else float("nan")

            near_m_vals = close_prices_med[:hm]
            far_m_vals = close_prices_med[hm:2 * hm]
            near_m = _weighted_avg_med(near_m_vals, 0)
            far_m = _weighted_avg_med(far_m_vals, hm)
            if not math.isnan(near_m) and not math.isnan(far_m):
                diff_m = near_m - far_m
                thr_m = 0.0
                if atr14 and len(atr14) >= 1 and not math.isnan(float(atr14[0])):
                    thr_m = 0.1 * float(atr14[0])
                if diff_m > thr_m:
                    med_confirm = 1
                    med_confirm_desc = (
                        f"中窗口重心(加权):+1(diff={diff_m:.4f}>thr={thr_m:.4f},W={W_med})"
                    )
                elif diff_m < -thr_m:
                    med_confirm = -1
                    med_confirm_desc = (
                        f"中窗口重心(加权):-1(diff={diff_m:.4f}<-thr={-thr_m:.4f},W={W_med})"
                    )
                else:
                    med_confirm_desc = f"中窗口重心(加权):0(diff={diff_m:.4f},W={W_med})"
    except (TypeError, ValueError):
        pass

    if med_confirm != 0 and score != 0 and med_confirm != (1 if score > 0 else -1):
        if abs(score) >= DIRECTION_STRONG_SHORT_SCORE:
            med_confirm_desc += (
                f"（背景窗口与短窗口冲突，但|score|={abs(score)}"
                f"≥{DIRECTION_STRONG_SHORT_SCORE}，新趋势优先，不扣分）"
            )
        else:
            score_before = score
            score = score - (1 if score > 0 else -1)
            med_confirm_desc += f"（与短窗口冲突，score {score_before}→{score}）"
    else:
        if med_confirm != 0:
            med_confirm_desc += "（与短窗口一致）"

    if score >= DIRECTION_BULL_THRESHOLD:
        direction = "bullish"
        answer = "是"
        branch = "bullish"
    elif score <= DIRECTION_BEAR_THRESHOLD:
        direction = "bearish"
        answer = "是"
        branch = "bearish"
    else:
        direction = "neutral"
        answer = "中性"
        branch = "neutral"

    bar_range = f"K{W}-K1"

    reason = (
        f"五信号投票（阈值±{DIRECTION_BULL_THRESHOLD}）："
        f"{s1_desc}；{s2_desc}；{s3_desc}；{s4_desc}；{s5_desc}。"
        f"{med_confirm_desc}。"
        f"综合score={score}（≥+{DIRECTION_BULL_THRESHOLD}→多头，"
        f"≤{DIRECTION_BEAR_THRESHOLD}→空头，否则中性）→{direction}。"
    )

    fill = NodeFill(
        node_id="2.3",
        answer=answer,
        reason=reason,
        bar_range=bar_range,
        branch=branch,
    )
    return direction, fill


# ── §2.4 Always-In judge (mirror upstream judge_always_in) ───────────────────


def _eval_always_in_gates(
    bars: Any,
    N: int,
    ema20: tuple,
    atr14: tuple,
    n: int,
    *,
    slope_lookback: int,
    same_side_ratio: float,
) -> dict[str, Any]:
    """Evaluate AIL/AIS gate bundle for a window of N bars (index 0 = newest)."""
    w_above, w_below = _weighted_ema_side_weights(bars, N, ema20)
    valid_w = w_above + w_below
    if valid_w <= 0:
        above_ratio = below_ratio = 0.0
    else:
        above_ratio = w_above / valid_w
        below_ratio = w_below / valid_w

    slope_sign = 0
    slope_desc = "EMA斜率:0"
    try:
        if ema20 and len(ema20) >= 1 and not math.isnan(float(ema20[0])):
            k = min(slope_lookback, n - 1)
            if k >= 1 and len(ema20) > k and not math.isnan(float(ema20[k])):
                d = float(ema20[0]) - float(ema20[k])
                thr = 0.0
                if atr14 and len(atr14) >= 1 and not math.isnan(float(atr14[0])):
                    thr = 0.05 * float(atr14[0])
                if d > thr:
                    slope_sign = 1
                    slope_desc = f"EMA斜率向上(d={d:.4f}>thr={thr:.4f})"
                elif d < -thr:
                    slope_sign = -1
                    slope_desc = f"EMA斜率向下(d={d:.4f}<-thr={thr:.4f})"
                else:
                    slope_desc = f"EMA斜率平坦(d={d:.4f},死区±{thr:.4f})"
    except (TypeError, ValueError):
        pass

    swing_confirms_bull = False
    swing_confirms_bear = False
    swing_desc = "波段结构:未验证"
    try:
        swing_highs, swing_lows = _find_swings(bars, N)
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[0] > swing_highs[1]
            hl = swing_lows[0] > swing_lows[1]
            ll = swing_lows[0] < swing_lows[1]
            lh = swing_highs[0] < swing_highs[1]
            if hh and hl:
                swing_confirms_bull = True
                swing_desc = "波段结构HH+HL✓(多头)"
            elif ll and lh:
                swing_confirms_bear = True
                swing_desc = "波段结构LL+LH✓(空头)"
            else:
                swing_desc = f"波段结构混乱(HH={hh},HL={hl},LL={ll},LH={lh})"
        else:
            swing_desc = (
                f"波段结构:枢轴不足(highs={len(swing_highs)},lows={len(swing_lows)})"
            )
    except (TypeError, ValueError, IndexError):
        pass

    pullback_atr = _max_pullback_atr(bars, N, ema20, atr14)
    shallow: bool | None = None
    pullback_desc = "回撤:未知(ATR缺失)"
    if pullback_atr is not None:
        shallow = pullback_atr <= ALWAYS_IN_PULLBACK_ATR_RATIO
        pullback_desc = (
            f"最大价格区间{pullback_atr:.2f}×ATR"
            f"({'≤' if shallow else '>'}{ALWAYS_IN_PULLBACK_ATR_RATIO}×阈值,"
            f"{'浅回撤✓' if shallow else '回撤较深✗'})"
        )

    bull_core = above_ratio >= same_side_ratio and slope_sign > 0
    bear_core = below_ratio >= same_side_ratio and slope_sign < 0
    gate3_bull = swing_confirms_bull and (shallow is None or shallow)
    gate3_bear = swing_confirms_bear and (shallow is None or shallow)

    return {
        "N": N,
        "above_ratio": above_ratio,
        "below_ratio": below_ratio,
        "slope_sign": slope_sign,
        "slope_desc": slope_desc,
        "swing_desc": swing_desc,
        "pullback_desc": pullback_desc,
        "bull_core": bull_core,
        "bear_core": bear_core,
        "gate3_bull": gate3_bull,
        "gate3_bear": gate3_bear,
    }


def judge_always_in(frame: Any) -> NodeFill:
    """Judge Always In state (§2.4) with dual-window Brooks alignment.

    Near window (K8-K1) is authoritative — captures current inertia.
    Background window (K20-K1) is reference only — does not veto near conclusion.
    """
    bars = getattr(frame, "bars", ()) or ()
    indicators = getattr(frame, "indicators", None)
    ema20 = tuple(getattr(indicators, "ema20", ()) or ())
    atr14 = tuple(getattr(indicators, "atr14", ()) or ())

    n = 0
    try:
        n = max(int(getattr(b, "seq", 0)) for b in bars)
    except (TypeError, ValueError):
        n = len(bars)

    N_near = min(ALWAYS_IN_NEAR_WINDOW, n)
    N_bg = min(ALWAYS_IN_WINDOW, n)

    near = _eval_always_in_gates(
        bars, N_near, ema20, atr14, n,
        slope_lookback=min(5, n - 1),
        same_side_ratio=ALWAYS_IN_NEAR_SAME_SIDE_RATIO,
    )
    bg = _eval_always_in_gates(
        bars, N_bg, ema20, atr14, n,
        slope_lookback=EMA_SLOPE_LOOKBACK,
        same_side_ratio=ALWAYS_IN_SAME_SIDE_RATIO,
    )

    bar_range = f"K{N_near}-K1"
    conflict_note = ""

    if near["bull_core"]:
        answer = "是"
        branch = "AIL"
        strength = "（结构确认，强AIL）" if near["gate3_bull"] else "（结构弱/回撤深，弱AIL）"
        if bg["bear_core"]:
            conflict_note = (
                f" ⚠️ 近端K{N_near}-K1已切换多头惯性（加权同侧{near['above_ratio']:.0%}），"
                f"背景K{N_bg}-K1仍偏空（加权同侧{bg['below_ratio']:.0%}）——"
                "按Brooks并列原则：近端AIL为交易主方向，背景AIS仅作上方阻力风险提示，不否决做多。"
            )
        reason = (
            f"【近端主判K{N_near}-K1】加权收盘高于EMA占比{near['above_ratio']:.1%}"
            f"≥{ALWAYS_IN_NEAR_SAME_SIDE_RATIO:.0%}；{near['slope_desc']}；"
            f"{near['swing_desc']}；{near['pullback_desc']}。"
            f"判定为Always In Long（AIL）{strength}。"
            f"【背景参考K{N_bg}-K1】加权多侧{bg['above_ratio']:.1%}/空侧{bg['below_ratio']:.1%}；"
            f"{bg['slope_desc']}。"
            f"{conflict_note}"
        )
    elif near["bear_core"]:
        answer = "是"
        branch = "AIS"
        strength = "（结构确认，强AIS）" if near["gate3_bear"] else "（结构弱/回撤深，弱AIS）"
        if bg["bull_core"]:
            conflict_note = (
                f" ⚠️ 近端K{N_near}-K1已切换空头惯性（加权同侧{near['below_ratio']:.0%}），"
                f"背景K{N_bg}-K1仍偏多（加权同侧{bg['above_ratio']:.0%}）——"
                "按Brooks并列原则：近端AIS为交易主方向，背景AIL仅作下方支撑风险提示，不否决做空。"
            )
        reason = (
            f"【近端主判K{N_near}-K1】加权收盘低于EMA占比{near['below_ratio']:.1%}"
            f"≥{ALWAYS_IN_NEAR_SAME_SIDE_RATIO:.0%}；{near['slope_desc']}；"
            f"{near['swing_desc']}；{near['pullback_desc']}。"
            f"判定为Always In Short（AIS）{strength}。"
            f"【背景参考K{N_bg}-K1】加权多侧{bg['above_ratio']:.1%}/空侧{bg['below_ratio']:.1%}；"
            f"{bg['slope_desc']}。"
            f"{conflict_note}"
        )
    elif bg["bull_core"]:
        answer = "是"
        branch = "AIL"
        strength = "（仅背景确认，近端未共振，弱AIL）"
        reason = (
            f"【近端K{N_near}-K1】未达AIL阈值（多侧{near['above_ratio']:.1%}，{near['slope_desc']}）。"
            f"【背景K{N_bg}-K1】仍满足AIL（多侧{bg['above_ratio']:.1%}，{bg['slope_desc']}）"
            f"→弱AIL，优先等待近端结构确认。"
            f"{strength}"
        )
    elif bg["bear_core"]:
        answer = "是"
        branch = "AIS"
        strength = "（仅背景确认，近端未共振，弱AIS）"
        reason = (
            f"【近端K{N_near}-K1】未达AIS阈值（空侧{near['below_ratio']:.1%}，{near['slope_desc']}）。"
            f"【背景K{N_bg}-K1】仍满足AIS（空侧{bg['below_ratio']:.1%}，{bg['slope_desc']}）"
            f"→弱AIS，优先等待近端结构确认。"
            f"{strength}"
        )
    else:
        answer = "否"
        branch = None
        reason = (
            f"【近端K{N_near}-K1】多侧{near['above_ratio']:.1%}/空侧{near['below_ratio']:.1%}；"
            f"{near['slope_desc']}；{near['swing_desc']}。"
            f"【背景K{N_bg}-K1】多侧{bg['above_ratio']:.1%}/空侧{bg['below_ratio']:.1%}；"
            f"{bg['slope_desc']}。"
            "近端与背景均未达Always In阈值。"
        )

    return NodeFill(
        node_id="2.4",
        answer=answer,
        reason=reason,
        bar_range=bar_range,
        branch=branch,
    )


# ── Node merge / insertion ───────────────────────────────────────────────────


def _node_id_sort_key(node_id: str) -> tuple[int, int, str]:
    """Numeric sort key for gate_trace node_id values ('1.1' -> (1, 1, '1.1'))."""
    parts = str(node_id or "").split(".", 1)
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        return (999, 999, node_id)
    if len(parts) == 1:
        return (major, 0, node_id)
    sub = parts[1]
    try:
        return (major, int(sub), node_id)
    except ValueError:
        return (major, 999, node_id)


def _build_program_trace_node(fill: NodeFill) -> dict[str, Any]:
    """Convert a NodeFill to a trace dict (question from decision tree)."""
    questions = canonical_tree_questions()
    question = questions.get(fill.node_id, fill.node_id)
    node: dict[str, Any] = {
        "node_id": fill.node_id,
        "question": question,
        "answer": fill.answer,
        "reason": fill.reason,
        "bar_range": fill.bar_range,
        "skipped": False,
    }
    if fill.branch:
        node["branch"] = fill.branch
    if fill.section:
        node["section"] = fill.section
    return node


def _merge_program_nodes(
    trace: list[dict[str, Any]],
    program_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge program nodes into trace by node_id.

    Two merge modes based on node type (mirror upstream ``merge_program_nodes``):

    PROGRAM-AUTHORITATIVE (default):
      Program result replaces the AI node entirely. Used for §1.1, §2.3, §2.4
      where the program has definitive computed data.

    AI-PRIMARY (AI_PRIMARY_NODES — §1.3 and §2.5):
      If the AI already wrote the node, preserve the AI version. For
      AI_PRIMARY_SUPPLEMENT_NODES the program reason is appended as a
      reference note (none currently).

    New program nodes not already in the AI trace are inserted in chapter-section
    order (1.1 < 1.2 < 2.3 < 2.5).
    """
    result = [item for item in trace if isinstance(item, dict)]
    prog_by_id = {
        n["node_id"]: n for n in program_nodes
        if isinstance(n, dict) and "node_id" in n
    }
    replaced_ids: set[str] = set()
    for i, item in enumerate(result):
        nid = str(item.get("node_id", "")).strip()
        if nid not in prog_by_id:
            continue
        if nid in AI_PRIMARY_NODES:
            if nid in AI_PRIMARY_SUPPLEMENT_NODES:
                # AI-primary + program supplement in reason
                prog_node = prog_by_id[nid]
                prog_reason = str(prog_node.get("reason", "") or "").strip()
                prog_bar_range = str(prog_node.get("bar_range", "") or "").strip()
                if prog_reason:
                    ai_reason = str(item.get("reason", "") or "").strip()
                    supplement = f"【程序参考数据（{prog_bar_range}）：{prog_reason}】"
                    if supplement not in ai_reason:
                        result[i] = dict(item)
                        result[i]["reason"] = f"{ai_reason} {supplement}".strip()
            # §2.5: keep AI node as-is; program metrics not appended to reason.
        else:
            # Program-authoritative: program result replaces AI node
            result[i] = prog_by_id[nid]
        replaced_ids.add(nid)
    new_nodes = [
        node for nid, node in prog_by_id.items() if nid not in replaced_ids
    ]
    if new_nodes:
        result.extend(new_nodes)
        result.sort(key=lambda x: _node_id_sort_key(str(x.get("node_id", ""))))
    return result


def _merge_program_nodes_head(
    trace: list[dict[str, Any]],
    program_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge program nodes, placing NEW nodes at HEAD (mirror upstream).

    Used when gate_result=wait/unknown so the AI's terminating node stays last.
    Applies the same AI-PRIMARY / program-authoritative distinction as
    ``_merge_program_nodes``.
    """
    result = [item for item in trace if isinstance(item, dict)]
    prog_by_id = {
        n["node_id"]: n for n in program_nodes
        if isinstance(n, dict) and "node_id" in n
    }
    replaced_ids: set[str] = set()
    for i, item in enumerate(result):
        nid = str(item.get("node_id", "")).strip()
        if nid not in prog_by_id:
            continue
        if nid in AI_PRIMARY_NODES:
            if nid in AI_PRIMARY_SUPPLEMENT_NODES:
                prog_node = prog_by_id[nid]
                prog_reason = str(prog_node.get("reason", "") or "").strip()
                prog_bar_range = str(prog_node.get("bar_range", "") or "").strip()
                if prog_reason:
                    ai_reason = str(item.get("reason", "") or "").strip()
                    supplement = f"【程序参考数据（{prog_bar_range}）：{prog_reason}】"
                    if supplement not in ai_reason:
                        result[i] = dict(item)
                        result[i]["reason"] = f"{ai_reason} {supplement}".strip()
        else:
            result[i] = prog_by_id[nid]
        replaced_ids.add(nid)
    new_nodes = sorted(
        [node for nid, node in prog_by_id.items() if nid not in replaced_ids],
        key=lambda x: _node_id_sort_key(str(x.get("node_id", ""))),
    )
    return new_nodes + result



# ── §9 stage2 judges (mirror upstream decision_nodes.py:1536-1779) ──────────
# These read ``features: dict[int, KlineGeometryFeature]`` (seq → geometry).
# ``judge_signal_bar_closed`` keeps a ``frame`` param for signature parity with
# upstream but does not read it (all bars in the frame are closed).


def judge_signal_bar_closed(sig: int, frame: Any = None) -> NodeFill:
    """§9.1: signal bar is always closed (all bars in the frame are closed)."""
    return NodeFill(
        node_id="9.1",
        answer="是",
        reason=f"K{sig}为已收盘K线（frame内所有K线均已收盘），可作为信号棒。",
        bar_range=f"K{sig}",
    )


# §9.2 direction consistency sets. Outside bars are intentionally excluded from
# the primary "consistent" set (文件16 §外包棒: "almost never wise to enter on
# outside bar breakout"); they land in a "weak" set that earns 否 with a warning
# so AI can still override via node_overrides if context warrants.
_LONG_BAR_TYPES: frozenset[str] = frozenset({"trend_bull"})
_SHORT_BAR_TYPES: frozenset[str] = frozenset({"trend_bear"})
_LONG_BAR_TYPES_WEAK: frozenset[str] = frozenset({"outside_bull"})
_SHORT_BAR_TYPES_WEAK: frozenset[str] = frozenset({"outside_bear"})


def judge_signal_bar_direction(
    sig: int,
    order_direction: str | None,
    features: dict[int, KlineGeometryFeature],
) -> NodeFill:
    """§9.2: check signal bar direction consistency with order direction.

    Classification:
      trend_bull / trend_bear → 是 (consistent, strong signal bar)
      outside_bull / outside_bear → 否 with warning (outside bar = K-line
        level congestion zone; Al Brooks: "almost never wise to enter on
        outside bar breakout")
      doji / inside / other / unknown → 否 (not directionally consistent)
    """
    if not order_direction or order_direction not in ("做多", "做空"):
        return NodeFill(
            node_id="9.2",
            answer="不适用",
            reason="无交易计划方向（order_direction缺失），§9.2不适用。",
            bar_range="不适用",
        )

    feat = features.get(sig)
    bar_type = str(feat.bar_type) if feat else "unknown"

    if order_direction == "做多":
        if bar_type in _LONG_BAR_TYPES:
            answer = "是"
            reason = (
                f"K{sig} bar_type={bar_type}，属于做多强信号棒类型"
                f"（{sorted(_LONG_BAR_TYPES)}），方向一致。"
            )
        elif bar_type in _LONG_BAR_TYPES_WEAK:
            answer = "否"
            reason = (
                f"K{sig} bar_type={bar_type}（外包棒），方向偏多但"
                "外包棒是K线级别的凝滞区，直接追外包棒突破风险高；"
                "建议等待后续确认棒或在 node_overrides 中说明理由后覆盖。"
            )
        else:
            answer = "否"
            reason = (
                f"K{sig} bar_type={bar_type}，"
                f"做多强信号棒类型={sorted(_LONG_BAR_TYPES)}，"
                "方向不一致。"
            )
    else:  # 做空
        if bar_type in _SHORT_BAR_TYPES:
            answer = "是"
            reason = (
                f"K{sig} bar_type={bar_type}，属于做空强信号棒类型"
                f"（{sorted(_SHORT_BAR_TYPES)}），方向一致。"
            )
        elif bar_type in _SHORT_BAR_TYPES_WEAK:
            answer = "否"
            reason = (
                f"K{sig} bar_type={bar_type}（外包棒），方向偏空但"
                "外包棒是K线级别的凝滞区，直接追外包棒突破风险高；"
                "建议等待后续确认棒或在 node_overrides 中说明理由后覆盖。"
            )
        else:
            answer = "否"
            reason = (
                f"K{sig} bar_type={bar_type}，"
                f"做空强信号棒类型={sorted(_SHORT_BAR_TYPES)}，"
                "方向不一致。"
            )

    return NodeFill(
        node_id="9.2",
        answer=answer,
        reason=reason,
        bar_range=f"K{sig}",
    )


def judge_signal_bar_length(
    sig: int,
    features: dict[int, KlineGeometryFeature],
) -> NodeFill:
    """§9.3: check if signal bar is overlong (range_atr_ratio > 2.0)."""
    feat = features.get(sig)
    ratio = feat.range_atr_ratio if feat else None

    if ratio is None:
        answer = "是"
        reason = (
            f"K{sig} range_atr_ratio无法计算（ATR预热不足或range=0），"
            "按潜在过长保守处理→是。"
        )
    elif ratio > SIGNAL_BAR_LONG_ATR_RATIO:
        answer = "是"
        reason = (
            f"K{sig} range_atr_ratio={ratio:.3f} > {SIGNAL_BAR_LONG_ATR_RATIO}，"
            "信号棒过长，止损可能超过ATR 2倍，需用资金管理止损或放弃。"
        )
    else:
        answer = "否"
        reason = (
            f"K{sig} range_atr_ratio={ratio:.3f} ≤ {SIGNAL_BAR_LONG_ATR_RATIO}，"
            "信号棒长度在可接受范围内，不过长。"
        )

    return NodeFill(
        node_id="9.3",
        answer=answer,
        reason=reason,
        bar_range=f"K{sig}",
    )


def judge_follow_through(
    sig: int,
    features: dict[int, KlineGeometryFeature],
) -> NodeFill:
    """§9.5: follow_through_1_2 mapping."""
    feat = features.get(sig)
    ft = feat.follow_through_1_2 if feat else None

    _FT_MAP = {
        "yes": "是",
        "failed": "否",
        "no": "否",
        "pending": "等待",
    }

    if ft in _FT_MAP:
        answer = _FT_MAP[ft]
        reason = f"K{sig}的follow_through_1_2={ft!r}→{answer}。"
    else:
        answer = "等待"
        reason = f"K{sig}的follow_through_1_2={ft!r}（缺失或未知），保守取等待。"

    # bar_range covers signal bar and subsequent bars
    if sig > 1:
        bar_range = f"K{sig}-K1"
    else:
        bar_range = "K1"

    return NodeFill(
        node_id="9.5",
        answer=answer,
        reason=reason,
        bar_range=bar_range,
    )


# ── OrderMethodRouter (mirror upstream decision_nodes.py:1817-2137) ─────────


def route_order_method(
    stage1_json: dict[str, Any] | None,
    decision: dict[str, Any],
    decision_trace: list[dict[str, Any]],
) -> list[NodeFill]:
    """Route order method based on cycle_position; return §11 NodeFill list.

    Returns a list of ``NodeFill`` objects (caller converts to trace dicts).
    Empty list when the order is already no-order or blocked by a safety gate
    (§10.3=否 / §14 violation), so no §11 nodes are injected.
    """
    decision = _coerce_dict(decision)
    stage1 = _coerce_dict(stage1_json)
    decision_trace = _coerce_trace_list(decision_trace)
    order_type = decision.get("order_type")

    # Safety: if already no-order, don't inject §11 nodes
    if order_type == "不下单":
        return []

    def _trace_answer(trace: list, node_id: str) -> str | None:
        for item in trace:
            if not isinstance(item, dict):
                continue
            if str(item.get("node_id", "")).strip() == node_id:
                return str(item.get("answer", "")).strip()
        return None

    # Check safety gates: §10.3=否
    if _trace_answer(decision_trace, "10.3") == "否":
        return []

    def _sec14_violated(trace: list) -> bool:
        _DENIAL_PHRASES = (
            "未触犯", "未违反", "无触犯", "无违规",
            "通过扫描", "扫描通过", "无禁止", "未触发",
        )
        for item in trace:
            if not isinstance(item, dict):
                continue
            nid = str(item.get("node_id", "")).strip()
            if not nid.startswith("14"):
                continue
            if str(item.get("answer", "")).strip() != "是":
                continue
            # Cross-check reason: denial phrase means AI used wrong answer
            reason = str(item.get("reason", "") or "")
            if any(phrase in reason for phrase in _DENIAL_PHRASES):
                continue
            return True
        return False

    if _sec14_violated(decision_trace):
        return []

    cycle = "unknown"
    if stage1:
        cycle = str(stage1.get("cycle_position", "unknown") or "unknown").strip()

    candidate = _CYCLE_ORDER_METHOD.get(cycle, "不下单")
    model_order_type = str(decision.get("order_type") or "").strip()

    def _has_trade_prices() -> bool:
        return all(
            decision.get(k) is not None
            for k in (
                "entry_price",
                "stop_loss_price",
                "take_profit_price",
                "take_profit_price_2",
            )
        )

    # Preserve model's explicit limit/market choice when §10.3 already passed.
    if (
        model_order_type == "限价单"
        and _trace_answer(decision_trace, "10.3") == "是"
        and _has_trade_prices()
    ):
        candidate = "限价单"
    elif (
        model_order_type == "市价单"
        and _trace_answer(decision_trace, "10.3") == "是"
        and _has_trade_prices()
    ):
        candidate = "市价单"
    elif (
        model_order_type == "突破单"
        and _trace_answer(decision_trace, "10.3") == "是"
        and _has_trade_prices()
        and decision.get("entry_basis_bar")
        and decision.get("entry_basis_extreme")
    ):
        # broad_channel defaults to 限价单, but a pending breakdown/breakout
        # at basis±tick is not a sell-limit-above-market plan — preserve 突破单.
        candidate = "突破单"

    if candidate == "不下单":
        # Not a trading context for this cycle
        return []

    # ── spike_ending / spike_pullback exception ─────────────────────────────
    # When cycle_position=spike but spike_stage indicates the spike has already
    # ended (ending/pullback/channel), the default candidate is 市价单. Once the
    # spike exhausts itself the market enters a consolidation/pullback phase
    # where waiting for a breakout of the signal bar is the textbook entry.
    # Forcing 市价单 on a pending 突破单 is wrong. Preserve the model's 突破单
    # choice when spike_stage is exhausted and a valid breakout anchor exists.
    if cycle == "spike" and candidate == "市价单":
        spike_stage = str(stage1.get("spike_stage") or "").strip().lower()
        if spike_stage in ("ending", "pullback", "channel") and model_order_type == "突破单":
            has_basis = bool(
                decision.get("entry_basis_bar") and decision.get("entry_basis_extreme")
            )
            if has_basis:
                candidate = "突破单"
        return []

    # Breakout order: check for valid entry_basis; fall back to limit.
    breakout_fallback_to_limit = False
    if candidate == "突破单":
        has_basis = bool(
            decision.get("entry_basis_bar") and decision.get("entry_basis_extreme")
        )
        if not has_basis:
            # No breakout anchor → try limit at structural level.
            breakout_fallback_to_limit = True
            candidate = "限价单"

    # Determine which §11 node corresponds to the final method.
    # §11 structure:
    #   11.1: 趋势/尖峰 → 市价单 (spike)
    #   11.2: 通道 → 突破单 (channel) / 限价单 (broad_channel)
    #   11.3: 区间 → 限价单 (range)
    #   11.4: broad_channel → 限价单 (broad)
    _METHOD_NODE: dict[str, tuple[str, str]] = {
        "spike": ("11.1", "市价单"),
        "micro_channel": ("11.2", "突破单"),
        "tight_channel": ("11.2", "突破单"),
        "normal_channel": ("11.2", "突破单"),
        "broad_channel": ("11.2", "限价单"),
        "trading_range": ("11.3", "限价单"),
        "trending_tr": ("11.2", "突破单"),
    }

    cycle_node_info = _METHOD_NODE.get(cycle)
    if not cycle_node_info:
        return []

    final_node_id, _ = cycle_node_info

    # Update decision order_type to match candidate
    decision["order_type"] = candidate

    _node_reasons: dict[str, str] = {
        "11.1": "趋势/尖峰阶段，价格快速移动，适合市价单立即入场。",
        "11.2": "通道结构，等待突破确认，使用突破单。",
        "11.3": "交易区间，在区间边界附近使用限价单。",
        "11.4": "宽通道/特殊情况，使用限价单。",
    }

    nodes: list[NodeFill] = []
    all_nodes = ["11.1", "11.2", "11.3", "11.4"]
    final_idx = all_nodes.index(final_node_id) if final_node_id in all_nodes else -1

    for i, nid in enumerate(all_nodes):
        if i > final_idx:
            break
        answer = "是" if nid == final_node_id else "否"
        reason = _node_reasons.get(nid, f"§{nid}判定。")
        if nid == final_node_id:
            spike_stage_label = str(stage1.get("spike_stage") or "").strip().lower()
            if (
                cycle == "spike"
                and candidate == "突破单"
                and spike_stage_label in ("ending", "pullback", "channel")
            ):
                reason = (
                    f"cycle_position={cycle}（spike_stage={spike_stage_label}，尖峰已结束）"
                    f"→{candidate}（保留模型突破单选择；尖峰结束后等待信号棒突破确认是正确做法，"
                    "不应强制市价单立即追入）。" + reason
                )
            elif breakout_fallback_to_limit and candidate == "限价单":
                reason = (
                    f"cycle_position={cycle} 默认突破单，但无有效 entry_basis_bar/extreme；"
                    f"§10.3 已通过 → 改用限价单在结构位挂单（回撤/反弹到位入场）。"
                    + reason
                )
            else:
                reason = f"cycle_position={cycle}→{candidate}。" + reason
        nodes.append(
            NodeFill(
                node_id=nid,
                answer=answer,
                reason=reason,
                bar_range="K1",
            )
        )

    return nodes


# ── OverrideArbiter (mirror upstream decision_nodes.py:2147-2707) ───────────


def _conservativeness_rank(node_id: str, answer: str) -> int:
    """Return conservativeness rank for safety gate ordering.

    Higher = more conservative. Used to reject overrides that would make a
    safety gate more aggressive (e.g. §10.3 是→否, §14 否→是).
    """
    nid = str(node_id).strip()
    ans = str(answer).strip()
    if nid == "10.3":
        return 5 if ans == "否" else 3
    if nid == "14":
        return 5 if ans == "是" else 3
    # order_type dimension (§11 nodes)
    if nid in ("11.1", "11.2", "11.3", "11.4"):
        return 5 if ans == "不下单" else 3
    return 3


def write_override_trace(node: dict[str, Any], override: dict[str, Any]) -> None:
    """Write override trace fields to node (in-place). Records program original values."""
    node["program_answer"] = node.get("answer")
    if "branch" in node:
        node["program_branch"] = node.get("branch")
    node["answer"] = override["answer"]
    if override.get("branch"):
        node["branch"] = override["branch"]
    node["override_reason"] = str(override.get("override_reason", "")).strip()
    node["overridden_by_ai"] = True


def apply_overrides(
    program_nodes: list[dict[str, Any]],
    node_overrides: Any,
    *,
    out: dict[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    """Apply controlled overrides to program nodes.

    Returns final node list with override traces written. Rules (in order):

    1. node_overrides not a list → ignore all
    2. invalid element → skip
    3. locked node → ignore (log)
    4. missing override_reason → reject
    5. safety gate in aggressive direction → reject
    6. §2.3 direction consistency check
    7. valid override → accept, write trace
    """
    result = [dict(n) for n in program_nodes]
    prog_ids = {n["node_id"] for n in result if isinstance(n, dict) and "node_id" in n}

    if not isinstance(node_overrides, list):
        return result

    # Build index for fast lookup
    node_index = {n["node_id"]: i for i, n in enumerate(result) if isinstance(n, dict) and "node_id" in n}
    seen_overrides: set[str] = set()

    for ov in node_overrides:
        if not isinstance(ov, dict):
            continue
        node_id = str(ov.get("node_id", "")).strip()
        if not node_id:
            continue
        if node_id not in prog_ids:
            continue
        answer = str(ov.get("answer", "")).strip()
        if answer not in TRACE_ANSWERS:
            continue

        # Take first valid override per node_id
        if node_id in seen_overrides:
            continue
        seen_overrides.add(node_id)

        # Rule 3: locked node
        if node_id in LOCKED_NODES:
            logger.info(
                "apply_overrides: ignoring override for locked node %s (stage=%s)",
                node_id, stage,
            )
            continue

        # Rule 4: missing override_reason
        override_reason = str(ov.get("override_reason", "") or "").strip()
        if not override_reason:
            logger.debug(
                "apply_overrides: rejecting override for %s - missing override_reason",
                node_id,
            )
            continue

        # Rule 5: safety gate direction check
        if node_id in SAFETY_GATE_NODES:
            idx = node_index.get(node_id)
            if idx is not None:
                current_answer = str(result[idx].get("answer", "")).strip()
                current_rank = _conservativeness_rank(node_id, current_answer)
                new_rank = _conservativeness_rank(node_id, answer)
                if new_rank < current_rank:
                    logger.debug(
                        "apply_overrides: rejecting aggressive safety gate override "
                        "for %s (rank %d -> %d is less conservative)",
                        node_id, current_rank, new_rank,
                    )
                    continue

        # Rule 6: §2.3 direction consistency
        if node_id == "2.3":
            branch = str(ov.get("branch", "") or "").strip()
            valid = _validate_dir_override(answer, branch)
            if not valid:
                logger.debug(
                    "apply_overrides: rejecting §2.3 override - "
                    "answer/branch inconsistent: answer=%s branch=%s",
                    answer, branch,
                )
                continue
            # Accept: write trace and sync direction
            idx = node_index.get(node_id)
            if idx is not None:
                write_override_trace(result[idx], ov)
                direction_map = {"bullish": "bullish", "bearish": "bearish", "neutral": "neutral"}
                if branch in direction_map:
                    out["direction"] = direction_map[branch]
            continue

        # Rule 7: accept override for OVERRIDABLE_NODES
        if node_id in OVERRIDABLE_NODES:
            idx = node_index.get(node_id)
            if idx is not None:
                write_override_trace(result[idx], ov)
                # §11 override: sync order_type
                if node_id in ("11.1", "11.2", "11.3", "11.4"):
                    _sync_order_type_from_11_override(out, result[idx], ov)
                # §2.4 override: sync bar_analysis.always_in so the field stays
                # consistent with the final (possibly AI-overridden) §2.4 branch.
                if node_id == "2.4":
                    _sync_always_in_from_24_override(out, ov)

    return result


def _validate_dir_override(answer: str, branch: str) -> bool:
    """Validate §2.3 answer/branch consistency."""
    if branch in ("bullish", "bearish"):
        return answer == "是"
    elif branch == "neutral":
        return answer == "中性"
    return False  # invalid branch


def _sync_always_in_from_24_override(
    out: dict[str, Any],
    override: dict[str, Any],
) -> None:
    """After §2.4 override accepted, sync bar_analysis.always_in to match.

    Mapping:
      branch=AIL  → always_in="long"
      branch=AIS  → always_in="short"
      answer=否   → always_in="neutral"
    """
    bar_analysis = out.get("bar_analysis")
    if not isinstance(bar_analysis, dict):
        return
    branch = str(override.get("branch", "") or "").strip()
    answer = str(override.get("answer", "") or "").strip()
    if branch == "AIL":
        bar_analysis["always_in"] = "long"
    elif branch == "AIS":
        bar_analysis["always_in"] = "short"
    elif answer == "否":
        bar_analysis["always_in"] = "neutral"
    # If branch is unrecognised or missing, leave as-is to avoid silent corruption.


def _sync_order_type_from_11_override(
    out: dict[str, Any],
    node: dict[str, Any],
    override: dict[str, Any],
) -> None:
    """After §11 override accepted, sync decision.order_type if not 不下单."""
    decision = out.get("decision")
    if not isinstance(decision, dict):
        return
    new_answer = str(override.get("answer", "")).strip()
    if new_answer == "是":
        # Determine which order type this §11 node represents
        node_id = str(node.get("node_id", ""))
        node_method_map = {
            "11.1": "市价单",
            "11.2": "突破单",
            "11.3": "限价单",
            "11.4": "限价单",
        }
        method = node_method_map.get(node_id)
        if method and decision.get("order_type") != "不下单":
            decision["order_type"] = method


# ── Public API: apply_stage1_nodes ───────────────────────────────────────────


def apply_stage1_nodes(
    out: dict[str, Any],
    feature_rows: list[dict[str, Any]] | None,
) -> bool:
    """Compute §1.1/§2.3/§2.4 programmatically and merge into ``out`` in place.

    Called from :func:`normalize_market_diagnosis` before the validator runs.
    Overwrites any LLM-provided 1.1/2.3/2.4 nodes and syncs the top-level
    ``direction`` field with node 2.3's branch.

    Returns True when the engine wrote nodes successfully.
    """
    frame = _frame_from_feature_rows(feature_rows)
    if frame is None:
        return False

    fill_11 = judge_data_sufficiency(frame)
    direction, fill_23 = judge_direction(frame)
    fill_24 = judge_always_in(frame)

    node_11 = _build_program_trace_node(fill_11)
    node_23 = _build_program_trace_node(fill_23)
    node_24 = _build_program_trace_node(fill_24)
    program_nodes = [node_11, node_23, node_24]

    # Step 4: Apply overrides (mirror upstream apply_stage1 step 4).
    node_overrides = out.get("node_overrides")
    final_nodes = apply_overrides(
        program_nodes,
        node_overrides,
        out=out,
        stage="stage1",
    )

    # Step 5: Merge into gate_trace.
    # If gate_result is wait/unknown, prepend program nodes so the AI's
    # terminating node (answer=否/等待) remains at the end.
    gate_trace = out.get("gate_trace")
    if not isinstance(gate_trace, list):
        gate_trace = []
    gate_result = str(out.get("gate_result", "")).lower()
    if gate_result in ("wait", "unknown"):
        out["gate_trace"] = _merge_program_nodes_head(gate_trace, final_nodes)
    else:
        out["gate_trace"] = _merge_program_nodes(gate_trace, final_nodes)

    # Sync top-level direction with program-computed §2.3 result.
    out["direction"] = direction

    # Step 6: Sync bar_analysis.always_in from the final §2.4 node.
    # apply_overrides handles the AI-override path via
    # _sync_always_in_from_24_override; this covers the non-override path.
    node_24_final = next(
        (n for n in final_nodes if isinstance(n, dict) and str(n.get("node_id", "")) == "2.4"),
        None,
    )
    if node_24_final is not None:
        bar_analysis = out.get("bar_analysis")
        if isinstance(bar_analysis, dict):
            branch_24 = str(node_24_final.get("branch", "") or "").strip()
            answer_24 = str(node_24_final.get("answer", "") or "").strip()
            if branch_24 == "AIL":
                bar_analysis["always_in"] = "long"
            elif branch_24 == "AIS":
                bar_analysis["always_in"] = "short"
            elif answer_24 == "否":
                bar_analysis["always_in"] = "neutral"

    return True


# ── DecisionNodeEngine (mirror upstream decision_nodes.py:2715-3041) ────────


class DecisionNodeEngine:
    """Deterministic decision node engine (stateless, pure-function based).

    ``apply_stage2`` orchestrates §9.1/§9.2/§9.3/§9.5 + §11 routing for a
    stage2 decision, applying overrides and merging into ``decision_trace``.
    """

    @staticmethod
    def apply_stage2(
        out: dict[str, Any],
        feature_rows: list[dict[str, Any]] | None,
        stage1_json: dict[str, Any] | None,
    ) -> None:
        """In-place modify stage2 JSON: fill §9.1/§9.2/§9.3/§9.5/§11, apply overrides.

        Adapts upstream ``DecisionNodeEngine.apply_stage2(out, frame, stage1_json)``:
        the ``frame`` (KlineFrame) argument is replaced by ``feature_rows``
        (list[dict]), which is converted to geometry features via
        ``compute_kline_geometry_features``.
        """
        # Short-circuit for gate-shortcircuited stage2
        if out.get("gate_shortcircuited"):
            return

        # Ensure decision_trace exists and is a clean list of dicts
        out.setdefault("decision_trace", [])
        out["decision_trace"] = _coerce_trace_list(out.get("decision_trace"))

        raw_decision = out.get("decision")
        decision = _coerce_dict(raw_decision)
        if raw_decision is not None and not isinstance(raw_decision, dict):
            out["decision"] = decision

        order_direction = str(decision.get("order_direction", "") or "").strip() or None

        # Get geometry features (dict[int, KlineGeometryFeature])
        features: dict[int, KlineGeometryFeature] = {}
        try:
            features = compute_kline_geometry_features(feature_rows)
        except Exception:  # noqa: BLE001
            pass

        # Locate signal bar seq
        sig = _get_signal_seq(out)

        # Check §9.0 answer: if AI said no valid signal bar, skip §9.1-9.5
        # rather than injecting misleading program-computed values.
        # "否"  = no valid signal bar exists right now
        # "等待" = AI semantically means "no valid signal bar" (should be "否" but
        #          AI sometimes conflates "does it exist?" with "should I wait?").
        # Both map to skip §9.1-9.5 — unless this is a planned limit order.
        _dt = out["decision_trace"]
        _node_90 = next(
            (x for x in _dt if isinstance(x, dict) and str(x.get("node_id", "")) == "9.0"),
            None,
        )
        _planned_limit = is_planned_limit_order(out)
        _section9_has_signal = True
        if _node_90 is not None:
            _ans_90 = str(_node_90.get("answer", "") or "").strip()
            if _ans_90 in ("否", "等待") and not _planned_limit:
                _section9_has_signal = False
            elif _ans_90 in ("否", "等待") and has_background_limit_path(out):
                _section9_has_signal = True

        # Step 1: SignalBarJudge → §9.1, §9.2, §9.3
        fill_91 = judge_signal_bar_closed(sig)
        fill_92 = judge_signal_bar_direction(sig, order_direction, features)
        fill_93 = judge_signal_bar_length(sig, features)

        # Step 2: FollowThroughJudge → §9.5
        fill_95 = judge_follow_through(sig, features)

        # Step 3: OrderMethodRouter → §11 nodes
        # Only inject if order is a trade type (not 不下单)
        current_order_type = decision.get("order_type")
        decision_trace = out["decision_trace"]
        sec11_fills: list[NodeFill] = []
        if current_order_type != "不下单":
            sec11_fills = route_order_method(stage1_json, decision, decision_trace)

        # Convert to dicts
        node_91 = _build_program_trace_node(fill_91)
        node_92 = _build_program_trace_node(fill_92)
        if fill_92.answer == "不适用":
            node_92["skipped"] = True
        node_93 = _build_program_trace_node(fill_93)
        node_95 = _build_program_trace_node(fill_95)

        # When §9.0=否 (no valid signal bar), mark §9.1-9.5 as skipped so they
        # don't appear as contradictory program-filled nodes in the trace.
        if not _section9_has_signal:
            _skip_reason = "§9.0=否（无有效信号棒），§9.1-9.5不适用，程序跳过。"
            for _node in (node_91, node_92, node_93, node_95):
                _node["skipped"] = True
                _node["answer"] = "不适用"
                _node["reason"] = _skip_reason
        elif _planned_limit:
            _bar_analysis = out.get("bar_analysis")
            _signal_bar = (
                _bar_analysis.get("signal_bar")
                if isinstance(_bar_analysis, dict)
                else None
            )
            _no_signal_bar = (
                not isinstance(_signal_bar, dict) or not _signal_bar.get("bar")
            )
            if _no_signal_bar or has_background_limit_path(out):
                _skip_reason = (
                    "计划型限价单（§9.0P 或 §9.0 背景路径），尚无已收盘信号棒，"
                    "§9.1-9.3不适用。"
                )
                for _node in (node_91, node_92, node_93):
                    _node["skipped"] = True
                    _node["answer"] = "不适用"
                    _node["reason"] = _skip_reason

        sec11_nodes = [_build_program_trace_node(f) for f in sec11_fills]
        program_nodes = [node_91, node_92, node_93, node_95] + sec11_nodes

        # Step 4: Apply overrides
        node_overrides = out.get("node_overrides")
        final_nodes = apply_overrides(
            program_nodes,
            node_overrides,
            out=out,
            stage="stage2",
        )

        # Step 5: Merge into decision_trace
        out["decision_trace"] = _merge_program_nodes(
            out["decision_trace"], final_nodes
        )


# ── Signal-bar seq resolver (mirror upstream decision_nodes.py:1442-1472) ───


def _get_signal_seq(out: dict[str, Any]) -> int:
    """Locate signal bar seq: prefer ``bar_analysis.signal_bar.bar``, else K1.

    Upstream imports ``parse_k_seq`` from ``price_tick``; here we reuse the
    module-local ``_seq_from_k`` (same regex) to avoid a cross-module dep.
    """
    bar_analysis = out.get("bar_analysis")
    if isinstance(bar_analysis, dict):
        signal_bar = bar_analysis.get("signal_bar")
        if isinstance(signal_bar, dict):
            seq = _seq_from_k(signal_bar.get("bar"))
            if seq is not None and seq >= 1:
                return seq
    return 1  # default to K1


# ── Planned-limit helpers (mirror upstream decision_nodes.py:1475-1530) ──────
# 纯 dict 逻辑,无外部依赖,从 SOURCE 原样搬运。

def has_background_limit_path(out: dict[str, Any]) -> bool:
    """True when decision_trace records §9.0P=是 (background-driven limit path)."""
    trace = out.get("decision_trace")
    if not isinstance(trace, list):
        return False
    for item in trace:
        if not isinstance(item, dict):
            continue
        if str(item.get("node_id", "")).strip() != "9.0P":
            continue
        return str(item.get("answer", "") or "").strip() == "是"
    return False


def is_planned_limit_order(out: dict[str, Any]) -> bool:
    """True when order is a pending limit plan without requiring a closed signal bar."""
    decision = out.get("decision")
    if not isinstance(decision, dict) or decision.get("order_type") != "限价单":
        return False
    if has_background_limit_path(out):
        return True
    bar_analysis = out.get("bar_analysis")
    if not isinstance(bar_analysis, dict):
        return False
    entry_bar = bar_analysis.get("entry_bar")
    signal_bar = bar_analysis.get("signal_bar")
    if not isinstance(entry_bar, dict) or not isinstance(signal_bar, dict):
        return False
    strength = str(entry_bar.get("strength", "") or "").strip().lower()
    freshness = str(entry_bar.get("freshness", "") or "").strip().lower()
    pending = (
        strength == "not_triggered"
        or entry_bar.get("bar") is None
        or freshness == "pending"
    )
    if not pending:
        return False
    quality = str(signal_bar.get("quality", "") or "").strip().lower()
    pattern = str(signal_bar.get("pattern", "") or "").strip().lower()
    if signal_bar.get("bar") is None and quality in ("invalid", "weak"):
        return True
    if quality == "weak" and pattern in (
        "",
        "none",
        "tr_boundary",
        "breakout_pullback",
        "h1",
        "h2",
        "l1",
        "l2",
        "wedge",
        "mtr",
        "trendline",
    ):
        return True
    return False
