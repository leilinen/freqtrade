"""Stage1 DecisionNodeEngine port (subset: §1.1, §2.3 direction + §2.4 Always-In).

The upstream ``PA_Agent/pa_agent/ai/decision_nodes.py`` is 3041 lines and
covers stage1 (1.1/1.3/2.3/2.4/2.5) plus stage2 (9.x/11.x). This port
ships §1.1, §2.3 and §2.4 because:

* freqtrade should mirror upstream PA_Agent: §1.1/§2.3/§2.4 are
  deterministic program nodes, not AI-written trace nodes.
* freqtrade previously asked the LLM to fill 2.3/2.4 with no schema
  guidance, producing systematic ``branch`` field errors
  (``market_diagnosis_gate_trace_direction_branch_conflict``).
* Computing these two nodes in code (5-signal direction vote + Brooks
  Always-In dual-window check) eliminates the entire error class.

Not ported here (future issues):
  * §1.3 / §2.5 / §9.x / §11.x — freqtrade lets the LLM fill these
    today without systematic errors; no urgent need.
  * ``node_overrides`` AI-override mechanism — no AI主动覆盖 use case yet.

This module is **self-contained**: it only consumes OHLC + EMA20 + ATR14
and recomputes all derived features (trend bars, swings, overlap, gravity)
inline. It does not depend on ``kline_features`` or any pre-computed
feature rows.

The frame adapter (``_Frame`` / ``_frame_from_feature_rows``) bridges
freqtrade's ``feature_rows: list[dict]`` (with ``k``/``open``/``high``/
``low``/``close``/``ema20``/``atr14`` keys) to the bar/indicators shape
expected by the upstream functions.
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
    """Replace AI-written program nodes in trace; insert if missing."""
    result = [item for item in trace if isinstance(item, dict)]
    prog_by_id = {
        n["node_id"]: n for n in program_nodes
        if isinstance(n, dict) and "node_id" in n
    }
    replaced_ids: set[str] = set()
    for i, item in enumerate(result):
        nid = str(item.get("node_id", "")).strip()
        if nid in prog_by_id:
            result[i] = prog_by_id[nid]
            replaced_ids.add(nid)
    new_nodes = [
        node for nid, node in prog_by_id.items() if nid not in replaced_ids
    ]
    if new_nodes:
        result.extend(new_nodes)
        result.sort(key=lambda x: _node_id_sort_key(str(x.get("node_id", ""))))
    return result


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

    gate_trace = out.get("gate_trace")
    if not isinstance(gate_trace, list):
        gate_trace = []
    merged = _merge_program_nodes(gate_trace, program_nodes)
    out["gate_trace"] = merged

    # Sync top-level direction with program-computed §2.3 result.
    # Mirror upstream behavior (decision_nodes.py line 956-959 returns direction
    # and stage1_normalizer assigns it to out["direction"]).
    out["direction"] = direction

    # bar_analysis.always_in sync (mirror upstream apply_stage1, Step 6).
    # The validator's MARKET_DIAGNOSIS_ALWAYS_IN enum is {"long","short","neutral"},
    # so the §2.4 branch vocabulary (AIL/AIS) must be translated. See upstream
    # decision_nodes.py apply_stage1 lines 2832-2837.
    bar_analysis = out.get("bar_analysis")
    if isinstance(bar_analysis, dict):
        branch_24 = str(fill_24.branch or "").strip()
        answer_24 = str(fill_24.answer or "").strip()
        if branch_24 == "AIL":
            bar_analysis["always_in"] = "long"
        elif branch_24 == "AIS":
            bar_analysis["always_in"] = "short"
        elif answer_24 == "否":
            bar_analysis["always_in"] = "neutral"

    return True
