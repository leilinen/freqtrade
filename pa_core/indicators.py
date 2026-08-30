"""EMA and ATR indicators — full and incremental.

Ported verbatim from PA_Agent ``pa_agent/indicators/ema.py`` and
``pa_agent/indicators/atr.py`` (baseline 1090a5b), merged into one module.
Do not swap these for TA-Lib equivalents: PA decision prompts are built from
values produced by exactly these implementations (seed = simple mean of first
*period* values, Wilder smoothing for ATR).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ── EMA ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmaState:
    """Minimal state for incremental EMA computation."""
    last: float       # most recent EMA value (nan during warm-up)
    period: int
    count: int        # number of values seen so far
    _sum: float       # running sum during warm-up phase


def ema_full(values: list[float], period: int) -> list[float]:
    """Compute EMA over *values* (oldest-first).

    Returns a list of the same length:
    - Indices 0 .. period-2: nan  (warm-up)
    - Index period-1: simple mean of first *period* values
    - Index period .. end: EMA with multiplier α = 2/(period+1)

    Args:
        values: Price series, oldest first.
        period: EMA period (must be >= 1).
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    n = len(values)
    result = [math.nan] * n
    if n < period:
        return result

    alpha = 2.0 / (period + 1)
    # Seed with simple mean of first *period* values
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    prev = seed
    for i in range(period, n):
        prev = values[i] * alpha + prev * (1.0 - alpha)
        result[i] = prev
    return result


def ema_incremental(state: EmaState, x: float) -> EmaState:
    """Update EMA state with one new value *x*.

    During warm-up (count < period), accumulates a running sum.
    At count == period-1 (i.e., after this call count == period), seeds the EMA.
    After warm-up, applies the standard EMA formula.
    """
    period = state.period
    count = state.count + 1
    alpha = 2.0 / (period + 1)

    if count < period:
        # Still in warm-up: accumulate sum, last stays nan
        return EmaState(last=math.nan, period=period, count=count, _sum=state._sum + x)
    elif count == period:
        # Seed: simple mean
        seed = (state._sum + x) / period
        return EmaState(last=seed, period=period, count=count, _sum=0.0)
    else:
        # Normal EMA update
        new_last = x * alpha + state.last * (1.0 - alpha)
        return EmaState(last=new_last, period=period, count=count, _sum=0.0)


def make_ema_state(period: int) -> EmaState:
    """Create a fresh EmaState for a given period."""
    return EmaState(last=math.nan, period=period, count=0, _sum=0.0)


def state_after(values: list[float], period: int) -> EmaState:
    """Return the EmaState after processing all values in *values*."""
    state = make_ema_state(period)
    for v in values:
        state = ema_incremental(state, v)
    return state


# ── ATR (Wilder) ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AtrState:
    """Minimal state for incremental ATR computation."""
    last: float       # most recent ATR value (nan during warm-up)
    period: int
    count: int        # number of bars seen so far
    prev_close: float # previous bar's close (nan if not yet set)
    _sum_tr: float    # running sum of TR during warm-up


def _true_range(high: float, low: float, prev_close: float) -> float:
    """Compute True Range for a single bar."""
    hl = abs(high - low)
    if math.isnan(prev_close):
        return hl
    return max(hl, abs(high - prev_close), abs(low - prev_close))


def atr_full(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float]:
    """Compute ATR over parallel OHLC lists (oldest-first).

    Returns a list of the same length:
    - Indices 0 .. period-2: nan  (warm-up)
    - Index period-1: simple mean of first *period* True Ranges
    - Index period .. end: Wilder smoothing  ATR_t = (ATR_{t-1}*(period-1) + TR_t) / period

    Args:
        highs, lows, closes: Price series, oldest first, same length.
        period: ATR period (must be >= 1).
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    n = len(highs)
    if n != len(lows) or n != len(closes):
        raise ValueError("highs, lows, closes must have the same length")
    result = [math.nan] * n
    if n < period:
        return result

    # Compute TR for each bar
    trs: list[float] = []
    for i in range(n):
        prev_c = closes[i - 1] if i > 0 else math.nan
        trs.append(_true_range(highs[i], lows[i], prev_c))

    # Seed with simple mean of first *period* TRs
    seed = sum(trs[:period]) / period
    result[period - 1] = seed
    prev_atr = seed
    for i in range(period, n):
        prev_atr = (prev_atr * (period - 1) + trs[i]) / period
        result[i] = prev_atr
    return result


def atr_incremental(state: AtrState, high: float, low: float, close: float) -> AtrState:
    """Update ATR state with one new bar (high, low, close).

    During warm-up (count < period), accumulates TR sum.
    At count == period, seeds the ATR.
    After warm-up, applies Wilder smoothing.
    """
    period = state.period
    count = state.count + 1
    tr = _true_range(high, low, state.prev_close)

    if count < period:
        return AtrState(
            last=math.nan, period=period, count=count,
            prev_close=close, _sum_tr=state._sum_tr + tr,
        )
    elif count == period:
        seed = (state._sum_tr + tr) / period
        return AtrState(
            last=seed, period=period, count=count,
            prev_close=close, _sum_tr=0.0,
        )
    else:
        new_last = (state.last * (period - 1) + tr) / period
        return AtrState(
            last=new_last, period=period, count=count,
            prev_close=close, _sum_tr=0.0,
        )


def make_atr_state(period: int = 14) -> AtrState:
    """Create a fresh AtrState for a given period."""
    return AtrState(last=math.nan, period=period, count=0, prev_close=math.nan, _sum_tr=0.0)


def state_after_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> AtrState:
    """Return the AtrState after processing all bars."""
    state = make_atr_state(period)
    for h, l, c in zip(highs, lows, closes):
        state = atr_incremental(state, h, l, c)
    return state
