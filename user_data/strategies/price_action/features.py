"""L1 feature engineering for PA_Agent-style price-action analysis."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


@dataclass(frozen=True)
class L1FeatureResult:
    """Pure-Python feature output consumed by L2/L4 prompts."""

    symbol: str
    timeframe: str
    market: str
    candle_time: datetime
    kline_table: str
    feature_table: str
    market_features_text: str
    latest_features: dict[str, Any]
    market_features: dict[str, Any]
    rows: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market": self.market,
            "candle_time": self.candle_time.isoformat(),
            "latest_features": self.latest_features,
            "market_features": self.market_features,
            "rows": self.rows,
        }


def parse_timeframe(timeframe: str) -> timedelta | None:
    """Parse common Freqtrade timeframes into a duration."""
    match = re.fullmatch(r"(\d+)([mhdw])", timeframe.strip().lower())
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    return None


def latest_closed_candle_time(
    dataframe: DataFrame,
    timeframe: str,
    *,
    market: str | None = None,
    now: datetime | pd.Timestamp | None = None,
) -> datetime | None:
    """Return the newest closed candle open-time after PA_Agent seq0 filtering."""
    closed = select_closed_candles(dataframe, timeframe, market=market, now=now)
    if closed.empty:
        return None
    ts = closed.iloc[-1]["date"]
    return _to_utc_naive(ts)


def select_closed_candles(
    dataframe: DataFrame,
    timeframe: str,
    *,
    market: str | None = None,
    now: datetime | pd.Timestamp | None = None,
) -> DataFrame:
    """Drop any still-forming tail candle; the returned tail has K1 as last row.

    Freqtrade normally passes only closed candles. The wall-clock guard keeps the
    PA_Agent convention when a feed includes a seq0/forming bar at the end.
    """
    if dataframe.empty:
        return dataframe.copy()
    if "date" not in dataframe.columns:
        raise ValueError("OHLCV dataframe must include a date column")

    df = dataframe.copy()
    df["_date_utc"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df[df["_date_utc"].notna()].sort_values("_date_utc")
    duration = parse_timeframe(timeframe)
    if duration is None:
        return df.drop(columns=["_date_utc"])

    if now is None:
        now_ts = pd.Timestamp.now(tz="UTC")
    else:
        now_ts = pd.Timestamp(now)
        now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")

    close_times = _expected_close_times(df["_date_utc"], timeframe, duration, market)
    closed = df[close_times <= now_ts]
    return closed.drop(columns=["_date_utc"])


def _expected_close_times(
    open_times_utc: pd.Series,
    timeframe: str,
    duration: timedelta,
    market: str | None,
) -> pd.Series:
    """Return expected close times for generic crypto/A-share bars."""
    if (market or "").lower() == "ashare" and timeframe.lower() == "1d":
        local = open_times_utc.dt.tz_convert("Asia/Shanghai")
        local_close = local.dt.normalize() + pd.Timedelta(hours=15)
        return local_close.dt.tz_convert("UTC")
    return open_times_utc + pd.Timedelta(duration)


def calculate_ema(values: pd.Series, period: int = 20) -> pd.Series:
    """EMA with PA_Agent-style SMA seed."""
    values = pd.to_numeric(values, errors="coerce").astype(float)
    ema = pd.Series(np.nan, index=values.index, dtype=float)
    if len(values) < period:
        return ema
    seed = values.iloc[:period].mean()
    ema.iloc[period - 1] = seed
    alpha = 2.0 / (period + 1.0)
    prev = seed
    for pos in range(period, len(values)):
        current = values.iloc[pos]
        if pd.isna(current):
            ema.iloc[pos] = prev
            continue
        prev = alpha * float(current) + (1.0 - alpha) * prev
        ema.iloc[pos] = prev
    return ema


def calculate_atr(df: DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR seeded by the first period true ranges."""
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = pd.Series(np.nan, index=df.index, dtype=float)
    if len(tr) < period:
        return atr
    seed = tr.iloc[:period].mean()
    atr.iloc[period - 1] = seed
    prev = seed
    for pos in range(period, len(tr)):
        current = tr.iloc[pos]
        if pd.isna(current):
            atr.iloc[pos] = prev
            continue
        prev = ((period - 1.0) * prev + float(current)) / period
        atr.iloc[pos] = prev
    return atr


def build_l1_features(
    dataframe: DataFrame,
    *,
    symbol: str,
    timeframe: str,
    market: str,
    window: int = 30,
    warmup: int = 50,
    now: datetime | pd.Timestamp | None = None,
) -> L1FeatureResult:
    """Build newest-first K-line and feature tables from OHLCV."""
    closed = select_closed_candles(dataframe, timeframe, market=market, now=now)
    if closed.empty:
        raise ValueError("No closed OHLCV candle is available for PA analysis")

    needed = max(window + warmup, window)
    df = closed.tail(needed).copy().reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        if column not in df.columns:
            raise ValueError(f"OHLCV dataframe missing {column}")
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df[df["date"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid dated candle is available for PA analysis")

    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["direction"] = np.where(df["close"] >= df["open"], "bull", "bear")
    df["body_pct"] = _safe_div(df["body"], df["range"], fill=0.0)
    df["upper_wick"] = df["high"] - np.maximum(df["open"], df["close"])
    df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]
    df["upper_wick_pct"] = _safe_div(df["upper_wick"], df["range"], fill=0.0)
    df["lower_wick_pct"] = _safe_div(df["lower_wick"], df["range"], fill=0.0)
    df["close_position"] = _safe_div(df["close"] - df["low"], df["range"], fill=0.5)
    median_body = df["body"].rolling(window=20, min_periods=5).median()
    df["body_ratio"] = _safe_div(df["body"], median_body, fill=0.0)

    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    df["inside"] = (df["high"] <= prev_high) & (df["low"] >= prev_low)
    df["outside"] = (df["high"] >= prev_high) & (df["low"] <= prev_low)
    df["overlap_prev_ratio"] = _safe_div(
        (np.minimum(df["high"], prev_high) - np.maximum(df["low"], prev_low)).clip(lower=0),
        np.maximum(df["high"], prev_high) - np.minimum(df["low"], prev_low),
        fill=np.nan,
    )
    df["inside_sequence"] = "none"
    df.loc[df["inside"] & df["inside"].shift(1).fillna(False), "inside_sequence"] = "ii"
    df.loc[
        df["inside"]
        & df["inside"].shift(1).fillna(False)
        & df["inside"].shift(2).fillna(False),
        "inside_sequence",
    ] = "iii"

    df["ema20"] = calculate_ema(df["close"], 20)
    df["atr14"] = calculate_atr(df, 14)
    df["range_atr"] = _safe_div(df["range"], df["atr14"], fill=np.nan)
    df["ema20_relation"] = "unknown"
    df.loc[df["close"] > df["ema20"], "ema20_relation"] = "above"
    df.loc[df["close"] < df["ema20"], "ema20_relation"] = "below"
    df.loc[df["close"] == df["ema20"], "ema20_relation"] = "touch"
    df["ema20_gap_atr"] = _safe_div((df["close"] - df["ema20"]).abs(), df["atr14"], fill=np.nan)
    prev_atr = df["atr14"].shift(1)
    df["atr_expand_ratio"] = _safe_div(df["atr14"], prev_atr, fill=np.nan)
    df["bar_type"] = [_classify_bar(row) for _, row in df.iterrows()]
    df["ioi_pattern"] = (
        df["inside"]
        & df["outside"].shift(1).fillna(False)
        & df["inside"].shift(2).fillna(False)
    )
    tolerance = np.where(
        df["atr14"].notna() & (df["atr14"] > 0),
        df["atr14"] * 0.02,
        0.0,
    )
    df["micro_double"] = "none"
    df.loc[(df["low"] - prev_low).abs() <= tolerance, "micro_double"] = "MDB"
    df.loc[
        (df["micro_double"] == "none") & ((df["high"] - prev_high).abs() <= tolerance),
        "micro_double",
    ] = "MDT"
    df.loc[prev_low.isna(), "micro_double"] = "none"
    df["gap_bar"] = "none"
    df.loc[df["low"] > df["ema20"], "gap_bar"] = "bull_gap"
    df.loc[df["high"] < df["ema20"], "gap_bar"] = "bear_gap"
    df["ema_gap_count"] = _ema_gap_counts(df["gap_bar"].tolist())

    prev_range_high = df["high"].shift(1).rolling(window=5, min_periods=1).max()
    prev_range_low = df["low"].shift(1).rolling(window=5, min_periods=1).min()
    breakout_up = (df["high"] > prev_range_high) & prev_range_high.notna()
    breakout_down = (df["low"] < prev_range_low) & prev_range_low.notna()
    df["breakout_prev"] = "none"
    df.loc[breakout_up, "breakout_prev"] = "up"
    df.loc[breakout_down, "breakout_prev"] = "down"
    df.loc[breakout_up & breakout_down, "breakout_prev"] = "both"
    df["follow_through_1_2"] = _follow_through_1_2(df)

    gate_high = df["high"].shift(1).rolling(window=20, min_periods=3).max()
    gate_low = df["low"].shift(1).rolling(window=20, min_periods=3).min()
    df["gate_high"] = gate_high
    df["gate_low"] = gate_low
    broke_up = (df["high"] > gate_high) & gate_high.notna()
    broke_down = (df["low"] < gate_low) & gate_low.notna()
    df["gate_break"] = "none"
    df.loc[broke_up, "gate_break"] = "up"
    df.loc[broke_down, "gate_break"] = "down"
    df.loc[broke_up & broke_down, "gate_break"] = "both"
    df["gate_position"] = "unknown"
    df.loc[(df["close"] <= gate_high) & (df["close"] >= gate_low), "gate_position"] = "inside"
    df.loc[df["close"] > gate_high, "gate_position"] = "above"
    df.loc[df["close"] < gate_low, "gate_position"] = "below"

    analysis = df.tail(window).copy()
    analysis["seq"] = list(range(len(analysis), 0, -1))
    newest_first = analysis.iloc[::-1].reset_index(drop=True)
    rows = [_row_to_feature_dict(row) for _, row in newest_first.iterrows()]
    latest = rows[0]
    kline_table = _build_kline_table(rows)
    feature_table = _build_feature_table(rows)
    market_features = _build_market_features(rows)
    market_features_text = _build_market_features_text(market_features)
    candle_time = _to_utc_naive(newest_first.iloc[0]["date"])

    return L1FeatureResult(
        symbol=symbol,
        timeframe=timeframe,
        market=market,
        candle_time=candle_time,
        kline_table=kline_table,
        feature_table=feature_table,
        market_features_text=market_features_text,
        latest_features=latest,
        market_features=market_features,
        rows=rows,
    )


def _safe_div(numerator: Any, denominator: Any, *, fill: float) -> Any:
    result = numerator / denominator
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan).fillna(fill)
    if denominator:
        return numerator / denominator
    return fill


def _row_to_feature_dict(row: pd.Series) -> dict[str, Any]:
    patterns = []
    bar_type = str(row.get("bar_type", "other"))
    if bar_type != "other":
        patterns.append(bar_type)
    if bool(row.get("inside")):
        patterns.append("inside")
    if bool(row.get("outside")):
        patterns.append("outside")
    seq = row.get("inside_sequence", "none")
    if seq in ("ii", "iii"):
        patterns.append(seq)
    if bool(row.get("ioi_pattern", False)):
        patterns.append("ioi")
    micro_double = str(row.get("micro_double", "none"))
    if micro_double in ("MDB", "MDT"):
        patterns.append(micro_double)
    gap_bar = str(row.get("gap_bar", "none"))
    if gap_bar != "none":
        patterns.append(gap_bar)
    breakout_prev = str(row.get("breakout_prev", "none"))
    if breakout_prev != "none":
        patterns.append(f"breakout_prev_{breakout_prev}")
    gate_break = row.get("gate_break", "none")
    if gate_break != "none":
        patterns.append(f"gate_break_{gate_break}")

    return {
        "k": f"K{int(row['seq'])}",
        "time": _to_utc_naive(row["date"]).isoformat(),
        "open": _finite(row["open"]),
        "high": _finite(row["high"]),
        "low": _finite(row["low"]),
        "close": _finite(row["close"]),
        "volume": _finite(row["volume"]),
        "direction": str(row["direction"]),
        "body_pct": _finite(row["body_pct"]),
        "body_ratio": _finite(row["body_ratio"]),
        "upper_wick_pct": _finite(row["upper_wick_pct"]),
        "lower_wick_pct": _finite(row["lower_wick_pct"]),
        "close_position": _finite(row["close_position"]),
        "bar_type": bar_type,
        "inside": bool(row.get("inside", False)),
        "outside": bool(row.get("outside", False)),
        "overlap_prev_ratio": _finite(row.get("overlap_prev_ratio")),
        "inside_sequence": str(row.get("inside_sequence", "none")),
        "ioi_pattern": bool(row.get("ioi_pattern", False)),
        "micro_double": micro_double,
        "gap_bar": gap_bar,
        "ema_gap_count": int(row.get("ema_gap_count", 0) or 0),
        "breakout_prev": breakout_prev,
        "follow_through_1_2": str(row.get("follow_through_1_2", "pending")),
        "gate_high": _finite(row.get("gate_high")),
        "gate_low": _finite(row.get("gate_low")),
        "gate_break": str(row.get("gate_break", "none")),
        "gate_position": str(row.get("gate_position", "unknown")),
        "ema20": _finite(row.get("ema20")),
        "atr14": _finite(row.get("atr14")),
        "range_atr": _finite(row.get("range_atr")),
        "ema20_relation": str(row.get("ema20_relation", "unknown")),
        "ema20_gap_atr": _finite(row.get("ema20_gap_atr")),
        "atr_expand_ratio": _finite(row.get("atr_expand_ratio")),
        "patterns": patterns,
    }


def _build_kline_table(rows: list[dict[str, Any]]) -> str:
    table_rows = [
        [
            row["k"],
            row["time"],
            _fmt(row["open"]),
            _fmt(row["high"]),
            _fmt(row["low"]),
            _fmt(row["close"]),
            _fmt(row["volume"]),
            row["direction"],
        ]
        for row in rows
    ]
    return _render_table(["K", "time", "open", "high", "low", "close", "volume", "dir"], table_rows)


def _build_feature_table(rows: list[dict[str, Any]]) -> str:
    table_rows = [
        [
            row["k"],
            _fmt(row["body_pct"]),
            _fmt(row["body_ratio"]),
            _fmt(row["upper_wick_pct"]),
            _fmt(row["lower_wick_pct"]),
            row["bar_type"],
            _fmt(row["overlap_prev_ratio"]),
            row["inside_sequence"] if row["inside_sequence"] != "none" else _inside_outside(row),
            "yes" if row["ioi_pattern"] else "no",
            row["micro_double"],
            row["gap_bar"],
            row["ema_gap_count"],
            row["breakout_prev"],
            row["follow_through_1_2"],
            row["gate_break"],
            row["gate_position"],
            _fmt(row["ema20"]),
            _fmt(row["atr14"]),
            _fmt(row["range_atr"]),
            _fmt(row["atr_expand_ratio"]),
            ",".join(row["patterns"]) or "-",
        ]
        for row in rows
    ]
    return _render_table(
        [
            "K",
            "body_pct",
            "body_ratio",
            "upper_wick",
            "lower_wick",
            "bar_type",
            "overlap",
            "in/out",
            "ioi",
            "micro",
            "gap",
            "ema_gap_n",
            "breakout",
            "follow",
            "gate_break",
            "gate_pos",
            "EMA20",
            "ATR14",
            "range_ATR",
            "ATR_x",
            "patterns",
        ],
        table_rows,
    )


def _render_table(headers: list[str], rows: list[list[Any]]) -> str:
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[pos]), *(len(row[pos]) for row in string_rows))
        for pos in range(len(headers))
    ]
    header = " | ".join(headers[pos].ljust(widths[pos]) for pos in range(len(headers)))
    sep = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(row[pos].ljust(widths[pos]) for pos in range(len(headers)))
        for row in string_rows
    ]
    return "\n".join([header, sep, *body])


def _inside_outside(row: dict[str, Any]) -> str:
    labels = []
    if row["inside"]:
        labels.append("inside")
    if row["outside"]:
        labels.append("outside")
    return ",".join(labels) if labels else "none"


def _classify_bar(row: pd.Series) -> str:
    """Classify an objective candle type using PA_Agent precedence."""
    if bool(row.get("inside", False)):
        return "inside"
    if bool(row.get("outside", False)):
        return "outside_bull" if float(row["close"]) >= float(row["open"]) else "outside_bear"

    full_range = _finite(row.get("range"))
    if full_range is None or full_range <= 0:
        return "flat"
    body_ratio = _finite(row.get("body_pct"))
    close_position = _finite(row.get("close_position"))
    if body_ratio is None or close_position is None:
        return "flat"
    if body_ratio <= 0.25:
        return "doji"
    if float(row["close"]) > float(row["open"]) and close_position >= 0.65:
        return "trend_bull"
    if float(row["close"]) < float(row["open"]) and close_position <= 0.35:
        return "trend_bear"
    return "other"


def _ema_gap_counts(gaps: list[str]) -> list[int]:
    """Count consecutive current-and-older EMA gap bars of the same side."""
    counts: list[int] = []
    for idx, side in enumerate(gaps):
        if side == "none":
            counts.append(0)
            continue
        count = 0
        for older in range(idx, -1, -1):
            if gaps[older] != side:
                break
            count += 1
        counts.append(count)
    return counts


def _follow_through_1_2(df: DataFrame) -> list[str]:
    """Evaluate one-to-two newer-bar follow-through for each signal bar."""
    values: list[str] = []
    for pos, row in df.iterrows():
        newer = df.iloc[pos + 1: pos + 3]
        if newer.empty:
            values.append("pending")
            continue
        open_ = float(row["open"])
        close = float(row["close"])
        if close > open_:
            same = (newer["close"] > close).sum()
            opposite = (newer["close"] < open_).sum()
        elif close < open_:
            same = (newer["close"] < close).sum()
            opposite = (newer["close"] > open_).sum()
        else:
            values.append("pending")
            continue
        if same > 0:
            values.append("yes")
        elif opposite > 0:
            values.append("failed")
        else:
            values.append("no")
    return values


def _build_market_features(rows: list[dict[str, Any]], lookback: int = 40) -> dict[str, Any]:
    """Build PA_Agent-style objective market-structure facts from newest-first rows."""
    window = rows[: min(lookback, len(rows))]
    if not window:
        return {}

    latest = window[0]
    close = _finite(latest.get("close"))
    atr = _finite(latest.get("atr14"))
    highs = [_finite(row.get("high")) for row in window]
    lows = [_finite(row.get("low")) for row in window]
    highs_f = [value for value in highs if value is not None]
    lows_f = [value for value in lows if value is not None]
    range_high = max(highs_f) if highs_f else None
    range_low = min(lows_f) if lows_f else None
    width = (
        range_high - range_low
        if range_high is not None and range_low is not None and range_high > range_low
        else None
    )

    price_position = (
        _round_or_none((close - range_low) / width)
        if close is not None and width
        else None
    )
    range_width_atr = _round_or_none(width / atr) if width and atr and atr > 0 else None
    dist_to_high_atr = (
        _round_or_none((range_high - close) / atr)
        if close is not None and range_high is not None and atr and atr > 0
        else None
    )
    dist_to_low_atr = (
        _round_or_none((close - range_low) / atr)
        if close is not None and range_low is not None and atr and atr > 0
        else None
    )

    overlap_mean_10 = _mean_finite(
        row.get("overlap_prev_ratio") for row in window[:10]
    )
    doji_inside_ratio_10 = _mean_bool(
        row.get("bar_type") in ("doji", "inside") or bool(row.get("inside"))
        for row in window[:10]
    )
    barbwire_score = _barbwire_score(
        overlap_mean_10,
        doji_inside_ratio_10,
        range_width_atr,
        window[:10],
        atr,
    )
    swings = _find_swing_pivots(window)
    supports, resistances = _structure_levels(window, close, swings, range_low, range_high)
    breakout_events = _detect_breakout_events(window)
    hl_count = _hl_count(window)
    measured_moves = _measured_moves(range_high, range_low, close)

    return {
        "lookback_bars": len(window),
        "range_high": _round_or_none(range_high),
        "range_low": _round_or_none(range_low),
        "range_width_atr": range_width_atr,
        "price_position": price_position,
        "zone": _zone_from_position(price_position),
        "dist_to_high_atr": dist_to_high_atr,
        "dist_to_low_atr": dist_to_low_atr,
        "overlap_mean_10": _round_or_none(overlap_mean_10),
        "doji_inside_ratio_10": _round_or_none(doji_inside_ratio_10),
        "barbwire_score": _round_or_none(barbwire_score) or 0.0,
        "barbwire_candidate": barbwire_score >= 0.6,
        "swing_structure": _swing_structure(swings),
        "swings": swings[:6],
        "breakout_events": breakout_events[:6],
        "hl_count": hl_count,
        "supports": supports[:3],
        "resistances": resistances[:3],
        "invalidation_long": supports[0] if supports else None,
        "invalidation_short": resistances[0] if resistances else None,
        "measured_moves": measured_moves,
    }


def _build_market_features_text(features: dict[str, Any]) -> str:
    """Render compact Chinese market-structure facts for the Stage 1 prompt."""
    if not features:
        return "程序结构辅助特征：无可用数据"
    lines = [
        "## 程序结构辅助特征",
        "### 区间位置",
        (
            f"- 近{features['lookback_bars']}棒包络：高 {features['range_high']} / "
            f"低 {features['range_low']}；位置 {features['price_position']} "
            f"({features['zone']})"
        ),
        (
            f"- 区间宽度 {features['range_width_atr']}xATR；距上沿 "
            f"{features['dist_to_high_atr']}xATR / 距下沿 {features['dist_to_low_atr']}xATR"
        ),
        "### 重叠 / 铁丝网",
        (
            f"- overlap_mean_10={features['overlap_mean_10']}；"
            f"doji_inside_ratio_10={features['doji_inside_ratio_10']}；"
            f"barbwire_score={features['barbwire_score']}；"
            f"candidate={features['barbwire_candidate']}"
        ),
        "### 波段结构",
        f"- swing_structure={features['swing_structure']}；swings={features['swings']}",
        "### 突破 / H-L / 价位",
        f"- breakout_events={features['breakout_events']}",
        f"- hl_count={features['hl_count']}",
        f"- supports={features['supports']}；resistances={features['resistances']}",
        (
            f"- invalidation_long={features['invalidation_long']}；"
            f"invalidation_short={features['invalidation_short']}"
        ),
        "### Measured Move",
        f"- measured_moves={features['measured_moves']}",
    ]
    return "\n".join(lines)


def _mean_finite(values: Any) -> float | None:
    nums = [_finite(value) for value in values]
    nums = [value for value in nums if value is not None]
    return sum(nums) / len(nums) if nums else None


def _mean_bool(values: Any) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return sum(1 for value in vals if value) / len(vals)


def _barbwire_score(
    overlap_mean: float | None,
    doji_inside_ratio: float | None,
    range_width_atr: float | None,
    recent: list[dict[str, Any]],
    atr: float | None,
) -> float:
    del atr
    score = 0.0
    if overlap_mean is not None and overlap_mean >= 0.65:
        score += 0.4
    if doji_inside_ratio is not None and doji_inside_ratio >= 0.4:
        score += 0.2
    if range_width_atr is not None and range_width_atr <= 3.0:
        score += 0.2
    ranges = [
        (_finite(row.get("high")) or 0.0) - (_finite(row.get("low")) or 0.0)
        for row in recent
        if _finite(row.get("high")) is not None and _finite(row.get("low")) is not None
    ]
    highs = [_finite(row.get("high")) for row in recent]
    lows = [_finite(row.get("low")) for row in recent]
    highs_f = [value for value in highs if value is not None]
    lows_f = [value for value in lows if value is not None]
    avg_range = sum(ranges) / len(ranges) if ranges else None
    if avg_range and avg_range > 0 and highs_f and lows_f:
        compressed = (max(highs_f) - min(lows_f)) / avg_range
        if compressed <= 3.0:
            score += 0.2
    return min(score, 1.0)


def _zone_from_position(position: float | None) -> str:
    if position is None:
        return "unknown"
    if position <= 1 / 3:
        return "lower_third"
    if position >= 2 / 3:
        return "upper_third"
    return "middle_third"


def _find_swing_pivots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pivots: list[dict[str, Any]] = []
    for pos in range(1, len(rows) - 1):
        row = rows[pos]
        newer = rows[pos - 1]
        older = rows[pos + 1]
        high = _finite(row.get("high"))
        low = _finite(row.get("low"))
        if high is None or low is None:
            continue
        newer_high = _finite(newer.get("high"))
        older_high = _finite(older.get("high"))
        newer_low = _finite(newer.get("low"))
        older_low = _finite(older.get("low"))
        if (
            newer_high is not None
            and older_high is not None
            and high > newer_high
            and high > older_high
        ):
            pivots.append({"seq": _seq_num(row), "kind": "high", "price": _round_or_none(high)})
        if newer_low is not None and older_low is not None and low < newer_low and low < older_low:
            pivots.append({"seq": _seq_num(row), "kind": "low", "price": _round_or_none(low)})
    if not any(pivot["kind"] == "high" for pivot in pivots) and rows:
        high_row = max(rows, key=lambda item: _finite(item.get("high")) or float("-inf"))
        high = _finite(high_row.get("high"))
        if high is not None:
            pivots.append(
                {"seq": _seq_num(high_row), "kind": "high", "price": _round_or_none(high)}
            )
    if not any(pivot["kind"] == "low" for pivot in pivots) and rows:
        low_row = min(rows, key=lambda item: _finite(item.get("low")) or float("inf"))
        low = _finite(low_row.get("low"))
        if low is not None:
            pivots.append({"seq": _seq_num(low_row), "kind": "low", "price": _round_or_none(low)})
    return pivots


def _swing_structure(swings: list[dict[str, Any]]) -> str:
    highs = [s["price"] for s in swings if s["kind"] == "high" and s["price"] is not None]
    lows = [s["price"] for s in swings if s["kind"] == "low" and s["price"] is not None]
    if len(highs) < 2 or len(lows) < 2:
        return "insufficient"
    latest_high, prev_high = highs[0], highs[1]
    latest_low, prev_low = lows[0], lows[1]
    if latest_high > prev_high and latest_low > prev_low:
        return "HH+HL"
    if latest_high < prev_high and latest_low < prev_low:
        return "LL+LH"
    return "mixed"


def _detect_breakout_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for pos, row in enumerate(rows[:-2]):
        older = rows[pos + 1:]
        older_highs = [_finite(r.get("high")) for r in older]
        older_lows = [_finite(r.get("low")) for r in older]
        older_highs = [v for v in older_highs if v is not None]
        older_lows = [v for v in older_lows if v is not None]
        if not older_highs or not older_lows:
            continue
        level_high = max(older_highs)
        level_low = min(older_lows)
        high = _finite(row.get("high"))
        low = _finite(row.get("low"))
        close = _finite(row.get("close"))
        if high is None or low is None or close is None:
            continue
        newer = rows[:pos]
        if high > level_high:
            failed = any((_finite(n.get("close")) or close) < level_high for n in newer)
            event = "failed" if failed else "breakout"
            events.append(_breakout_event(level_high, "range_high", event, row, "up"))
        if low < level_low:
            failed = any((_finite(n.get("close")) or close) > level_low for n in newer)
            event = "failed" if failed else "breakout"
            events.append(_breakout_event(level_low, "range_low", event, row, "down"))
    return _dedupe_events(events)


def _breakout_event(
    level: float,
    level_kind: str,
    event: str,
    row: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    seq = _seq_num(row)
    return {
        "level_price": _round_or_none(level),
        "level_kind": level_kind,
        "event": event,
        "trigger_seq": seq,
        "bar_range": f"K{seq}-K1",
        "direction": direction,
    }


def _dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"failed": 0, "breakout": 1, "test": 2}
    best: dict[tuple[Any, Any], dict[str, Any]] = {}
    for event in events:
        key = (event.get("level_kind"), event.get("level_price"))
        old = best.get(key)
        if old is None:
            best[key] = event
            continue
        old_rank = priority.get(str(old.get("event")), 9)
        new_rank = priority.get(str(event.get("event")), 9)
        if new_rank < old_rank or (
            new_rank == old_rank
            and int(event.get("trigger_seq") or 0) < int(old.get("trigger_seq") or 0)
        ):
            best[key] = event
    return sorted(best.values(), key=lambda ev: int(ev.get("trigger_seq") or 0))


def _hl_count(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bull = 0
    bear = 0
    last_bull = None
    last_bear = None
    for pos, row in enumerate(rows[:-1]):
        older = rows[pos + 1]
        high = _finite(row.get("high"))
        low = _finite(row.get("low"))
        older_high = _finite(older.get("high"))
        older_low = _finite(older.get("low"))
        seq = _seq_num(row)
        if high is not None and older_high is not None and high > older_high:
            bull += 1
            last_bull = seq if last_bull is None or seq < last_bull else last_bull
        if low is not None and older_low is not None and low < older_low:
            bear += 1
            last_bear = seq if last_bear is None or seq < last_bear else last_bear
    return {
        "bull_count": bull,
        "bear_count": bear,
        "last_bull_trigger_seq": last_bull,
        "last_bear_trigger_seq": last_bear,
        "bull_candidate": _hl_candidate("h", bull),
        "bear_candidate": _hl_candidate("l", bear),
        "bar_range": f"K{len(rows)}-K1" if rows else "",
    }


def _hl_candidate(prefix: str, count: int) -> str:
    if count <= 0:
        return "none"
    return f"{prefix}{min(count, 3)}"


def _structure_levels(
    rows: list[dict[str, Any]],
    close: float | None,
    swings: list[dict[str, Any]],
    range_low: float | None,
    range_high: float | None,
) -> tuple[list[float], list[float]]:
    if close is None:
        return [], []
    supports = [
        _finite(s.get("price"))
        for s in swings
        if s.get("kind") == "low" and _finite(s.get("price")) is not None
    ]
    resistances = [
        _finite(s.get("price"))
        for s in swings
        if s.get("kind") == "high" and _finite(s.get("price")) is not None
    ]
    if range_low is not None:
        supports.append(range_low)
    if range_high is not None:
        resistances.append(range_high)
    lows = [_finite(row.get("low")) for row in rows[:10]]
    highs = [_finite(row.get("high")) for row in rows[:10]]
    supports.extend(value for value in lows if value is not None and value < close)
    resistances.extend(value for value in highs if value is not None and value > close)
    supports_f = sorted(
        {round(v, 8) for v in supports if v is not None and v < close},
        reverse=True,
    )
    resistances_f = sorted({round(v, 8) for v in resistances if v is not None and v > close})
    return [_round_or_none(v) for v in supports_f], [_round_or_none(v) for v in resistances_f]


def _measured_moves(
    range_high: float | None,
    range_low: float | None,
    close: float | None,
) -> list[dict[str, Any]]:
    if range_high is None or range_low is None or close is None or range_high <= range_low:
        return []
    height = range_high - range_low
    return [
        {
            "kind": "range_up",
            "reference": "range_high + range_height",
            "height": _round_or_none(height),
            "target_price": _round_or_none(range_high + height),
        },
        {
            "kind": "range_down",
            "reference": "range_low - range_height",
            "height": _round_or_none(height),
            "target_price": _round_or_none(range_low - height),
        },
    ]


def _seq_num(row: dict[str, Any]) -> int:
    k = str(row.get("k", "K0"))
    try:
        return int(k.lstrip("K"))
    except ValueError:
        return 0


def _fmt(value: Any) -> str:
    number = _finite(value)
    if number is None:
        return "-"
    if abs(number) >= 1000:
        return f"{number:.2f}"
    if abs(number) >= 1:
        return f"{number:.4f}"
    return f"{number:.6f}"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), 3)


def _to_utc_naive(value: Any) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime().astimezone(UTC).replace(tzinfo=None)
