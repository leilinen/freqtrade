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
    latest_features: dict[str, Any]
    rows: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market": self.market,
            "candle_time": self.candle_time.isoformat(),
            "latest_features": self.latest_features,
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
    df["ema20_gap_atr"] = _safe_div((df["close"] - df["ema20"]).abs(), df["atr14"], fill=np.nan)
    prev_atr = df["atr14"].shift(1)
    df["atr_expand_ratio"] = _safe_div(df["atr14"], prev_atr, fill=np.nan)

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
    candle_time = _to_utc_naive(newest_first.iloc[0]["date"])

    return L1FeatureResult(
        symbol=symbol,
        timeframe=timeframe,
        market=market,
        candle_time=candle_time,
        kline_table=kline_table,
        feature_table=feature_table,
        latest_features=latest,
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
    if bool(row.get("inside")):
        patterns.append("inside")
    if bool(row.get("outside")):
        patterns.append("outside")
    seq = row.get("inside_sequence", "none")
    if seq in ("ii", "iii"):
        patterns.append(seq)
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
        "inside": bool(row.get("inside", False)),
        "outside": bool(row.get("outside", False)),
        "inside_sequence": str(row.get("inside_sequence", "none")),
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
            row["inside_sequence"] if row["inside_sequence"] != "none" else _inside_outside(row),
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
            "in/out",
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


def _to_utc_naive(value: Any) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime().astimezone(UTC).replace(tzinfo=None)
