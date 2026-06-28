"""Multi-window market background detection for price-action signals."""
from __future__ import annotations

import math

import numpy as np
from pandas import DataFrame


class MarketBackgroundAnalyzer:
    """PA_Agent-inspired deterministic background classifier.

    DataFrames are expected to be oldest-first, which matches Freqtrade strategy
    input. Each row is classified using only that row and older candles.
    """

    TOTAL_LOOKBACK = 120
    BACKGROUND_RECENT_SPLIT = 40
    TRADING_WINDOW = 40
    ENVIRONMENT_WINDOW = 20
    MOMENTUM_WINDOW = 8

    DIRECTION_BULL_THRESHOLD = 3
    DIRECTION_BEAR_THRESHOLD = -3
    TREND_BAR_DOMINANCE_RATIO = 1.5
    OVERLAP_LOW_THRESHOLD = 0.45
    OVERLAP_HIGH_THRESHOLD = 0.65

    def evaluate(self, df: DataFrame) -> DataFrame:
        """Add background columns to *df* without changing signal decisions."""
        if df.empty:
            return self._fill_defaults(df)

        missing = {"open", "high", "low", "close"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing OHLC columns for background analysis: {missing}")

        df = self._ensure_geometry(df)
        result = df.copy()

        background_dirs: list[str] = []
        trading_dirs: list[str] = []
        relationships: list[str] = []
        momentums: list[str] = []
        background_scores: list[int] = []
        trading_scores: list[int] = []
        strengths: list[float] = []

        for pos in range(len(result)):
            background_slice = self._background_slice(result, pos)
            trading_slice = self._trading_slice(result, pos)
            environment_slice = self._window_slice(result, pos, self.ENVIRONMENT_WINDOW)
            momentum_slice = self._window_slice(result, pos, self.MOMENTUM_WINDOW)

            bg_score = self._direction_score(background_slice)
            tr_score = self._direction_score(trading_slice)
            env_score = self._direction_score(environment_slice)

            bg_dir = self._score_to_direction(bg_score, len(background_slice))
            tr_dir = self._score_to_direction(tr_score, len(trading_slice))
            relationship = self._relationship(bg_dir, tr_dir)
            momentum = self._recent_momentum(momentum_slice)
            strength = self._trend_strength(bg_score, tr_score, env_score, relationship)

            background_dirs.append(bg_dir)
            trading_dirs.append(tr_dir)
            relationships.append(relationship)
            momentums.append(momentum)
            background_scores.append(bg_score)
            trading_scores.append(tr_score)
            strengths.append(strength)

        result["background_direction"] = background_dirs
        result["trading_direction"] = trading_dirs
        result["trend_relationship"] = relationships
        result["recent_momentum"] = momentums
        result["background_score"] = background_scores
        result["trading_score"] = trading_scores
        result["trend_strength"] = strengths
        return result

    def _fill_defaults(self, df: DataFrame) -> DataFrame:
        df = df.copy()
        df["background_direction"] = "unknown"
        df["trading_direction"] = "unknown"
        df["trend_relationship"] = "unknown"
        df["recent_momentum"] = "neutral"
        df["background_score"] = 0
        df["trading_score"] = 0
        df["trend_strength"] = 0.0
        return df

    def _ensure_geometry(self, df: DataFrame) -> DataFrame:
        df = df.copy()
        if "range" not in df:
            df["range"] = df["high"] - df["low"]
        if "body" not in df:
            df["body"] = (df["close"] - df["open"]).abs()
        if "body_pct" not in df:
            df["body_pct"] = np.where(df["range"] > 0, df["body"] / df["range"], 0.0)
        if "close_location" not in df:
            df["close_location"] = np.where(
                df["range"] > 0,
                (df["close"] - df["low"]) / df["range"],
                0.5,
            )
        return df

    def _background_slice(self, df: DataFrame, pos: int) -> DataFrame:
        recent_start = max(0, pos + 1 - self.BACKGROUND_RECENT_SPLIT)
        start = max(0, pos + 1 - self.TOTAL_LOOKBACK)
        return df.iloc[start:recent_start]

    def _trading_slice(self, df: DataFrame, pos: int) -> DataFrame:
        return self._window_slice(df, pos, self.TRADING_WINDOW)

    def _window_slice(self, df: DataFrame, pos: int, window: int) -> DataFrame:
        start = max(0, pos + 1 - window)
        return df.iloc[start:pos + 1]

    def _direction_score(self, window: DataFrame) -> int:
        if len(window) < 5:
            return 0

        signals = [
            self._ema_slope_signal(window),
            self._close_gravity_signal(window),
            self._swing_structure_signal(window),
            self._trend_bar_signal(window),
            self._overlap_signal(window),
        ]
        return int(sum(signals))

    def _ema_slope_signal(self, window: DataFrame) -> int:
        if "ema20" not in window or window["ema20"].isna().all():
            return 0

        lookback = min(10, len(window) - 1)
        if lookback < 1:
            return 0
        current = self._finite_float(window["ema20"].iloc[-1])
        previous = self._finite_float(window["ema20"].iloc[-1 - lookback])
        if current is None or previous is None:
            return 0

        atr = self._last_atr(window)
        threshold = 0.05 * atr if atr and atr > 0 else 0.0
        delta = current - previous
        if delta > threshold:
            return 1
        if delta < -threshold:
            return -1
        return 0

    def _close_gravity_signal(self, window: DataFrame) -> int:
        half = len(window) // 2
        if half < 2:
            return 0

        far = window["close"].iloc[:half].mean()
        near = window["close"].iloc[-half:].mean()
        atr = self._last_atr(window)
        threshold = 0.1 * atr if atr and atr > 0 else 0.0
        delta = float(near - far)
        if delta > threshold:
            return 1
        if delta < -threshold:
            return -1
        return 0

    def _swing_structure_signal(self, window: DataFrame) -> int:
        swing_highs, swing_lows = self._find_swings(window)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0

        prev_high, latest_high = swing_highs[-2], swing_highs[-1]
        prev_low, latest_low = swing_lows[-2], swing_lows[-1]
        if latest_high > prev_high and latest_low > prev_low:
            return 1
        if latest_high < prev_high and latest_low < prev_low:
            return -1
        return 0

    def _find_swings(self, window: DataFrame) -> tuple[list[float], list[float]]:
        highs: list[float] = []
        lows: list[float] = []
        for idx in range(1, len(window) - 1):
            prev_row = window.iloc[idx - 1]
            row = window.iloc[idx]
            next_row = window.iloc[idx + 1]
            high = float(row["high"])
            low = float(row["low"])
            if high > float(prev_row["high"]) and high > float(next_row["high"]):
                highs.append(high)
            if low < float(prev_row["low"]) and low < float(next_row["low"]):
                lows.append(low)

        if len(highs) < 2:
            highs = [float(window["high"].iloc[0]), float(window["high"].iloc[-1])]
        if len(lows) < 2:
            lows = [float(window["low"].iloc[0]), float(window["low"].iloc[-1])]
        return highs, lows

    def _trend_bar_signal(self, window: DataFrame) -> int:
        bull = (
            (window["close"] > window["open"])
            & (window["body_pct"] > 0.25)
            & (window["close_location"] >= 0.60)
        ).sum()
        bear = (
            (window["close"] < window["open"])
            & (window["body_pct"] > 0.25)
            & (window["close_location"] <= 0.40)
        ).sum()

        if bull + bear < 3:
            return 0
        if bull >= bear * self.TREND_BAR_DOMINANCE_RATIO:
            return 1
        if bear >= bull * self.TREND_BAR_DOMINANCE_RATIO:
            return -1
        return 0

    def _overlap_signal(self, window: DataFrame) -> int:
        overlap = self._mean_overlap_ratio(window)
        if overlap is None:
            return 0
        if overlap >= self.OVERLAP_HIGH_THRESHOLD:
            return 0
        if overlap < self.OVERLAP_LOW_THRESHOLD:
            return self._ema_slope_signal(window)
        return 0

    def _mean_overlap_ratio(self, window: DataFrame) -> float | None:
        ratios: list[float] = []
        for idx in range(1, len(window)):
            current = window.iloc[idx]
            previous = window.iloc[idx - 1]
            high = min(float(current["high"]), float(previous["high"]))
            low = max(float(current["low"]), float(previous["low"]))
            overlap = max(0.0, high - low)
            denominator = max(float(current["high"]), float(previous["high"])) - min(
                float(current["low"]), float(previous["low"])
            )
            if denominator > 0:
                ratios.append(overlap / denominator)
        if not ratios:
            return None
        return float(sum(ratios) / len(ratios))

    def _score_to_direction(self, score: int, window_len: int) -> str:
        if window_len < 8:
            return "unknown"
        if score >= self.DIRECTION_BULL_THRESHOLD:
            return "up"
        if score <= self.DIRECTION_BEAR_THRESHOLD:
            return "down"
        return "range"

    def _relationship(self, background: str, trading: str) -> str:
        if background == "unknown" or trading == "unknown":
            return "unknown"
        if background == "range" and trading == "range":
            return "range"
        if background == "range" or trading == "range":
            return "mixed"
        if background == trading:
            return "aligned"
        return "conflict"

    def _recent_momentum(self, window: DataFrame) -> str:
        if len(window) < 4:
            return "neutral"
        score = self._direction_score(window)
        if score >= 2:
            return "bullish"
        if score <= -2:
            return "bearish"
        return "neutral"

    def _trend_strength(
        self,
        background_score: int,
        trading_score: int,
        environment_score: int,
        relationship: str,
    ) -> float:
        raw = (
            0.35 * min(abs(background_score), 5)
            + 0.45 * min(abs(trading_score), 5)
            + 0.20 * min(abs(environment_score), 5)
        ) / 5.0
        if relationship == "conflict":
            raw *= 0.65
        elif relationship in ("range", "unknown"):
            raw *= 0.4
        return round(max(0.0, min(1.0, raw)), 3)

    def _last_atr(self, window: DataFrame) -> float | None:
        if "atr14" not in window:
            return None
        values = window["atr14"].dropna()
        if values.empty:
            return None
        return self._finite_float(values.iloc[-1])

    def _finite_float(self, value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number
