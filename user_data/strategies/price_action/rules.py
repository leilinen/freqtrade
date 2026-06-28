"""Pure signal-bar rules for the Price Action monitor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame


class PriceActionSignalRules:
    """Stateless DataFrame/Series rules used by the Freqtrade strategy."""

    GOOD_LONG_BODY_PCT = 0.85
    GOOD_LONG_CLOSE_LOC = 0.90
    GOOD_LONG_UPPER_SHADOW = 0.05
    GOOD_MIN_BODY_RATIO = 1.5

    ACCEPT_LONG_BODY_PCT = 0.65
    ACCEPT_LONG_CLOSE_LOC = 0.75
    ACCEPT_LONG_UPPER_SHADOW = 0.15
    ACCEPT_MIN_BODY_RATIO = 1.2

    FAIR_LONG_BODY_PCT = 0.5
    FAIR_LONG_CLOSE_LOC = 0.6

    GOOD_SHORT_CLOSE_LOC = 0.10
    GOOD_SHORT_LOWER_SHADOW = 0.05

    ACCEPT_SHORT_CLOSE_LOC = 0.25
    ACCEPT_SHORT_LOWER_SHADOW = 0.15

    FAIR_SHORT_CLOSE_LOC = 0.4

    EMA_MAX_GAP = 2.0
    BULL_STRENGTH_LONG_MIN = 0.5
    BULL_STRENGTH_SHORT_MAX = 0.5
    FOLLOW_THROUGH_WINDOW = 3

    SURPRISE_LOOKBACK = 20
    SURPRISE_MIN_BODY_PCT = 0.5

    RANGE_LOOKBACK = 40
    RANGE_MIN_PERIODS = 10
    OVERLAP_LOOKBACK = 10
    OVERLAP_MIN_PERIODS = 3
    BARBWIRE_OVERLAP_THRESHOLD = 0.65
    BARBWIRE_DOJI_INSIDE_THRESHOLD = 0.4
    BARBWIRE_RANGE_ATR_MAX = 3.0
    BARBWIRE_SCORE_THRESHOLD = 0.6
    MICRO_DOUBLE_ATR_TOLERANCE = 0.02
    BREAKOUT_LOOKBACK = 5

    def calc_basic_indicators(self, df: DataFrame) -> DataFrame:
        """Calculate candle geometry indicators."""
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]

        df["body_pct"] = np.where(df["range"] > 0, df["body"] / df["range"], 0.0)
        df["close_location"] = np.where(
            df["range"] > 0, (df["close"] - df["low"]) / df["range"], 0.5
        )

        df["upper_shadow"] = df["high"] - np.maximum(df["close"], df["open"])
        df["lower_shadow"] = np.minimum(df["close"], df["open"]) - df["low"]
        df["upper_shadow_pct"] = np.where(
            df["range"] > 0, df["upper_shadow"] / df["range"], 0.0
        )
        df["lower_shadow_pct"] = np.where(
            df["range"] > 0, df["lower_shadow"] / df["range"], 0.0
        )

        df["is_bull"] = df["close"] > df["open"]

        df["median_body_20"] = df["body"].rolling(window=20, min_periods=10).median()
        df["body_ratio"] = np.where(
            df["median_body_20"] > 0, df["body"] / df["median_body_20"], 0.0
        )

        df["is_trend_bar"] = df["body_pct"] >= 0.5
        df["is_doji"] = df["body_pct"] < 0.1

        return df

    def calc_ema_atr(self, df: DataFrame) -> DataFrame:
        """Calculate EMA20 and ATR14."""
        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["atr14"] = ta.ATR(df, timeperiod=14)
        df["ema20_position"] = np.where(
            df["atr14"] > 0, (df["close"] - df["ema20"]) / df["atr14"], 0.0
        )
        return df

    def detect_special_bars(self, df: DataFrame) -> DataFrame:
        """Detect simple special candle types."""
        df["is_inside"] = (
            (df["high"] <= df["high"].shift(1)) & (df["low"] >= df["low"].shift(1))
        )

        df["is_engulfing"] = (
            (df["high"] > df["high"].shift(1))
            & (df["low"] < df["low"].shift(1))
            & (df["body"] > df["body"].shift(1))
        )

        df["prev_max_range_20"] = (
            df["range"].shift(1).rolling(window=self.SURPRISE_LOOKBACK, min_periods=5).max()
        )
        df["is_surprise"] = (
            (df["range"] > df["prev_max_range_20"])
            & (df["body_pct"] >= self.SURPRISE_MIN_BODY_PCT)
        )

        prev_bull = df["is_bull"].shift(1).fillna(False).astype(bool)
        curr_bull = df["is_bull"].astype(bool)
        prev_open = df["open"].shift(1)

        df["is_2k_reversal_long"] = (~prev_bull) & curr_bull & (df["close"] > prev_open)
        df["is_2k_reversal_short"] = prev_bull & (~curr_bull) & (df["close"] < prev_open)
        df["is_2k_reversal"] = df["is_2k_reversal_long"] | df["is_2k_reversal_short"]

        return df

    def classify_signal_quality(self, df: DataFrame) -> DataFrame:
        """Classify signal-bar quality."""
        df["signal_quality"] = "none"
        df["signal_direction"] = "none"

        is_bull = df["is_bull"]

        good_long = (
            is_bull
            & (df["body_pct"] >= self.GOOD_LONG_BODY_PCT)
            & (df["close_location"] >= self.GOOD_LONG_CLOSE_LOC)
            & (df["upper_shadow_pct"] <= self.GOOD_LONG_UPPER_SHADOW)
            & (df["body_ratio"] >= self.GOOD_MIN_BODY_RATIO)
        )
        df.loc[good_long, "signal_quality"] = "good"
        df.loc[good_long, "signal_direction"] = "long"

        accept_long = (
            is_bull
            & (df["signal_quality"] == "none")
            & (df["body_pct"] >= self.ACCEPT_LONG_BODY_PCT)
            & (df["close_location"] >= self.ACCEPT_LONG_CLOSE_LOC)
            & (df["upper_shadow_pct"] <= self.ACCEPT_LONG_UPPER_SHADOW)
            & (df["body_ratio"] >= self.ACCEPT_MIN_BODY_RATIO)
        )
        df.loc[accept_long, "signal_quality"] = "acceptable"
        df.loc[accept_long, "signal_direction"] = "long"

        fair_long = (
            is_bull
            & (df["signal_quality"] == "none")
            & (df["body_pct"] >= self.FAIR_LONG_BODY_PCT)
            & (df["close_location"] >= self.FAIR_LONG_CLOSE_LOC)
        )
        df.loc[fair_long, "signal_quality"] = "fair"
        df.loc[fair_long, "signal_direction"] = "long"

        is_bear = ~df["is_bull"].astype(bool)

        good_short = (
            is_bear
            & (df["body_pct"] >= self.GOOD_LONG_BODY_PCT)
            & (df["close_location"] <= self.GOOD_SHORT_CLOSE_LOC)
            & (df["lower_shadow_pct"] <= self.GOOD_SHORT_LOWER_SHADOW)
            & (df["body_ratio"] >= self.GOOD_MIN_BODY_RATIO)
        )
        df.loc[good_short, "signal_quality"] = "good"
        df.loc[good_short, "signal_direction"] = "short"

        accept_short = (
            is_bear
            & (df["signal_quality"] == "none")
            & (df["body_pct"] >= self.ACCEPT_LONG_BODY_PCT)
            & (df["close_location"] <= self.ACCEPT_SHORT_CLOSE_LOC)
            & (df["lower_shadow_pct"] <= self.ACCEPT_SHORT_LOWER_SHADOW)
            & (df["body_ratio"] >= self.ACCEPT_MIN_BODY_RATIO)
        )
        df.loc[accept_short, "signal_quality"] = "acceptable"
        df.loc[accept_short, "signal_direction"] = "short"

        fair_short = (
            is_bear
            & (df["signal_quality"] == "none")
            & (df["body_pct"] >= self.FAIR_LONG_BODY_PCT)
            & (df["close_location"] <= self.FAIR_SHORT_CLOSE_LOC)
        )
        df.loc[fair_short, "signal_quality"] = "fair"
        df.loc[fair_short, "signal_direction"] = "short"

        return df

    def evaluate_context(self, df: DataFrame) -> DataFrame:
        """Calculate EMA and short-window directional context."""
        df["above_ema20"] = df["close"] > df["ema20"]
        df["ema_gap"] = np.where(
            df["atr14"] > 0, abs(df["close"] - df["ema20"]) / df["atr14"], 0.0
        )

        bull_body = np.where(df["is_bull"], df["body"], 0.0)
        bear_body = np.where(~df["is_bull"].astype(bool), df["body"], 0.0)

        bull_sum_5 = pd.Series(bull_body).rolling(window=5, min_periods=3).sum()
        bear_sum_5 = pd.Series(bear_body).rolling(window=5, min_periods=3).sum()
        total_body_5 = bull_sum_5 + bear_sum_5

        df["bull_strength_5"] = np.where(
            total_body_5 > 0, bull_sum_5 / total_body_5, 0.5
        )

        return df

    def calc_structure_context(self, df: DataFrame) -> DataFrame:
        """Calculate deterministic structure context used to suppress noisy signals."""
        range_high = df["high"].rolling(
            window=self.RANGE_LOOKBACK, min_periods=self.RANGE_MIN_PERIODS
        ).max()
        range_low = df["low"].rolling(
            window=self.RANGE_LOOKBACK, min_periods=self.RANGE_MIN_PERIODS
        ).min()
        range_width = range_high - range_low

        df["range_high_40"] = range_high
        df["range_low_40"] = range_low
        df["range_width_atr_40"] = np.where(
            (range_width > 0) & (df["atr14"] > 0), range_width / df["atr14"], np.nan
        )
        df["range_position_40"] = np.where(
            range_width > 0, (df["close"] - range_low) / range_width, np.nan
        )
        df["range_zone"] = "unknown"
        df.loc[df["range_position_40"] <= 1 / 3, "range_zone"] = "lower"
        df.loc[df["range_position_40"] >= 2 / 3, "range_zone"] = "upper"
        df.loc[
            (df["range_position_40"] > 1 / 3) & (df["range_position_40"] < 2 / 3),
            "range_zone",
        ] = "middle"

        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        overlap = (np.minimum(df["high"], prev_high) - np.maximum(df["low"], prev_low)).clip(
            lower=0
        )
        overlap_denominator = np.maximum(df["high"], prev_high) - np.minimum(
            df["low"], prev_low
        )
        df["overlap_prev_ratio"] = np.where(
            overlap_denominator > 0, overlap / overlap_denominator, np.nan
        )
        df["overlap_mean_10"] = df["overlap_prev_ratio"].rolling(
            window=self.OVERLAP_LOOKBACK, min_periods=self.OVERLAP_MIN_PERIODS
        ).mean()

        doji_inside = (df["is_doji"] | df["is_inside"]).astype(float)
        df["doji_inside_ratio_10"] = doji_inside.rolling(
            window=self.OVERLAP_LOOKBACK, min_periods=self.OVERLAP_MIN_PERIODS
        ).mean()

        barbwire_score = pd.Series(0.0, index=df.index)
        barbwire_score += np.where(
            df["overlap_mean_10"] >= self.BARBWIRE_OVERLAP_THRESHOLD, 0.4, 0.0
        )
        barbwire_score += np.where(
            df["doji_inside_ratio_10"] >= self.BARBWIRE_DOJI_INSIDE_THRESHOLD, 0.2, 0.0
        )
        barbwire_score += np.where(
            df["range_width_atr_40"] <= self.BARBWIRE_RANGE_ATR_MAX, 0.2, 0.0
        )
        avg_range_10 = df["range"].rolling(
            window=self.OVERLAP_LOOKBACK, min_periods=self.OVERLAP_MIN_PERIODS
        ).mean()
        recent_high_10 = df["high"].rolling(
            window=self.OVERLAP_LOOKBACK, min_periods=self.OVERLAP_MIN_PERIODS
        ).max()
        recent_low_10 = df["low"].rolling(
            window=self.OVERLAP_LOOKBACK, min_periods=self.OVERLAP_MIN_PERIODS
        ).min()
        compressed = np.where(
            avg_range_10 > 0, (recent_high_10 - recent_low_10) / avg_range_10, np.nan
        )
        barbwire_score += np.where(compressed <= 3.0, 0.2, 0.0)
        df["barbwire_score"] = barbwire_score.clip(upper=1.0)
        df["is_barbwire"] = df["barbwire_score"] >= self.BARBWIRE_SCORE_THRESHOLD

        df["is_outside"] = (
            (df["high"] >= prev_high) & (df["low"] <= prev_low)
        )
        df["inside_sequence"] = "none"
        df.loc[df["is_inside"] & df["is_inside"].shift(1).fillna(False), "inside_sequence"] = "ii"
        df.loc[
            df["is_inside"]
            & df["is_inside"].shift(1).fillna(False)
            & df["is_inside"].shift(2).fillna(False),
            "inside_sequence",
        ] = "iii"
        df["is_ioi"] = (
            df["is_inside"].shift(2).fillna(False)
            & df["is_outside"].shift(1).fillna(False)
            & df["is_inside"]
        )

        tolerance = np.where(
            df["atr14"] > 0,
            df["atr14"] * self.MICRO_DOUBLE_ATR_TOLERANCE,
            df["close"].abs() * 0.0002,
        )
        df["is_mdb"] = (df["low"] - prev_low).abs() <= tolerance
        df["is_mdt"] = (df["high"] - prev_high).abs() <= tolerance
        df["micro_double"] = "none"
        df.loc[df["is_mdb"], "micro_double"] = "mdb"
        df.loc[df["is_mdt"], "micro_double"] = "mdt"

        prev_max_high = df["high"].shift(1).rolling(
            window=self.BREAKOUT_LOOKBACK, min_periods=2
        ).max()
        prev_min_low = df["low"].shift(1).rolling(
            window=self.BREAKOUT_LOOKBACK, min_periods=2
        ).min()
        broke_up = df["high"] > prev_max_high
        broke_down = df["low"] < prev_min_low
        df["breakout_prev_5"] = "none"
        df.loc[broke_up, "breakout_prev_5"] = "up"
        df.loc[broke_down, "breakout_prev_5"] = "down"
        df.loc[broke_up & broke_down, "breakout_prev_5"] = "both"

        df["swing_structure"] = "insufficient"
        higher_high = df["high"] > df["high"].shift(5)
        higher_low = df["low"] > df["low"].shift(5)
        lower_high = df["high"] < df["high"].shift(5)
        lower_low = df["low"] < df["low"].shift(5)
        df.loc[higher_high & higher_low, "swing_structure"] = "HH+HL"
        df.loc[lower_high & lower_low, "swing_structure"] = "LL+LH"
        df.loc[
            ~(higher_high & higher_low)
            & ~(lower_high & lower_low)
            & df["high"].shift(5).notna(),
            "swing_structure",
        ] = "mixed"

        return df

    def detect_ema20_cross(self, df: DataFrame) -> DataFrame:
        """Mark EMA20 cross signals."""
        prev_above = (df["close"].shift(1) > df["ema20"].shift(1)).fillna(False)
        prev_below = (df["close"].shift(1) < df["ema20"].shift(1)).fillna(False)
        curr_above = df["close"] > df["ema20"]
        curr_below = df["close"] < df["ema20"]
        df["ema20_cross"] = "none"
        df.loc[prev_below & curr_above, "ema20_cross"] = "long"
        df.loc[prev_above & curr_below, "ema20_cross"] = "short"
        return df

    def ema_context_ok(self, row: pd.Series, direction: str) -> bool:
        """Return whether EMA20 context agrees with signal direction."""
        ema_gap = row.get("ema_gap", 0)
        if pd.isna(ema_gap):
            return True

        if ema_gap > self.EMA_MAX_GAP:
            return False

        above_ema = row.get("above_ema20", False)
        bull_strength = row.get("bull_strength_5", 0.5)
        if pd.isna(bull_strength):
            bull_strength = 0.5

        if direction == "long":
            return above_ema and bull_strength >= self.BULL_STRENGTH_LONG_MIN
        return (not above_ema) and bull_strength <= self.BULL_STRENGTH_SHORT_MAX

    def has_special_context(self, row: pd.Series) -> bool:
        """Return whether row has a structural feature that can support weaker signals."""
        return bool(
            row.get("is_inside", False)
            or row.get("is_engulfing", False)
            or row.get("is_surprise", False)
            or row.get("is_2k_reversal", False)
            or row.get("is_ioi", False)
            or row.get("inside_sequence", "none") != "none"
            or row.get("micro_double", "none") != "none"
            or row.get("breakout_prev_5", "none") != "none"
        )

    def context_filter_ok(self, row: pd.Series) -> bool:
        """Suppress signals in noisy middle-range context."""
        quality = row.get("signal_quality", "none")
        if quality == "none":
            return False
        if bool(row.get("is_barbwire", False)):
            return False

        range_zone = row.get("range_zone", "unknown")
        has_context = self.has_special_context(row)
        has_breakout = row.get("breakout_prev_5", "none") != "none"
        is_range_edge = range_zone in ("lower", "upper")

        if quality == "fair":
            return bool(has_context or has_breakout or is_range_edge)
        if quality == "acceptable" and range_zone == "middle":
            return bool(has_context or has_breakout)
        return True

    def candidate_signal_ok(self, row: pd.Series) -> bool:
        """Return whether a row is a signal-bar candidate worth tracking."""
        quality = row.get("signal_quality", "none")
        direction = row.get("signal_direction", "none")

        if quality == "none":
            return False
        if not self.ema_context_ok(row, direction):
            return False
        if not self.context_filter_ok(row):
            return False
        if quality in ("good", "acceptable"):
            return True
        if quality == "fair":
            return True
        return False

    def follow_through_ok(self, candidate: pd.Series, future: DataFrame) -> bool:
        """Confirm signal-bar extreme breaks and is not quickly rejected."""
        if future.empty:
            return False

        direction = candidate.get("signal_direction", "none")
        high = candidate.get("high")
        low = candidate.get("low")
        close = candidate.get("close")
        if pd.isna(high) or pd.isna(low) or pd.isna(close):
            return False

        midpoint = (float(high) + float(low)) / 2
        closes = future["close"]

        if direction == "long":
            triggered = (future["high"] > float(high)).any()
            if not triggered:
                return False
            rejected = (closes < midpoint).sum() >= 2 or closes.iloc[-1] < midpoint
            continued = (closes > float(close)).any()
            return bool(continued and not rejected)

        if direction == "short":
            triggered = (future["low"] < float(low)).any()
            if not triggered:
                return False
            rejected = (closes > midpoint).sum() >= 2 or closes.iloc[-1] > midpoint
            continued = (closes < float(close)).any()
            return bool(continued and not rejected)

        return False

    def get_bar_types(self, row: pd.Series) -> list[str]:
        """Collect special candle type tags."""
        types = [
            tag
            for field, tag in (
                ("is_surprise", "surprise"),
                ("is_engulfing", "engulfing"),
                ("is_inside", "inside"),
                ("is_2k_reversal", "2k_reversal"),
                ("is_doji", "doji"),
                ("is_ioi", "ioi"),
                ("is_barbwire", "barbwire"),
            )
            if row.get(field, False)
        ]

        inside_sequence = row.get("inside_sequence", "none")
        if inside_sequence in ("ii", "iii"):
            types.append(inside_sequence)

        micro_double = row.get("micro_double", "none")
        if micro_double in ("mdb", "mdt"):
            types.append(micro_double)

        breakout = row.get("breakout_prev_5", "none")
        types.extend(
            tag
            for value, tag in (("up", "breakout_up"), ("down", "breakout_down"))
            if breakout in (value, "both")
        )

        range_zone = row.get("range_zone", "unknown")
        if range_zone in ("lower", "upper"):
            types.append("range_edge")
        elif range_zone == "middle":
            types.append("range_middle")

        return types
