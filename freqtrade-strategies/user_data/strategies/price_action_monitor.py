"""
价格行为信号K线盯盘策略 (Price Action Signal Bar Monitor)

基于 Al Brooks 价格行为学，量化识别信号K线并通过独立 TG Bot 推送通知。
纯盯盘工具，不执行实际交易。信号数据写入 PostgreSQL，通知通过 HTTP POST 发送到 tg-bot。

量化规则来源: signal-bar-spec.md, breakout-scoring.md

配置要求 (config JSON):
  pa_db_url: PostgreSQL 连接串
    例: "postgresql://postgres:postgres@localhost:15432/freqtrade_monitor"
  tg_api_url: TG Bot HTTP API 地址
    例: "http://tg-bot:8090"
"""

import json
import logging
import io
import atexit
import time as _time
from datetime import datetime, timezone

import requests as http_requests

import numpy as np
import pandas as pd
from pandas import DataFrame
from sqlalchemy import (
    Column, Integer, BigInteger, String, Float, Boolean, DateTime,
    UniqueConstraint, create_engine, func, text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, sessionmaker

from freqtrade.strategy import IStrategy
import talib.abstract as ta

logger = logging.getLogger(__name__)

# ================================================================
# SQLAlchemy ORM — 独立 Base，不碰 freqtrade 内部
# ================================================================

class _Base(DeclarativeBase):
    pass


class WatchPair(_Base):
    """盯盘标的配置表"""
    __tablename__ = "watch_pair"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    market: Mapped[str] = mapped_column(String, default="crypto")
    display_name: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PaSignal(_Base):
    """信号K线记录表"""
    __tablename__ = "pa_signal"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # 品种信息
    market: Mapped[str] = mapped_column(String, nullable=False)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    timeframe: Mapped[str] = mapped_column(String, nullable=False)
    candle_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    # 信号
    signal_type: Mapped[str] = mapped_column(String, nullable=False)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    quality: Mapped[str] = mapped_column(String, nullable=False)
    # K线 OHLCV
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    # 量化指标
    body_pct: Mapped[float] = mapped_column(Float)
    close_location: Mapped[float] = mapped_column(Float)
    body_ratio: Mapped[float] = mapped_column(Float)
    upper_shadow_pct: Mapped[float] = mapped_column(Float)
    lower_shadow_pct: Mapped[float] = mapped_column(Float)
    # 背景
    ema20: Mapped[float] = mapped_column(Float)
    atr14: Mapped[float] = mapped_column(Float)
    ema20_position: Mapped[float] = mapped_column(Float)
    ema_gap: Mapped[float] = mapped_column(Float)
    bull_strength_5: Mapped[float] = mapped_column(Float)
    # 特殊K线 (JSON array string)
    bar_types: Mapped[str] = mapped_column(String, default="[]")
    # 交易参数
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    target_price: Mapped[float] = mapped_column(Float)
    # 通知文本
    reason: Mapped[str] = mapped_column(String, default="")

    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "candle_time", "signal_type", "direction",
            name="ux_pa_signal_identity",
        ),
    )


class PaKline(_Base):
    """盯盘 K 线原始 OHLCV 记录表"""
    __tablename__ = "pa_kline"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    timeframe: Mapped[str] = mapped_column(String, nullable=False)
    candle_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "candle_time", name="ux_pa_kline_identity"),
    )


DEFAULT_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]


class PriceActionMonitor(IStrategy):
    """
    价格行为信号K线盯盘策略。

    检测信号K线质量分级（好/可接受/一般）和特殊K线类型
    （内包线、吞噬线、惊喜K线、2K反转），
    将信号写入 PostgreSQL 并通过 HTTP POST 通知独立 TG Bot 服务。

    盯盘标的从 PG watch_pair 表动态读取。

    必须以 dry_run: true 模式运行。
    config 中需设置 pa_db_url。
    """

    INTERFACE_VERSION = 3

    # 不执行交易
    can_short: bool = False
    minimal_roi = {}
    stoploss = -0.99
    trailing_stop = False
    use_exit_signal = False
    ignore_roi_if_entry_signal = True

    # 基础配置（可通过 config 覆盖）
    timeframe = "1h"
    process_only_new_candles = True
    startup_candle_count: int = 30

    # --- 信号质量阈值 (V2, 经 6 个月数据评估优化) ---
    # V1: good 胜率 45.0% → V2: 51.9%, 净收益 +0.297 ATR, 盈亏比 1.38

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

    # EMA20 背景过滤
    EMA_MAX_GAP = 2.0
    BULL_STRENGTH_LONG_MIN = 0.5
    BULL_STRENGTH_SHORT_MAX = 0.5

    # 特殊K线参数
    SURPRISE_LOOKBACK = 20
    SURPRISE_MIN_BODY_PCT = 0.5

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._pg_engine = None
        self._pg_session_factory = None

    # ================================================================
    # 生命周期回调
    # ================================================================

    def bot_start(self, **kwargs) -> None:
        """初始化 PostgreSQL 连接和表结构。"""
        db_url = self.config.get("pa_db_url")
        if not db_url:
            logger.warning("pa_db_url not set in config, PG persistence disabled")
            return

        self._pg_engine = create_engine(db_url)
        _Base.metadata.create_all(self._pg_engine)
        self._pg_session_factory = sessionmaker(bind=self._pg_engine)
        atexit.register(self._cleanup)

        # 初始化默认标的
        self._init_default_pairs()

        logger.info("PG persistence initialized: %s", db_url)

    def _cleanup(self) -> None:
        if self._pg_engine:
            self._pg_engine.dispose()

    def _init_default_pairs(self) -> None:
        """将默认标的写入 watch_pair 表（如不存在）。"""
        exchange_name = self.config.get("exchange", {}).get("name", "")
        if exchange_name == "ashare":
            pairs = self.config.get("exchange", {}).get("pair_whitelist", [])
            market = "ashare"
        else:
            pairs = DEFAULT_PAIRS
            market = "crypto"

        with self._pg_session_factory() as session:
            for symbol in pairs:
                existing = session.query(WatchPair).filter_by(symbol=symbol).first()
                if not existing:
                    display_name = None
                    if market == "ashare":
                        from freqtrade.exchange.ashare import fetch_ashare_name
                        display_name = fetch_ashare_name(symbol)
                        _time.sleep(1)
                    session.add(
                        WatchPair(
                            symbol=symbol,
                            enabled=True,
                            market=market,
                            display_name=display_name,
                        )
                    )
            session.commit()

    # ================================================================
    # 策略接口
    # ================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算所有价格行为指标。"""
        dataframe = self._calc_basic_indicators(dataframe)
        dataframe = self._calc_ema_atr(dataframe)
        dataframe = self._detect_special_bars(dataframe)
        dataframe = self._classify_signal_quality(dataframe)
        dataframe = self._evaluate_context(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """检测最新K线信号，发送通知并写入 PG。"""
        if len(dataframe) == 0:
            return dataframe

        # 只在 live/dry_run 模式下发送通知
        if self.dp and self.dp.runmode.value in ("live", "dry_run"):
            last = dataframe.iloc[-1]
            pair = metadata.get("pair", "Unknown")
            quality = last.get("signal_quality", "none")
            direction = last.get("signal_direction", "none")
            if quality != "none":
                logger.info("Scan %s %s: %s %s", pair, self.timeframe, direction, quality)
            else:
                logger.debug("Scan %s %s: no signal", pair, self.timeframe)
            self._check_and_notify(pair, last, dataframe)
            self._persist_kline(pair, dataframe)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """空实现，不产生退出信号。"""
        return dataframe

    # ================================================================
    # 指标计算
    # ================================================================

    def _calc_basic_indicators(self, df: DataFrame) -> DataFrame:
        """K线基础指标 — signal-bar-spec.md §2.1"""
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

    def _calc_ema_atr(self, df: DataFrame) -> DataFrame:
        """EMA20 和 ATR14 — signal-bar-spec.md §4.3"""
        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["atr14"] = ta.ATR(df, timeperiod=14)
        df["ema20_position"] = np.where(
            df["atr14"] > 0, (df["close"] - df["ema20"]) / df["atr14"], 0.0
        )
        return df

    def _detect_special_bars(self, df: DataFrame) -> DataFrame:
        """特殊K线类型检测 — signal-bar-spec.md §3"""
        # Inside Bar
        df["is_inside"] = (
            (df["high"] <= df["high"].shift(1)) & (df["low"] >= df["low"].shift(1))
        )

        # Engulfing
        df["is_engulfing"] = (
            (df["high"] > df["high"].shift(1))
            & (df["low"] < df["low"].shift(1))
            & (df["body"] > df["body"].shift(1))
        )

        # Surprise Bar
        df["prev_max_range_20"] = (
            df["range"].shift(1).rolling(window=self.SURPRISE_LOOKBACK, min_periods=5).max()
        )
        df["is_surprise"] = (
            (df["range"] > df["prev_max_range_20"])
            & (df["body_pct"] >= self.SURPRISE_MIN_BODY_PCT)
        )

        # 2K Reversal
        prev_bull = df["is_bull"].shift(1).fillna(False).astype(bool)
        curr_bull = df["is_bull"].astype(bool)
        prev_open = df["open"].shift(1)

        df["is_2k_reversal_long"] = (~prev_bull) & curr_bull & (df["close"] > prev_open)
        df["is_2k_reversal_short"] = prev_bull & (~curr_bull) & (df["close"] < prev_open)
        df["is_2k_reversal"] = df["is_2k_reversal_long"] | df["is_2k_reversal_short"]

        return df

    def _classify_signal_quality(self, df: DataFrame) -> DataFrame:
        """信号K线质量分级 — V2 收紧阈值 + body_ratio 过滤"""
        df["signal_quality"] = "none"
        df["signal_direction"] = "none"

        is_bull = df["is_bull"]

        # Good long
        good_long = (
            is_bull
            & (df["body_pct"] >= self.GOOD_LONG_BODY_PCT)
            & (df["close_location"] >= self.GOOD_LONG_CLOSE_LOC)
            & (df["upper_shadow_pct"] <= self.GOOD_LONG_UPPER_SHADOW)
            & (df["body_ratio"] >= self.GOOD_MIN_BODY_RATIO)
        )
        df.loc[good_long, "signal_quality"] = "good"
        df.loc[good_long, "signal_direction"] = "long"

        # Acceptable long
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

        # Fair long
        fair_long = (
            is_bull
            & (df["signal_quality"] == "none")
            & (df["body_pct"] >= self.FAIR_LONG_BODY_PCT)
            & (df["close_location"] >= self.FAIR_LONG_CLOSE_LOC)
        )
        df.loc[fair_long, "signal_quality"] = "fair"
        df.loc[fair_long, "signal_direction"] = "long"

        # 做空
        is_bear = ~df["is_bull"].astype(bool)

        # Good short
        good_short = (
            is_bear
            & (df["body_pct"] >= self.GOOD_LONG_BODY_PCT)
            & (df["close_location"] <= self.GOOD_SHORT_CLOSE_LOC)
            & (df["lower_shadow_pct"] <= self.GOOD_SHORT_LOWER_SHADOW)
            & (df["body_ratio"] >= self.GOOD_MIN_BODY_RATIO)
        )
        df.loc[good_short, "signal_quality"] = "good"
        df.loc[good_short, "signal_direction"] = "short"

        # Acceptable short
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

        # Fair short
        fair_short = (
            is_bear
            & (df["signal_quality"] == "none")
            & (df["body_pct"] >= self.FAIR_LONG_BODY_PCT)
            & (df["close_location"] <= self.FAIR_SHORT_CLOSE_LOC)
        )
        df.loc[fair_short, "signal_quality"] = "fair"
        df.loc[fair_short, "signal_direction"] = "short"

        return df

    def _evaluate_context(self, df: DataFrame) -> DataFrame:
        """背景评估指标 — signal-bar-spec.md §4"""
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

    # ================================================================
    # 通知 + PG 持久化
    # ================================================================

    def _ema_context_ok(self, row: pd.Series, direction: str) -> bool:
        """EMA20 背景过滤: 方向一致性 + ema_gap 限制。"""
        ema_gap = row.get("ema_gap", 0)
        if pd.isna(ema_gap):
            return True

        # 距 EMA20 太远 (>2 ATR) 不参与
        if ema_gap > self.EMA_MAX_GAP:
            return False

        above_ema = row.get("above_ema20", False)
        bull_strength = row.get("bull_strength_5", 0.5)
        if pd.isna(bull_strength):
            bull_strength = 0.5

        if direction == "long":
            return above_ema and bull_strength >= self.BULL_STRENGTH_LONG_MIN
        else:
            return (not above_ema) and bull_strength <= self.BULL_STRENGTH_SHORT_MAX

    def _check_and_notify(self, pair: str, last: pd.Series, dataframe: DataFrame) -> None:
        """检查最新K线，写入 PG 并 POST 通知到 tg-bot。EMA20 背景过滤。"""
        quality = last.get("signal_quality", "none")
        direction = last.get("signal_direction", "none")

        if quality == "none":
            return

        # EMA20 背景方向过滤
        if not self._ema_context_ok(last, direction):
            logger.debug("Signal filtered by EMA context: %s %s %s", pair, direction, quality)
            return

        should_notify = False

        if quality in ("good", "acceptable"):
            should_notify = True
        elif quality == "fair":
            should_notify = bool(
                last.get("is_inside", False)
                or last.get("is_engulfing", False)
                or last.get("is_surprise", False)
                or last.get("is_2k_reversal", False)
            )

        if should_notify:
            msg = self._format_signal_message(pair, last)
            self._save_signal(pair, last, msg)
            self._notify_tg_bot(pair, last, dataframe)

    def _get_bar_types(self, row: pd.Series) -> list[str]:
        """收集当前K线的特殊类型标签。"""
        types = []
        if row.get("is_surprise", False):
            types.append("surprise")
        if row.get("is_engulfing", False):
            types.append("engulfing")
        if row.get("is_inside", False):
            types.append("inside")
        if row.get("is_2k_reversal", False):
            types.append("2k_reversal")
        if row.get("is_doji", False):
            types.append("doji")
        return types

    def _save_signal(self, pair: str, row: pd.Series, reason: str) -> None:
        """将信号写入 PostgreSQL。"""
        if not self._pg_session_factory:
            return

        quality = row.get("signal_quality", "none")
        direction = row.get("signal_direction", "none")
        entry_price = float(row.get("close", 0))

        # 止损和目标位 — signal-bar-spec.md §5
        if direction == "long":
            stop_loss = float(row.get("low", 0))
            risk = entry_price - stop_loss
            target_price = entry_price + 2 * risk if risk > 0 else entry_price
        else:
            stop_loss = float(row.get("high", 0))
            risk = stop_loss - entry_price
            target_price = entry_price - 2 * risk if risk > 0 else entry_price

        # K线时间
        candle_time = row.get("date", None)
        if candle_time is None:
            candle_time = datetime.now(timezone.utc)

        signal = PaSignal(
            market="crypto",
            symbol=pair,
            timeframe=self.timeframe,
            candle_time=candle_time,
            signal_type=f"signal_bar_{quality}",
            direction=direction,
            quality=quality,
            open=float(row.get("open", 0)),
            high=float(row.get("high", 0)),
            low=float(row.get("low", 0)),
            close=float(row.get("close", 0)),
            volume=float(row.get("volume", 0)),
            body_pct=float(row.get("body_pct", 0)),
            close_location=float(row.get("close_location", 0)),
            body_ratio=float(row.get("body_ratio", 0)),
            upper_shadow_pct=float(row.get("upper_shadow_pct", 0)),
            lower_shadow_pct=float(row.get("lower_shadow_pct", 0)),
            ema20=float(row.get("ema20", 0)) if pd.notna(row.get("ema20")) else 0,
            atr14=float(row.get("atr14", 0)) if pd.notna(row.get("atr14")) else 0,
            ema20_position=float(row.get("ema20_position", 0))
            if pd.notna(row.get("ema20_position"))
            else 0,
            ema_gap=float(row.get("ema_gap", 0)) if pd.notna(row.get("ema_gap")) else 0,
            bull_strength_5=float(row.get("bull_strength_5", 0.5))
            if pd.notna(row.get("bull_strength_5"))
            else 0.5,
            bar_types=json.dumps(self._get_bar_types(row)),
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            reason=reason,
        )

        try:
            with self._pg_session_factory() as session:
                session.add(signal)
                session.commit()
                logger.info("Signal saved to PG: %s %s %s", pair, direction, quality)
        except Exception:
            # 唯一约束冲突 = 已写入，忽略
            logger.debug("Signal already exists or write failed for %s", pair)

    def _persist_kline(self, pair: str, dataframe: DataFrame) -> None:
        """持久化最新已收盘 K 线到 PostgreSQL（UPSERT）。"""
        if not self._pg_session_factory or len(dataframe) == 0:
            return

        # 当前 timeframe 周期的开盘时间（UTC，floor 到频率）
        # date < 周期开盘 的 K 线视为已收盘
        now_utc = pd.Timestamp.utcnow().tz_localize(None)
        period_start = now_utc.floor(self.timeframe)

        # Normalize date column to tz-naive for comparison
        dates = dataframe["date"]
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)

        closed = dataframe[dates < period_start]
        if len(closed) == 0:
            return

        last = closed.iloc[-1]  # 最新一根已收盘 K 线
        candle_time = last["date"]
        ct = (
            candle_time.to_pydatetime()
            if hasattr(candle_time, "to_pydatetime")
            else candle_time
        )

        try:
            with self._pg_session_factory() as session:
                session.execute(
                    text("""
                        INSERT INTO pa_kline
                            (symbol, timeframe, candle_time,
                             open, high, low, close, volume)
                        VALUES
                            (:symbol, :timeframe, :candle_time,
                             :open, :high, :low, :close, :volume)
                        ON CONFLICT (symbol, timeframe, candle_time) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """),
                    {
                        "symbol": pair,
                        "timeframe": self.timeframe,
                        "candle_time": ct,
                        "open": float(last["open"]),
                        "high": float(last["high"]),
                        "low": float(last["low"]),
                        "close": float(last["close"]),
                        "volume": float(last["volume"]),
                    },
                )
                session.commit()
        except Exception:
            logger.warning("K-line write failed for %s", pair, exc_info=True)

    # ================================================================
    # K 线图表生成
    # ================================================================

    def _generate_chart(self, pair: str, dataframe: DataFrame) -> bytes:
        """生成最近 20 根 K 线蜡烛图 + EMA20，返回 PNG bytes。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mplfinance as mpf

        df = dataframe.tail(20).copy()
        df = df.set_index("date")
        df.index = pd.DatetimeIndex(df.index)

        apds = [mpf.make_addplot(df["ema20"], color="orange", width=1.5)]

        fig, _ = mpf.plot(
            df,
            type="candle",
            style="charles",
            volume=False,
            addplot=apds,
            returnfig=True,
            figratio=(16, 9),
            figscale=1.2,
            title=f"\n{pair} {self.timeframe}",
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # ================================================================
    # TG Bot 通知 (HTTP POST)
    # ================================================================

    def _notify_tg_bot(self, pair: str, row: pd.Series, dataframe: DataFrame) -> None:
        """POST 信号数据 + K线图表到独立 tg-bot 的 HTTP API。"""
        tg_api = self.config.get("tg_api_url", "http://tg-bot:8090")
        direction = row.get("signal_direction", "none")
        quality = row.get("signal_quality", "none")

        # 查询标的显示名称（A 股为中文名，其余为空）
        display_name = None
        with self._pg_session_factory() as session:
            wp = session.query(WatchPair).filter_by(symbol=pair).first()
            if wp:
                display_name = wp.display_name

        payload = {
            "symbol": pair,
            "display_name": display_name,
            "timeframe": self.timeframe,
            "direction": direction,
            "quality": quality,
            "body_pct": float(row.get("body_pct", 0)),
            "close_location": float(row.get("close_location", 0)),
            "body_ratio": float(row.get("body_ratio", 0)),
            "bar_types": self._get_bar_types(row),
            "ema20_above": bool(row.get("above_ema20", False)),
            "ema_gap": float(row.get("ema_gap", 0)) if pd.notna(row.get("ema_gap")) else 0,
            "bull_strength_5": float(row.get("bull_strength_5", 0.5))
            if pd.notna(row.get("bull_strength_5"))
            else 0.5,
            "entry_price": float(row.get("close", 0)),
        }
        if direction == "long":
            payload["stop_loss"] = float(row.get("low", 0))
            risk = payload["entry_price"] - payload["stop_loss"]
            payload["target_price"] = payload["entry_price"] + 2 * risk if risk > 0 else payload["entry_price"]
        else:
            payload["stop_loss"] = float(row.get("high", 0))
            risk = payload["stop_loss"] - payload["entry_price"]
            payload["target_price"] = payload["entry_price"] - 2 * risk if risk > 0 else payload["entry_price"]

        # 生成蜡烛图（失败不阻断通知）
        chart_png = None
        try:
            chart_png = self._generate_chart(pair, dataframe)
        except Exception:
            logger.warning("Failed to generate chart for %s", pair, exc_info=True)

        try:
            if chart_png:
                http_requests.post(
                    f"{tg_api}/signal",
                    files={"chart": ("chart.png", chart_png, "image/png")},
                    data={"payload": json.dumps(payload)},
                    timeout=10,
                )
            else:
                http_requests.post(
                    f"{tg_api}/signal",
                    data={"payload": json.dumps(payload)},
                    timeout=10,
                )
            logger.info("Signal notified to tg-bot: %s %s %s", pair, direction, quality)
        except Exception:
            logger.warning("Failed to notify tg-bot for %s", pair, exc_info=True)

    # ================================================================
    # 消息格式化
    # ================================================================

    def _format_signal_message(self, pair: str, row: pd.Series) -> str:
        """格式化 Telegram 消息。"""
        direction = row.get("signal_direction", "none")
        quality = row.get("signal_quality", "none")

        emoji = "+" if direction == "long" else "-"
        quality_map = {"good": "Good", "acceptable": "OK", "fair": "Fair"}
        quality_cn = quality_map.get(quality, quality)

        lines = [
            f"{emoji} {pair} {self.timeframe}",
            f"{direction.upper()} [{quality_cn}]",
            f"body={row.get('body_pct', 0):.2f} "
            f"close_loc={row.get('close_location', 0):.2f} "
            f"ratio={row.get('body_ratio', 0):.1f}",
        ]

        types = []
        if row.get("is_surprise", False):
            types.append("Surprise")
        if row.get("is_engulfing", False):
            types.append("Engulfing")
        if row.get("is_inside", False):
            types.append("Inside")
        if row.get("is_2k_reversal", False):
            types.append("2K-Reversal")
        if row.get("is_doji", False):
            types.append("Doji")
        if types:
            lines.append("Types: " + " | ".join(types))

        above_ema = row.get("above_ema20", False)
        ema_gap = row.get("ema_gap", 0)
        ema_str = "above" if above_ema else "below"
        lines.append(f"EMA20: {ema_str} (gap={ema_gap:.1f}x ATR)")

        bs = row.get("bull_strength_5", 0.5)
        if bs > 0.6:
            bias = "bullish"
        elif bs < 0.4:
            bias = "bearish"
        else:
            bias = "neutral"
        lines.append(f"5-bar: {bias} ({bs:.0%})")

        return "\n".join(lines)


