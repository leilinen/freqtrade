"""SQLAlchemy models for the Price Action monitor."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class _Base(DeclarativeBase):
    pass


class WatchPair(_Base):
    """Watched pair configuration."""

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
    """Signal-bar record."""

    __tablename__ = "pa_signal"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    market: Mapped[str] = mapped_column(String, nullable=False)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    timeframe: Mapped[str] = mapped_column(String, nullable=False)
    candle_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    signal_type: Mapped[str] = mapped_column(String, nullable=False)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    quality: Mapped[str] = mapped_column(String, nullable=False)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    body_pct: Mapped[float] = mapped_column(Float)
    close_location: Mapped[float] = mapped_column(Float)
    body_ratio: Mapped[float] = mapped_column(Float)
    upper_shadow_pct: Mapped[float] = mapped_column(Float)
    lower_shadow_pct: Mapped[float] = mapped_column(Float)
    ema20: Mapped[float] = mapped_column(Float)
    atr14: Mapped[float] = mapped_column(Float)
    ema20_position: Mapped[float] = mapped_column(Float)
    ema_gap: Mapped[float] = mapped_column(Float)
    bull_strength_5: Mapped[float] = mapped_column(Float)
    bar_types: Mapped[str] = mapped_column(String, default="[]")
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    target_price: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(String, default="")

    __table_args__ = (
        UniqueConstraint(
            "symbol", "timeframe", "candle_time", "signal_type", "direction",
            name="ux_pa_signal_identity",
        ),
    )


class PaKline(_Base):
    """Persisted raw monitor OHLCV record."""

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

