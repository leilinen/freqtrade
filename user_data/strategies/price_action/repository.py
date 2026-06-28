"""Persistence helpers for the Price Action monitor."""
from __future__ import annotations

import json
import logging
import time as _time
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame
from sqlalchemy import text

from .models import PaSignal, WatchPair
from .rules import PriceActionSignalRules


logger = logging.getLogger(__name__)


class PriceActionRepository:
    """Repository for watch pairs, signal rows, and persisted OHLCV."""

    def __init__(
        self,
        session_factory,
        *,
        timeframe: str,
        market: str = "crypto",
        rules: PriceActionSignalRules | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._timeframe = timeframe
        self._market = market
        self._rules = rules or PriceActionSignalRules()

    @property
    def market(self) -> str:
        return self._market

    @market.setter
    def market(self, value: str) -> None:
        self._market = value

    def init_default_pairs(
        self,
        pairs: list[str],
        *,
        market: str,
        display_name_fetcher: Callable[[str], str | None] | None = None,
    ) -> None:
        """Insert default watch pairs if they do not exist."""
        self._market = market
        with self._session_factory() as session:
            for symbol in pairs:
                existing = session.query(WatchPair).filter_by(symbol=symbol).first()
                if existing:
                    continue
                display_name = display_name_fetcher(symbol) if display_name_fetcher else None
                if display_name_fetcher:
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

    def save_signal(
        self,
        pair: str,
        row: pd.Series,
        reason: str,
        *,
        signal_type: str | None = None,
    ) -> bool:
        """Persist a signal row. Returns False on duplicate or disabled PG."""
        if not self._session_factory:
            return False

        quality = row.get("signal_quality", "none")
        direction = row.get("signal_direction", "none")
        entry_price = float(row.get("close", 0))

        if direction == "long":
            stop_loss = float(row.get("low", 0))
            risk = entry_price - stop_loss
            target_price = entry_price + 2 * risk if risk > 0 else entry_price
        else:
            stop_loss = float(row.get("high", 0))
            risk = stop_loss - entry_price
            target_price = entry_price - 2 * risk if risk > 0 else entry_price

        candle_time = row.get("date", None)
        if candle_time is None:
            candle_time = datetime.now(UTC)

        signal = PaSignal(
            market=self._market,
            symbol=pair,
            timeframe=self._timeframe,
            candle_time=candle_time,
            signal_type=f"signal_bar_{quality}" if signal_type is None else signal_type,
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
            bar_types=json.dumps(self._rules.get_bar_types(row)),
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            reason=reason,
        )

        try:
            with self._session_factory() as session:
                session.add(signal)
                session.commit()
                logger.info("Signal saved to PG: %s %s %s", pair, direction, quality)
                return True
        except Exception:
            logger.debug("Signal already exists or write failed for %s", pair)
            return False

    def persist_kline(self, pair: str, dataframe: DataFrame) -> None:
        """Persist strategy OHLCV rows into PostgreSQL via batch upsert."""
        if not self._session_factory or len(dataframe) == 0:
            return

        rows = []
        for _, candle in dataframe.iterrows():
            candle_time = candle["date"]
            ct = (
                candle_time.to_pydatetime()
                if hasattr(candle_time, "to_pydatetime")
                else candle_time
            )
            if getattr(ct, "tzinfo", None) is not None:
                ct = ct.astimezone(UTC).replace(tzinfo=None)
            rows.append({
                "symbol": pair,
                "timeframe": self._timeframe,
                "candle_time": ct,
                "open": float(candle.get("open", 0)),
                "high": float(candle.get("high", 0)),
                "low": float(candle.get("low", 0)),
                "close": float(candle.get("close", 0)),
                "volume": float(candle.get("volume", 0)),
            })

        if not rows:
            return

        try:
            with self._session_factory() as session:
                session.execute(text("""
                    INSERT INTO pa_kline
                        (symbol, timeframe, candle_time, open, high, low, close, volume)
                    VALUES
                        (:symbol, :timeframe, :candle_time, :open, :high, :low, :close, :volume)
                    ON CONFLICT (symbol, timeframe, candle_time) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """), rows)
                session.commit()
                logger.debug(
                    "Klines persisted to PG: %s %s rows=%d",
                    pair,
                    self._timeframe,
                    len(rows),
                )
        except Exception:
            logger.warning("K-line write failed for %s", pair, exc_info=True)
