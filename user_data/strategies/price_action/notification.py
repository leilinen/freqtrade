"""Telegram notification and chart rendering helpers."""
from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd
import requests as http_requests
from pandas import DataFrame

from .models import WatchPair
from .rules import PriceActionSignalRules


logger = logging.getLogger(__name__)


class SignalNotifier:
    """Build signal payloads, render charts, and POST to tg-bot."""

    def __init__(
        self,
        session_factory,
        *,
        config: dict,
        timeframe: str,
        rules: PriceActionSignalRules | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._config = config
        self._timeframe = timeframe
        self._rules = rules or PriceActionSignalRules()

    def generate_chart(
        self,
        pair: str,
        timeframe: str,
        dataframe: DataFrame,
        num_candles: int = 20,
    ) -> bytes:
        """Render recent candles + EMA20 + volume as PNG bytes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mplfinance as mpf

        if len(dataframe) == 0:
            raise ValueError(
                f"Not enough candles for {pair}: have {len(dataframe)}, need at least 1"
            )
        num_candles = min(num_candles, len(dataframe))

        df = dataframe.copy()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df = df.tail(num_candles).set_index("date")
        df.index = pd.DatetimeIndex(df.index)

        apds = [mpf.make_addplot(df["ema20"], color="orange", width=1.5)]

        fig, _ = mpf.plot(
            df,
            type="candle",
            style="charles",
            volume=True,
            addplot=apds,
            returnfig=True,
            figratio=(16, 9),
            figscale=1.2,
            title=f"\n{pair} {timeframe}",
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def notify_tg_bot(
        self,
        pair: str,
        row: pd.Series,
        dataframe: DataFrame,
        *,
        chart_generator: Callable[[str, str, DataFrame], bytes] | None = None,
    ) -> None:
        """POST signal data and optional chart to the tg-bot HTTP API."""
        tg_api = self._config.get("tg_api_url", "http://tg-bot:8090")
        direction = row.get("signal_direction", "none")
        quality = row.get("signal_quality", "none")

        display_name = None
        if self._session_factory:
            with self._session_factory() as session:
                wp = session.query(WatchPair).filter_by(symbol=pair).first()
                if wp:
                    display_name = wp.display_name

        candle_time = row.get("date", None)
        if candle_time is None:
            candle_time = datetime.now(UTC)

        payload = {
            "symbol": pair,
            "display_name": display_name,
            "signal_time": candle_time.isoformat(),
            "timeframe": self._timeframe,
            "direction": direction,
            "quality": quality,
            "signal_type": row.get("signal_type", f"signal_bar_{quality}"),
            "body_pct": float(row.get("body_pct", 0)),
            "close_location": float(row.get("close_location", 0)),
            "body_ratio": float(row.get("body_ratio", 0)),
            "bar_types": self._rules.get_bar_types(row),
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
            payload["target_price"] = (
                payload["entry_price"] + 2 * risk
                if risk > 0
                else payload["entry_price"]
            )
        else:
            payload["stop_loss"] = float(row.get("high", 0))
            risk = payload["stop_loss"] - payload["entry_price"]
            payload["target_price"] = (
                payload["entry_price"] - 2 * risk
                if risk > 0
                else payload["entry_price"]
            )

        chart_png = None
        try:
            generator = chart_generator or self.generate_chart
            chart_png = generator(pair, self._timeframe, dataframe)
        except Exception:
            logger.warning("Failed to generate chart for %s", pair, exc_info=True)

        try:
            if chart_png:
                http_requests.post(
                    f"{tg_api}/signal",
                    files={"chart": ("chart.png", chart_png, "image/png")},
                    data={"payload": json.dumps(payload)},
                    timeout=30,
                )
            else:
                http_requests.post(
                    f"{tg_api}/signal",
                    data={"payload": json.dumps(payload)},
                    timeout=30,
                )
            logger.info("Signal notified to tg-bot: %s %s %s", pair, direction, quality)
        except Exception:
            logger.warning("Failed to notify tg-bot for %s", pair, exc_info=True)
