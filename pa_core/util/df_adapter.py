"""Adapter: freqtrade OHLCV DataFrame → pa_core KlineFrame.

This is the bridge between the two data models (migration risk #1 in
docs/pa-migration-plan.md):

- freqtrade: pandas DataFrame, columns ``date/open/high/low/close/volume``,
  ascending (oldest row first), ``date`` is a tz-aware UTC Timestamp.
- pa_core: newest-first ``KlineBar`` tuples with ``seq`` (1 = newest closed)
  and ``IndicatorBundle`` aligned to that order.

Closed-candle guarantee: freqtrade drops the incomplete candle by default
(``_ft_has["ohlcv_partial_candle"] = True``, no exchange overrides it), both
for live/dry-run feeds and backtest data. Every dataframe row can therefore be
mapped to a ``closed=True`` KlineBar directly — no forming-bar detection.

Warmup semantics mirror PA_Agent's ``build_analysis_frame``: the frame holds
the newest *n_bars* candles; up to *warmup_bars* older candles feed the
EMA20/ATR14 warm-up but are not returned in the frame.

Only the tail (``n_bars + warmup_bars`` rows) is iterated, so the row-wise
conversion cost stays bounded regardless of dataframe size.
"""
from __future__ import annotations

from pandas import DataFrame

from pa_core.data_structures import IndicatorBundle, KlineBar, KlineFrame, normalize_kline_bar
from pa_core.datetime_ts import datetime_to_ts_ms
from pa_core.snapshot import INDICATOR_WARMUP_BARS, compute_indicators
from pa_core.util.timefmt import now_local_ms


def df_to_kline_frame(
    dataframe: DataFrame,
    symbol: str,
    timeframe: str,
    n_bars: int,
    *,
    warmup_bars: int = INDICATOR_WARMUP_BARS,
    now_ms: int | None = None,
) -> KlineFrame | None:
    """Convert a freqtrade OHLCV dataframe into an analysis KlineFrame.

    :param dataframe: freqtrade candle dataframe (oldest-first, closed candles).
    :param symbol: pair/symbol name for the frame.
    :param timeframe: timeframe string (e.g. ``"1h"``, ``"1d"``).
    :param n_bars: number of newest candles to include in the frame.
    :param warmup_bars: extra older candles used only for indicator warm-up.
    :param now_ms: override for ``snapshot_ts_local_ms`` (tests / replay).
    :return: KlineFrame, or ``None`` when the dataframe is empty / has fewer
        than *n_bars* rows — strategies must not raise inside ``populate_*``.
    """
    if dataframe is None or dataframe.empty or n_bars < 1:
        return None
    if len(dataframe) < n_bars:
        return None

    has_amount = "amount" in dataframe.columns
    has_pct_chg = "pct_chg" in dataframe.columns

    tail = dataframe.tail(n_bars + warmup_bars)
    rows = tail.iloc[::-1]  # newest-first, like PA_Agent bar lists

    bars: list[KlineBar] = []
    for i, (_, row) in enumerate(rows.iterrows()):
        bars.append(
            KlineBar(
                seq=i + 1,
                ts_open=datetime_to_ts_ms(row["date"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                amount=float(row["amount"]) if has_amount else 0.0,
                pct_chg=float(row["pct_chg"]) if has_pct_chg else None,
                closed=True,
            )
        )

    rebased = [normalize_kline_bar(b) for b in bars]
    indicators_all = compute_indicators(rebased)
    n = min(n_bars, len(rebased))
    return KlineFrame(
        symbol=symbol,
        timeframe=timeframe,
        bars=tuple(rebased[:n]),
        indicators=IndicatorBundle(
            ema20=indicators_all.ema20[:n],
            atr14=indicators_all.atr14[:n],
        ),
        snapshot_ts_local_ms=now_ms if now_ms is not None else now_local_ms(),
    )
