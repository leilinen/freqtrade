"""Candlestick chart image generator for Telegram /quote command.

Uses mplfinance + matplotlib (Agg backend) to render PNG bytes from OHLCV data.
Algorithm mirrors freqtrade-strategies/user_data/strategies/price_action_monitor.py
but decoupled from any strategy class.
"""

import io

import pandas as pd


def generate_candlestick_chart(
    pair: str,
    timeframe: str,
    df: pd.DataFrame,
    num_candles: int = 20,
) -> bytes:
    """Generate candlestick + volume + EMA20 chart as PNG bytes.

    :param pair: Pair label for chart title.
    :param timeframe: Timeframe label for chart title.
    :param df: OHLCV DataFrame with columns ['date','open','high','low','close','volume'].
               Must contain at least `num_candles` rows.
    :param num_candles: Number of most-recent candles to plot (default 20).
    :return: PNG image bytes.
    :raises ImportError: if matplotlib/mplfinance are not installed.
    :raises ValueError: if the DataFrame has fewer than `num_candles` rows.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplfinance as mpf

    if len(df) < num_candles:
        raise ValueError(f"Not enough candles for {pair}: have {len(df)}, need {num_candles}")

    # Compute EMA20 on full df to avoid warmup edge effects on the sliced window.
    df = df.copy()
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
