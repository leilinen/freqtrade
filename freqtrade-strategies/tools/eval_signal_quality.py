#!/usr/bin/env python3
"""价格行为信号K线质量评估器。

读取 pa_kline 历史 K 线，复用 PriceActionMonitor 策略的指标/分级方法，
统计各质量等级信号K线在后续 LOOKAHEAD 根 K 线内的胜率、盈亏比、净收益。

设计原则：不复制策略逻辑，直接调用 PriceActionMonitor 的指标方法，
策略阈值/逻辑改了，本评估器自动同步，无需两处维护。

支持市场：
  - crypto  (BTC/USDT, ETH/USDT ...)
  - ashare  (600519/SH, 000001/SZ, 830xxx/BJ ...)
  按 symbol 后缀自动识别，无需指定 market。

数据源：freqtrade_monitor.pa_kline（字段 candle_time / timeframe）。

前置条件：在 freqtrade 主仓库 venv 下运行（需 pip install -e . 且能 import talib）。

用法示例：

  # 单标的（默认连本地库）
  .venv/bin/python freqtrade-strategies/tools/eval_signal_quality.py \\
      --symbol 600519/SH --timeframe 1d

  # 连远程库 —— 用 DB_URL 环境变量（推荐，连 Azure/生产库最简洁）
  DB_URL="postgresql://postgres:postgres@xx.xx.xx.xx:5432/freqtrade_monitor" \\
      .venv/bin/python freqtrade-strategies/tools/eval_signal_quality.py \\
      --symbol 600519/SH --timeframe 1d

  # 连远程库 —— 或用 --db-url 临时指定
  .venv/bin/python freqtrade-strategies/tools/eval_signal_quality.py \\
      --symbol BTC/USDT --timeframe 1h \\
      --db-url postgresql://user:pass@remote-host:5432/freqtrade_monitor

  # 批量评估某市场所有 enabled 标的（从 watch_pair 表读取）
  DB_URL="..." .venv/bin/python freqtrade-strategies/tools/eval_signal_quality.py \\
      --market ashare --timeframe 1d

  # 评估时不叠加 EMA20 背景过滤（对比用）
  .venv/bin/python freqtrade-strategies/tools/eval_signal_quality.py \\
      --symbol 600519/SH --timeframe 1d --no-ema-filter

DB 连接优先级：--db-url > 环境变量 DB_URL > 默认 localhost。
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# 让脚本能 import 策略模块（位于 user_data/strategies/）
_HERE = Path(__file__).resolve().parent
_STRATEGY_DIR = _HERE.parent / "user_data" / "strategies"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from price_action_monitor import PriceActionMonitor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_signal_quality")

DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/freqtrade_monitor"
DEFAULT_LOOKAHEAD = 2
ASHARE_SUFFIXES = ("/SH", "/SZ", "/BJ")


# ================================================================
# 工具函数
# ================================================================

def detect_market(symbol: str) -> str:
    """按 symbol 后缀识别市场，与 tg_bot.py 一致。"""
    upper = symbol.upper()
    return "ashare" if any(upper.endswith(s) for s in ASHARE_SUFFIXES) else "crypto"


def db_host_port(db_url: str) -> str:
    """从 DB URL 提取 host:port，用于日志展示（本地/远程一目了然）。"""
    try:
        parsed = urlparse(db_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        return f"{host}:{port}"
    except Exception:
        return "unknown"


def load_klines(db_url: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """从 pa_kline 读取指定标的的 K 线，返回 freqtrade 风格 DataFrame。

    使用正确的字段名 candle_time / timeframe（旧脚本误用 ts / interval，已废弃）。
    """
    engine = create_engine(db_url)
    try:
        rows = engine.connect().execute(
            text(
                "SELECT candle_time, open, high, low, close, volume "
                "FROM pa_kline "
                "WHERE symbol = :symbol AND timeframe = :tf "
                "ORDER BY candle_time ASC"
            ),
            {"symbol": symbol, "tf": timeframe},
        ).fetchall()
    finally:
        engine.dispose()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def load_enabled_symbols(db_url: str, market: str) -> list[str]:
    """从 watch_pair 表读取指定市场、enabled 的标的列表。"""
    engine = create_engine(db_url)
    try:
        rows = engine.connect().execute(
            text(
                "SELECT symbol FROM watch_pair "
                "WHERE market = :m AND enabled = true ORDER BY id"
            ),
            {"m": market},
        ).fetchall()
    finally:
        engine.dispose()
    return [r[0] for r in rows]


# ================================================================
# 信号计算 —— 复用策略方法
# ================================================================

def compute_signals(
    df: pd.DataFrame, use_ema_filter: bool
) -> pd.DataFrame:
    """复用 PriceActionMonitor 的指标与分级方法。

    这些方法（_calc_basic_indicators / _calc_ema_atr / _evaluate_context /
    _classify_signal_quality）均为纯 DataFrame 操作，不依赖 freqtrade 运行时，
    因此 PriceActionMonitor({}) 实例化即可调用。
    策略阈值改了，这里自动同步。
    """
    strategy = PriceActionMonitor({})
    df = strategy._calc_basic_indicators(df)
    df = strategy._calc_ema_atr(df)
    df = strategy._evaluate_context(df)
    df = strategy._classify_signal_quality(df)

    if use_ema_filter:
        # 叠加 EMA20 背景方向过滤：不满足的信号降级为 none
        # （等价于策略里 _ema_context_ok 的判据）
        mask_keep = df.apply(
            lambda row: strategy._ema_context_ok(row, row["signal_direction"]),
            axis=1,
        )
        df.loc[~mask_keep, ["signal_quality", "signal_direction"]] = "none"

    return df


# ================================================================
# Follow-through 评估
# ================================================================

def evaluate_follow_through(df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
    """评估信号 K 线在后续 lookahead 根 K 线内的表现（ATR 归一化）。

    复用旧 eval_signal_strategy 的口径，保持结果可比：
      - same_dir_count: 后续 K 线收盘方向与信号一致的数量
      - max_favorable : 最大有利幅度 / ATR
      - max_adverse   : 最大不利幅度 / ATR
      - net_move      : 终值净位移 / ATR
      - win           : net_move > 0
    """
    results = []
    for idx in df.index[df["signal_quality"] != "none"]:
        if idx + lookahead >= len(df):
            continue
        row = df.loc[idx]
        direction = row["signal_direction"]
        atr = row["atr14"]
        if pd.isna(atr) or atr <= 0:
            continue

        entry_price = row["close"]
        future = df.loc[idx + 1: idx + lookahead]

        if direction == "long":
            same_dir = (future["close"] > future["open"]).sum()
            max_favorable = (future["high"].max() - entry_price) / atr
            max_adverse = (entry_price - future["low"].min()) / atr
            net_move = (future.iloc[-1]["close"] - entry_price) / atr
        else:  # short
            same_dir = (future["close"] < future["open"]).sum()
            max_favorable = (entry_price - future["low"].min()) / atr
            max_adverse = (future["high"].max() - entry_price) / atr
            net_move = (entry_price - future.iloc[-1]["close"]) / atr

        results.append({
            "quality": row["signal_quality"],
            "direction": direction,
            "same_dir_count": same_dir,
            "max_favorable": max_favorable,
            "max_adverse": max_adverse,
            "net_move": net_move,
            "win": net_move > 0,
        })

    return pd.DataFrame(results)


# ================================================================
# 结果输出
# ================================================================

QUALITY_ORDER = ("good", "acceptable", "fair")
DIRECTION_ORDER = ("long", "short")


def print_evaluation(
    rdf: pd.DataFrame,
    symbol: str,
    timeframe: str,
    market: str,
    n_bars: int,
    date_range: str,
    db_label: str,
    lookahead: int,
    use_ema_filter: bool,
) -> dict | None:
    """打印单个标的的评估表，返回汇总摘要（供批量模式汇总用）。"""
    header = (
        f"=== {symbol}  {timeframe}  ({market}, {n_bars} bars, {date_range}) "
        f"[db: {db_label}] [lookahead={lookahead}, ema_filter={use_ema_filter}] ==="
    )
    print(header)

    if rdf.empty:
        print("  (no signals)\n")
        return None

    print(
        f"  {'quality':<12}{'dir':<7}{'signals':>8}{'win%':>8}"
        f"{'avgFav':>10}{'avgAdv':>10}{'netMove':>11}{'F/A':>7}"
    )
    summary = {}
    for quality in QUALITY_ORDER:
        for direction in DIRECTION_ORDER:
            subset = rdf[
                (rdf["quality"] == quality) & (rdf["direction"] == direction)
            ]
            if len(subset) == 0:
                continue
            win_rate = subset["win"].mean() * 100
            avg_fav = subset["max_favorable"].mean()
            avg_adv = subset["max_adverse"].mean()
            avg_net = subset["net_move"].mean()
            rr = avg_fav / avg_adv if avg_adv > 0 else float("inf")
            print(
                f"  {quality:<12}{direction:<7}{len(subset):>8}{win_rate:>7.1f}%"
                f"{avg_fav:>9.3f}{avg_adv:>10.3f}{avg_net:>+10.3f}{rr:>7.2f}"
            )
            if quality not in summary:
                summary[quality] = {}
            summary[quality][direction] = {
                "n": len(subset),
                "win_rate": win_rate,
                "net_move": avg_net,
            }
    print()
    return summary


def print_batch_summary(
    summaries: list[tuple[str, dict]],
    market: str,
    timeframe: str,
    db_label: str,
) -> None:
    """批量模式：打印跨标的汇总表（各标的 good 档 win% 横向对比）。"""
    print("=" * 70)
    print(f"BATCH SUMMARY  market={market}  timeframe={timeframe}  [db: {db_label}]")
    print("=" * 70)

    # 以每个有 good 信号的标的为一行
    print(f"  {'symbol':<14}{'good L win%':>13}{'good S win%':>13}{'acc L win%':>13}")
    for symbol, sm in summaries:
        def _wr(q, d):
            cell = sm.get(q, {}).get(d)
            if not cell or cell["n"] == 0:
                return "   -"
            return f"{cell['win_rate']:.0f}%({cell['n']})"
        print(
            f"  {symbol:<14}"
            f"{_wr('good', 'long'):>13}{_wr('good', 'short'):>13}"
            f"{_wr('acceptable', 'long'):>13}"
        )
    print()


# ================================================================
# 主流程
# ================================================================

def evaluate_one(
    db_url: str,
    symbol: str,
    timeframe: str,
    lookahead: int,
    use_ema_filter: bool,
    since: str | None,
) -> tuple[dict | None, int, str, str]:
    """评估单个标的，返回 (summary, n_bars, date_range, error_msg)。"""
    df = load_klines(db_url, symbol, timeframe)
    if df.empty:
        return None, 0, "", f"no kline data for {symbol} {timeframe}"

    if since:
        cutoff = pd.Timestamp(since, tz="UTC")
        df = df[df["date"] >= cutoff].reset_index(drop=True)
        if df.empty:
            return None, 0, "", f"no data after {since} for {symbol} {timeframe}"

    date_range = f"{df['date'].iloc[0]:%Y-%m-%d} ~ {df['date'].iloc[-1]:%Y-%m-%d}"
    market = detect_market(symbol)

    df = compute_signals(df, use_ema_filter)
    rdf = evaluate_follow_through(df, lookahead)
    summary = print_evaluation(
        rdf, symbol, timeframe, market, len(df), date_range,
        db_host_port(db_url), lookahead, use_ema_filter,
    )
    return summary, len(df), date_range, ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="价格行为信号K线质量评估器（复用 PriceActionMonitor 策略逻辑）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="详见脚本顶部 docstring 的用法示例。",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--symbol", help="单个标的，如 600519/SH 或 BTC/USDT")
    target.add_argument(
        "--market", choices=["ashare", "crypto"],
        help="批量评估该市场所有 enabled 标的（从 watch_pair 读取）",
    )
    parser.add_argument("--timeframe", required=True, help="K 线周期，如 1h / 1d")
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD,
                        help=f"前瞻窗口（根数），默认 {DEFAULT_LOOKAHEAD}")
    parser.add_argument("--no-ema-filter", action="store_true",
                        help="不叠加 EMA20 背景过滤（对比用）")
    parser.add_argument("--since", help="只评估该日期之后的数据，如 2026-01-01")
    parser.add_argument(
        "--db-url",
        default=None,
        help="数据库连接串。优先级: --db-url > 环境变量 DB_URL > 默认 localhost",
    )
    args = parser.parse_args()

    db_url = args.db_url or os.environ.get("DB_URL", DEFAULT_DB_URL)
    db_label = db_host_port(db_url)
    use_ema_filter = not args.no_ema_filter
    logger.info("DB: %s | ema_filter=%s", db_label, use_ema_filter)

    if args.symbol:
        symbols = [args.symbol]
        batch = False
    else:
        symbols = load_enabled_symbols(db_url, args.market)
        if not symbols:
            logger.warning("No enabled %s symbols in watch_pair.", args.market)
            return
        logger.info("Loaded %d enabled %s symbol(s).", len(symbols), args.market)
        batch = True

    summaries: list[tuple[str, dict]] = []
    for i, symbol in enumerate(symbols, 1):
        if batch:
            logger.info("[%d/%d] %s ...", i, len(symbols), symbol)
        summary, n_bars, date_range, err = evaluate_one(
            db_url, symbol, args.timeframe, args.lookahead, use_ema_filter, args.since,
        )
        if err:
            logger.warning("%s -> %s", symbol, err)
        elif summary is not None:
            summaries.append((symbol, summary))

    if batch and summaries:
        print_batch_summary(summaries, args.market, args.timeframe, db_label)

    logger.info("Done.")


if __name__ == "__main__":
    main()
