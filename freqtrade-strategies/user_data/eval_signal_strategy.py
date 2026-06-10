"""
信号K线策略评估脚本 V2

读取 BTC/USDT 1h 近 6 个月真实数据，对比 V1(原始) vs V2(改进) 信号分级效果。

改进方向：
1. 收紧阈值 — body_pct、close_location、shadow_pct、body_ratio
2. EMA20 背景过滤 — 方向一致性 + ema_gap 限制
3. 缩短前瞻窗口 — 信号K线只在短时间内有效

评估指标：
- 胜率: 信号后 N 根K线收盘方向与信号一致的比例
- 平均净收益: ATR 归一化
- 盈亏比: 平均有利幅度 / 平均不利幅度
"""

import logging
import numpy as np
import pandas as pd
import talib
from sqlalchemy import create_engine, Column, Integer, String, Float, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:postgres@localhost:15432/pricedog"
LOOKAHEAD = 2  # 信号K线短期有效，只看 2 根后续K线

# ================================================================
# V1 / V2 阈值定义
# ================================================================

V1_THRESHOLDS = {
    "good":      {"body_pct": 0.6,  "close_loc": 0.85, "shadow": 0.10, "body_ratio": 0.0},
    "acceptable": {"body_pct": 0.4, "close_loc": 0.60, "shadow": 0.25, "body_ratio": 0.0},
    "fair":      {"body_pct": 0.3,  "close_loc": 0.50, "shadow": 1.00, "body_ratio": 0.0},
}

V2_THRESHOLDS = {
    "good":       {"body_pct": 0.85, "close_loc": 0.90, "shadow": 0.05, "body_ratio": 1.5},
    "acceptable": {"body_pct": 0.65, "close_loc": 0.75, "shadow": 0.15, "body_ratio": 1.2},
    "fair":       {"body_pct": 0.50, "close_loc": 0.60, "shadow": 1.00, "body_ratio": 0.0},
}

V2_EMA_FILTER = {
    "max_ema_gap": 2.0,
    "bull_strength_min_long": 0.5,
    "bull_strength_max_short": 0.5,
}


# ================================================================
# 数据加载 + 指标计算
# ================================================================

def load_data() -> pd.DataFrame:
    """从本地 PG 读取 BTCUSDT 1h 数据。"""
    logger.info("=" * 60)
    logger.info("Loading BTCUSDT 1h data from PostgreSQL...")

    engine = create_engine(DB_URL)
    query = text("""
        SELECT ts, open, high, low, close, volume
        FROM pa_kline
        WHERE symbol = 'BTCUSDT' AND "interval" = '1h'
        ORDER BY ts ASC
    """)
    df = pd.read_sql(query, engine)
    engine.dispose()

    df["date"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates(subset=["ts"]).sort_values("date").reset_index(drop=True)

    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    df = df[df["date"] >= cutoff].reset_index(drop=True)

    logger.info("Loaded %d candles: %s ~ %s", len(df), df["date"].iloc[0], df["date"].iloc[-1])
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有价格行为指标。"""
    logger.info("Computing indicators...")

    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["body_pct"] = np.where(df["range"] > 0, df["body"] / df["range"], 0.0)
    df["close_location"] = np.where(df["range"] > 0, (df["close"] - df["low"]) / df["range"], 0.5)
    df["upper_shadow_pct"] = np.where(
        df["range"] > 0,
        (df["high"] - np.maximum(df["close"], df["open"])) / df["range"], 0.0,
    )
    df["lower_shadow_pct"] = np.where(
        df["range"] > 0,
        (np.minimum(df["close"], df["open"]) - df["low"]) / df["range"], 0.0,
    )
    df["is_bull"] = df["close"] > df["open"]
    df["median_body_20"] = df["body"].rolling(window=20, min_periods=10).median()
    df["body_ratio"] = np.where(df["median_body_20"] > 0, df["body"] / df["median_body_20"], 0.0)
    df["ema20"] = talib.EMA(df["close"], timeperiod=20)
    df["atr14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

    # EMA 背景
    df["above_ema20"] = df["close"] > df["ema20"]
    df["ema_gap"] = np.where(df["atr14"] > 0, abs(df["close"] - df["ema20"]) / df["atr14"], 0.0)

    bull_body = np.where(df["is_bull"], df["body"], 0.0)
    bear_body = np.where(~df["is_bull"], df["body"], 0.0)
    bull_sum_5 = pd.Series(bull_body).rolling(window=5, min_periods=3).sum()
    bear_sum_5 = pd.Series(bear_body).rolling(window=5, min_periods=3).sum()
    total_body_5 = bull_sum_5 + bear_sum_5
    df["bull_strength_5"] = np.where(total_body_5 > 0, bull_sum_5 / total_body_5, 0.5)

    return df


# ================================================================
# 信号分级
# ================================================================

def classify_signals(
    df: pd.DataFrame,
    thresholds: dict,
    use_ema_filter: bool = False,
    prefix: str = "",
) -> pd.DataFrame:
    """按给定阈值分级信号K线，结果写入 df[prefix + 'signal_quality'] 等列。"""
    q_col = f"{prefix}signal_quality"
    d_col = f"{prefix}signal_direction"

    df[q_col] = "none"
    df[d_col] = "none"

    is_bull = df["is_bull"]
    is_bear = ~df["is_bull"]

    # 做多
    for quality in ("good", "acceptable", "fair"):
        t = thresholds[quality]
        mask = (
            is_bull
            & (df[q_col] == "none")
            & (df["body_pct"] >= t["body_pct"])
            & (df["close_location"] >= t["close_loc"])
            & (df["upper_shadow_pct"] <= t["shadow"])
        )
        if t["body_ratio"] > 0:
            mask = mask & (df["body_ratio"] >= t["body_ratio"])
        if use_ema_filter:
            mask = mask & df["above_ema20"] & (df["ema_gap"] <= V2_EMA_FILTER["max_ema_gap"])
            mask = mask & (df["bull_strength_5"] >= V2_EMA_FILTER["bull_strength_min_long"])
        df.loc[mask, q_col] = quality
        df.loc[mask, d_col] = "long"

    # 做空
    for quality in ("good", "acceptable", "fair"):
        t = thresholds[quality]
        mask = (
            is_bear
            & (df[q_col] == "none")
            & (df["body_pct"] >= t["body_pct"])
            & (df["close_location"] <= 1.0 - t["close_loc"])
            & (df["lower_shadow_pct"] <= t["shadow"])
        )
        if t["body_ratio"] > 0:
            mask = mask & (df["body_ratio"] >= t["body_ratio"])
        if use_ema_filter:
            mask = mask & (~df["above_ema20"]) & (df["ema_gap"] <= V2_EMA_FILTER["max_ema_gap"])
            mask = mask & (df["bull_strength_5"] <= V2_EMA_FILTER["bull_strength_max_short"])
        df.loc[mask, q_col] = quality
        df.loc[mask, d_col] = "short"

    return df


# ================================================================
# Follow-through 评估
# ================================================================

def evaluate_follow_through(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """评估信号后的跟随表现。"""
    q_col = f"{prefix}signal_quality"
    d_col = f"{prefix}signal_direction"

    results = []
    for quality in ("good", "acceptable", "fair"):
        indices = df.index[df[q_col] == quality].tolist()
        for idx in indices:
            if idx + LOOKAHEAD >= len(df):
                continue
            row = df.loc[idx]
            direction = row[d_col]
            atr = row["atr14"]
            if pd.isna(atr) or atr <= 0:
                continue

            entry_price = row["close"]
            future = df.loc[idx + 1: idx + LOOKAHEAD]

            if direction == "long":
                same_dir = (future["close"] > future["open"]).sum()
                max_favorable = (future["high"].max() - entry_price) / atr
                max_adverse = (entry_price - future["low"].min()) / atr
                net_move = (future.iloc[-1]["close"] - entry_price) / atr
            else:
                same_dir = (future["close"] < future["open"]).sum()
                max_favorable = (entry_price - future["low"].min()) / atr
                max_adverse = (future["high"].max() - entry_price) / atr
                net_move = (entry_price - future.iloc[-1]["close"]) / atr

            results.append({
                "quality": quality,
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

def print_signal_distribution(df: pd.DataFrame, label: str, prefix: str = "") -> None:
    """输出信号分布统计。"""
    q_col = f"{prefix}signal_quality"
    d_col = f"{prefix}signal_direction"

    quality_counts = df[q_col].value_counts()
    direction_counts = df[df[q_col] != "none"][d_col].value_counts()
    total = len(df)

    logger.info("--- %s Signal Distribution ---", label)
    logger.info("  Total candles: %d", total)
    for q in ("good", "acceptable", "fair", "none"):
        c = quality_counts.get(q, 0)
        pct = c / total * 100
        logger.info("  %-12s: %4d (%5.1f%%)", q, c, pct)

    for d in ("long", "short"):
        c = direction_counts.get(d, 0)
        logger.info("  %s: %d", d, c)


def print_evaluation(rdf: pd.DataFrame, label: str) -> None:
    """输出评估结果。"""
    logger.info("=" * 60)
    logger.info("%s EVALUATION RESULTS (lookahead=%d)", label, LOOKAHEAD)
    logger.info("=" * 60)

    for quality in ("good", "acceptable", "fair"):
        subset = rdf[rdf["quality"] == quality]
        if len(subset) == 0:
            logger.info("--- %s: no signals ---", quality.upper())
            continue

        win_rate = subset["win"].mean() * 100
        avg_favorable = subset["max_favorable"].mean()
        avg_adverse = subset["max_adverse"].mean()
        avg_net = subset["net_move"].mean()
        avg_same_dir = subset["same_dir_count"].mean()
        rr_ratio = avg_favorable / avg_adverse if avg_adverse > 0 else float("inf")

        logger.info("--- %s (%d signals) ---", quality.upper(), len(subset))
        logger.info("  Win rate:       %.1f%%", win_rate)
        logger.info("  Avg same-dir:   %.2f / %d", avg_same_dir, LOOKAHEAD)
        logger.info("  Avg favorable:  %.3f ATR", avg_favorable)
        logger.info("  Avg adverse:    %.3f ATR", avg_adverse)
        logger.info("  Avg net move:   %+.3f ATR", avg_net)
        logger.info("  F/A ratio:      %.2f", rr_ratio)

    # 按方向拆分
    logger.info("--- By Direction ---")
    for quality in ("good", "acceptable", "fair"):
        for direction in ("long", "short"):
            subset = rdf[(rdf["quality"] == quality) & (rdf["direction"] == direction)]
            if len(subset) == 0:
                continue
            win_rate = subset["win"].mean() * 100
            avg_net = subset["net_move"].mean()
            logger.info(
                "  %-12s %6s: %3d signals, win=%.0f%%, net=%+.3f ATR",
                quality, direction, len(subset), win_rate, avg_net,
            )


# ================================================================
# PG 写入
# ================================================================

class _Base(DeclarativeBase):
    pass


class EvalResult(_Base):
    __tablename__ = "eval_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String)
    quality: Mapped[str] = mapped_column(String)
    direction: Mapped[str] = mapped_column(String)
    count: Mapped[int] = mapped_column(Integer)
    win_rate: Mapped[float] = mapped_column(Float)
    avg_favorable: Mapped[float] = mapped_column(Float)
    avg_adverse: Mapped[float] = mapped_column(Float)
    avg_net: Mapped[float] = mapped_column(Float)
    rr_ratio: Mapped[float] = mapped_column(Float)


def save_to_pg(rdf: pd.DataFrame, version: str) -> None:
    """将评估结果写入 PG。"""
    engine = create_engine("postgresql://postgres:postgres@localhost:15432/freqtrade_monitor")
    _Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine)

    with SessionFactory() as session:
        # 清除同版本旧数据
        session.query(EvalResult).filter_by(version=version).delete()
        for quality in ("good", "acceptable", "fair"):
            subset = rdf[rdf["quality"] == quality]
            if len(subset) == 0:
                continue
            win_rate = float(subset["win"].mean() * 100)
            avg_fav = float(subset["max_favorable"].mean())
            avg_adv = float(subset["max_adverse"].mean())
            avg_net = float(subset["net_move"].mean())
            rr = avg_fav / avg_adv if avg_adv > 0 else 0.0
            session.add(EvalResult(
                version=version, quality=quality, direction="all",
                count=len(subset), win_rate=win_rate,
                avg_favorable=avg_fav, avg_adverse=avg_adv,
                avg_net=avg_net, rr_ratio=rr,
            ))
        session.commit()

    engine.dispose()
    logger.info("Saved %s results to freqtrade_monitor.eval_results", version)


# ================================================================
# Main
# ================================================================

def main() -> None:
    df = load_data()
    df = compute_indicators(df)

    # V1 分级
    df = classify_signals(df, V1_THRESHOLDS, use_ema_filter=False, prefix="v1_")
    print_signal_distribution(df, "V1", prefix="v1_")
    rdf_v1 = evaluate_follow_through(df, prefix="v1_")
    print_evaluation(rdf_v1, "V1")

    # V2 分级（收紧阈值 + EMA20 过滤）
    df = classify_signals(df, V2_THRESHOLDS, use_ema_filter=True, prefix="v2_")
    print_signal_distribution(df, "V2", prefix="v2_")
    rdf_v2 = evaluate_follow_through(df, prefix="v2_")
    print_evaluation(rdf_v2, "V2")

    # 对比摘要
    logger.info("=" * 60)
    logger.info("V1 vs V2 COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info("%-12s %8s %8s | %8s %8s", "Quality", "V1#n", "V1 Win%", "V2#n", "V2 Win%")
    for quality in ("good", "acceptable", "fair"):
        v1_sub = rdf_v1[rdf_v1["quality"] == quality]
        v2_sub = rdf_v2[rdf_v2["quality"] == quality]
        v1_n = len(v1_sub)
        v2_n = len(v2_sub)
        v1_wr = v1_sub["win"].mean() * 100 if v1_n > 0 else 0
        v2_wr = v2_sub["win"].mean() * 100 if v2_n > 0 else 0
        logger.info("%-12s %8d %7.1f%% | %8d %7.1f%%", quality, v1_n, v1_wr, v2_n, v2_wr)

    # PG 写入
    save_to_pg(rdf_v1, "v1")
    save_to_pg(rdf_v2, "v2")

    logger.info("DONE")


if __name__ == "__main__":
    main()
