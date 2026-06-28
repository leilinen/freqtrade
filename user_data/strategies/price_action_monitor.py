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

import atexit
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests as http_requests
from pandas import DataFrame
from price_action.models import PaKline, PaSignal, WatchPair, _Base
from price_action.notification import SignalNotifier
from price_action.repository import PriceActionRepository
from price_action.rules import PriceActionSignalRules
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)

__all__ = [
    "PaKline",
    "PaSignal",
    "PriceActionMonitor",
    "WatchPair",
    "_Base",
    "http_requests",
]

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
    FOLLOW_THROUGH_WINDOW = 3

    # 特殊K线参数
    SURPRISE_LOOKBACK = 20
    SURPRISE_MIN_BODY_PCT = 0.5

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._pg_engine = None
        self._pg_session_factory = None
        self._chart_http_server = None
        self._market = "crypto"
        self._rules = PriceActionSignalRules()
        self._repository: PriceActionRepository | None = None
        self._notifier: SignalNotifier | None = None

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
        self._repository = PriceActionRepository(
            self._pg_session_factory,
            timeframe=self.timeframe,
            market=self._market,
            rules=self._rules,
        )
        self._notifier = SignalNotifier(
            self._pg_session_factory,
            config=self.config,
            timeframe=self.timeframe,
            rules=self._rules,
        )
        atexit.register(self._cleanup)

        # 初始化默认标的
        self._init_default_pairs()

        logger.info("PG persistence initialized: %s", db_url)

        # 启动 /quote HTTP server（供 tg-bot 调用获取 K 线图）
        chart_port = self.config.get("pa_chart_port")
        if chart_port:
            try:
                self._start_chart_http_server(int(chart_port))
            except Exception:
                logger.warning("Failed to start chart HTTP server on port %s", chart_port, exc_info=True)

    def _cleanup(self) -> None:
        if self._chart_http_server is not None:
            try:
                self._chart_http_server.shutdown()
            except Exception:
                logger.warning("Failed to shutdown chart HTTP server", exc_info=True)
            self._chart_http_server = None
        if self._pg_engine:
            self._pg_engine.dispose()

    def _get_repository(self) -> PriceActionRepository | None:
        """Return the persistence repository when PG is configured."""
        if not self._pg_session_factory:
            return None
        if self._repository is None:
            self._repository = PriceActionRepository(
                self._pg_session_factory,
                timeframe=self.timeframe,
                market=self._market,
                rules=self._rules,
            )
        self._repository.market = self._market
        return self._repository

    def _get_notifier(self) -> SignalNotifier:
        """Return the notifier, creating it lazily for tests and direct calls."""
        if self._notifier is None:
            self._notifier = SignalNotifier(
                self._pg_session_factory,
                config=self.config,
                timeframe=self.timeframe,
                rules=self._rules,
            )
        return self._notifier

    # ================================================================
    # Chart HTTP server (/quote endpoint)
    # ================================================================

    def _start_chart_http_server(self, port: int) -> None:
        """启动 HTTP server 在后台线程，提供 GET /quote 返回 PNG。"""
        strategy_ref = self

        class _QuoteHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # 静默默认 access log
                pass

            def _send_json(self, code: int, msg: str) -> None:
                body = json.dumps({"error": msg}).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/quote":
                    self._send_json(404, "Not found")
                    return
                qs = parse_qs(parsed.query)
                pair = (qs.get("pair", [""])[0] or "").upper()
                tf = qs.get("tf", [""])[0] or strategy_ref.timeframe
                try:
                    n = int(qs.get("n", ["20"])[0])
                except ValueError:
                    self._send_json(400, "n must be integer")
                    return
                if not pair:
                    self._send_json(400, "pair is required")
                    return
                if n < 1 or n > 200:
                    self._send_json(400, "n must be 1-200")
                    return

                try:
                    df = strategy_ref.dp.get_pair_dataframe(pair, tf) if strategy_ref.dp else None
                except Exception:
                    df = None
                if df is None or df.empty:
                    self._send_json(404, f"No data for {pair} {tf}")
                    return

                try:
                    png = strategy_ref._generate_chart(pair, tf, df, n)
                except ValueError as e:
                    self._send_json(404, str(e))
                    return
                except Exception as e:
                    self._send_json(500, f"chart error: {e}")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)

        server = ThreadingHTTPServer(("0.0.0.0", port), _QuoteHandler)
        self._chart_http_server = server
        threading.Thread(target=server.serve_forever, daemon=True, name="chart-http").start()
        logger.info("Chart HTTP server listening on port %d (/quote)", port)

    def _init_default_pairs(self) -> None:
        """将默认标的写入 watch_pair 表（如不存在）。"""
        repository = self._get_repository()
        if not repository:
            return
        exchange_name = self.config.get("exchange", {}).get("name", "")
        if exchange_name == "ashare":
            pairs = self.config.get("exchange", {}).get("pair_whitelist", [])
            market = "ashare"
        else:
            pairs = DEFAULT_PAIRS
            market = "crypto"
        self._market = market
        repository.market = market

        display_name_fetcher = None
        if market == "ashare":
            from freqtrade.exchange.ashare import fetch_ashare_name
            display_name_fetcher = fetch_ashare_name
        repository.init_default_pairs(
            pairs,
            market=market,
            display_name_fetcher=display_name_fetcher,
        )

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
        dataframe = self._detect_ema20_cross(dataframe)
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

            # EMA20 穿越检测是状态提醒,独立于 signal-bar follow-through。
            cross = last.get("ema20_cross", "none")
            if cross != "none":
                self._notify_ema_cross(pair, last, dataframe, cross)

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
        return self._rules.calc_basic_indicators(df)

    def _calc_ema_atr(self, df: DataFrame) -> DataFrame:
        """EMA20 和 ATR14 — signal-bar-spec.md §4.3"""
        return self._rules.calc_ema_atr(df)

    def _detect_special_bars(self, df: DataFrame) -> DataFrame:
        """特殊K线类型检测 — signal-bar-spec.md §3"""
        return self._rules.detect_special_bars(df)

    def _classify_signal_quality(self, df: DataFrame) -> DataFrame:
        """信号K线质量分级 — V2 收紧阈值 + body_ratio 过滤"""
        return self._rules.classify_signal_quality(df)

    def _evaluate_context(self, df: DataFrame) -> DataFrame:
        """背景评估指标 — signal-bar-spec.md §4"""
        return self._rules.evaluate_context(df)

    def _detect_ema20_cross(self, df: DataFrame) -> DataFrame:
        """标记 EMA20 穿越:上穿(long)/ 下穿(short)/ 无(none)。

        上穿:前一根 close < ema20,当前 close > ema20
        下穿:前一根 close > ema20,当前 close < ema20
        首行因 shift(1) 产生 NaN,被视为无穿越。
        """
        return self._rules.detect_ema20_cross(df)

    # ================================================================
    # 通知 + PG 持久化
    # ================================================================

    def _ema_context_ok(self, row: pd.Series, direction: str) -> bool:
        """EMA20 背景过滤: 方向一致性 + ema_gap 限制。"""
        return self._rules.ema_context_ok(row, direction)

    def _candidate_signal_ok(self, row: pd.Series) -> bool:
        """Return whether a row is a signal-bar candidate worth tracking."""
        return self._rules.candidate_signal_ok(row)

    def _follow_through_ok(self, candidate: pd.Series, future: DataFrame) -> bool:
        """Brooks-style confirmation: signal-bar extreme breaks and is not quickly rejected."""
        return self._rules.follow_through_ok(candidate, future)

    def _confirm_recent_candidates(self, pair: str, dataframe: DataFrame) -> None:
        """Confirm prior candidates once follow-through appears within 3 bars."""
        if len(dataframe) < 2:
            return

        current_pos = len(dataframe) - 1
        start_pos = max(0, current_pos - self.FOLLOW_THROUGH_WINDOW)
        for pos in range(start_pos, current_pos):
            candidate = dataframe.iloc[pos]
            if not self._candidate_signal_ok(candidate):
                continue

            future = dataframe.iloc[pos + 1: current_pos + 1]
            if len(future) > self.FOLLOW_THROUGH_WINDOW:
                future = future.iloc[: self.FOLLOW_THROUGH_WINDOW]
            if not self._follow_through_ok(candidate, future):
                continue

            row = candidate.copy()
            quality = row.get("signal_quality", "none")
            row["signal_type"] = f"confirmed_signal_bar_{quality}"
            msg = self._format_signal_message(pair, row)
            saved = self._save_signal(
                pair,
                row,
                f"follow_through_confirmed:{msg}",
                signal_type=f"confirmed_signal_bar_{quality}",
            )
            if saved:
                self._notify_tg_bot(pair, row, dataframe)

    def _check_and_notify(self, pair: str, last: pd.Series, dataframe: DataFrame) -> None:
        """Record candidates immediately; notify only after follow-through confirmation."""
        quality = last.get("signal_quality", "none")
        direction = last.get("signal_direction", "none")

        if quality != "none" and not self._ema_context_ok(last, direction):
            logger.debug("Signal filtered by EMA context: %s %s %s", pair, direction, quality)

        if self._candidate_signal_ok(last):
            logger.info(
                "Candidate signal waiting for follow-through: %s %s %s",
                pair, direction, quality,
            )

        self._confirm_recent_candidates(pair, dataframe)

    def _get_bar_types(self, row: pd.Series) -> list[str]:
        """收集当前K线的特殊类型标签。"""
        return self._rules.get_bar_types(row)

    def _save_signal(
        self, pair: str, row: pd.Series, reason: str,
        signal_type: str | None = None,
    ) -> bool:
        """将信号写入 PostgreSQL。

        :param signal_type: 自定义 signal_type(默认 signal_bar_{quality})
        :return: True when a new row was committed, False when skipped or duplicate.
        """
        repository = self._get_repository()
        if not repository:
            return False
        return repository.save_signal(
            pair,
            row,
            reason,
            signal_type=signal_type,
        )

    def _persist_kline(self, pair: str, dataframe: DataFrame) -> None:
        """持久化 Freqtrade 传入策略的 K 线到 PostgreSQL（批量 UPSERT）。

        Freqtrade 在刷新 OHLCV 缓存时已经按交易所能力丢弃未完成 K 线；
        策略 dataframe 的最后一根就是本轮可分析的最新闭合 K 线。
        因此这里不能再额外跳过最后一根，否则 pa_kline 会永久落后一根，
        小周期健康检查会在每根新 K 线后误报。
        """
        repository = self._get_repository()
        if not repository:
            return
        repository.persist_kline(pair, dataframe)

    # ================================================================
    # K 线图表生成
    # ================================================================

    def _generate_chart(
        self,
        pair: str,
        timeframe: str | DataFrame,
        dataframe: DataFrame | None = None,
        num_candles: int = 20,
    ) -> bytes:
        """生成最近 num_candles 根 K 线蜡烛图 + EMA20 + 成交量，返回 PNG bytes。

        EMA 在完整 dataframe 上算完再切片，避免边界 warmup 失真。
        """
        if dataframe is None:
            dataframe = timeframe  # type: ignore[assignment]
            timeframe = self.timeframe
        notifier = self._get_notifier()
        return notifier.generate_chart(pair, str(timeframe), dataframe, num_candles)

    # ================================================================
    # TG Bot 通知 (HTTP POST)
    # ================================================================

    def _notify_tg_bot(self, pair: str, row: pd.Series, dataframe: DataFrame) -> None:
        """POST 信号数据 + K线图表到独立 tg-bot 的 HTTP API。"""
        notifier = self._get_notifier()
        notifier.notify_tg_bot(
            pair,
            row,
            dataframe,
            chart_generator=self._generate_chart,
        )

    def _notify_ema_cross(
        self, pair: str, row: pd.Series, dataframe: DataFrame, direction: str,
    ) -> None:
        """EMA20 穿越信号:落库 + 复用 _notify_tg_bot 发 K 线图。

        :param direction: "long"(上穿)或 "short"(下穿)
        """
        reason = f"ema20_cross_{'up' if direction == 'long' else 'down'}"
        # 用副本设置 signal_type / direction / quality,避免污染原 row
        row = row.copy()
        row["signal_direction"] = direction
        row["signal_quality"] = "cross"
        row["signal_type"] = "ema20_cross"
        self._save_signal(pair, row, reason, signal_type="ema20_cross")
        self._notify_tg_bot(pair, row, dataframe)

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
