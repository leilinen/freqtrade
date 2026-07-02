"""
价格行为 LLM 决策盯盘策略 (Price Action LLM Decision Monitor)

基于 Al Brooks 价格行为学 + PA 分析管道
（特征工程 -> 市场诊断 -> 策略路由 -> 交易决策），
架构对齐 PA_Agent。每根新 K 收盘后在后台异步运行 LLM 分析，
产出交易决策，写入 PostgreSQL（pa_analysis 表），
并通过 HTTP POST 推送到独立 tg-bot 的 /decision 端点。
纯盯盘工具，不执行实际交易。

LLM 调用使用 OpenAI SDK 兼容方式（base_url + api_key 可配置，默认 DeepSeek），
api_key 优先从 config 读，其次 DEEPSEEK_API_KEY 环境变量。

配置要求 (config JSON):
  pa_db_url: PostgreSQL 连接串
    例: "postgresql://postgres:postgres@localhost:15432/freqtrade_monitor"
  tg_api_url: TG Bot HTTP API 地址
    例: "http://tg-bot:8090"
  pa_agent_enabled: 是否启用 LLM 管道（默认 true，需 api_key 才真正生效）
  pa_llm_base_url / pa_llm_model / pa_llm_temperature / pa_llm_timeout:
    LLM 服务地址、模型、采样温度、超时（默认 DeepSeek deepseek-chat）
  pa_llm_window / pa_llm_warmup:
    特征工程取最近 N 根已收盘 K、warmup 预热根数（默认 30 / 50）
  pa_experience_limit: 经验库检索条数（默认 3）
  pa_validation_retry_max: LLM JSON 校验失败后的自动重试次数（默认 0，最大 3）
  pa_incremental_stage1_enabled:
    是否复用上一轮成功记录做阶段一增量分析（默认 true）
  pa_incremental_stage1_max_new_bars: 增量阶段一允许的最大新增 K 线数（默认 10）
  pa_decision_stance: 交易倾向 conservative/balanced/aggressive/extreme_aggressive
  pa_notify_wait: wait/avoid 是否也推送（默认 false，仅 enter 推送）
"""

import atexit
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import requests as http_requests
from pandas import DataFrame
from price_action.background import MarketBackgroundAnalyzer
from price_action.features import calculate_atr, calculate_ema
from price_action.llm import LlmClient, build_llm_client
from price_action.models import PaKline, WatchPair, _Base
from price_action.notification import SignalNotifier
from price_action.orchestrator import PriceActionOrchestrator
from price_action.repository import PriceActionRepository
from price_action.worker import PaAnalysisWorker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)

__all__ = [
    "PaKline",
    "PriceActionMonitor",
    "WatchPair",
    "_Base",
    "http_requests",
]

DEFAULT_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]


class PriceActionMonitor(IStrategy):
    """
    价格行为 LLM 决策盯盘策略。

    每根新 K 收盘后，在后台异步运行 PA 分析管道：
    纯 Python 特征工程 -> 市场诊断 -> 本地策略路由
    -> 交易决策（四重校验）。
    决策写入 PostgreSQL（pa_analysis 表），仅 enter_long/enter_short
    推送 tg-bot /decision。

    盯盘标的从 PG watch_pair 表动态读取。必须以 dry_run: true 模式运行，
    config 中需设置 pa_db_url；启用 LLM 管道需配置 pa_llm_api_key 或注入
    DEEPSEEK_API_KEY 环境变量。
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
    startup_candle_count: int = 120

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._pg_engine = None
        self._pg_session_factory = None
        self._chart_http_server = None
        self._market = "crypto"
        self._background = MarketBackgroundAnalyzer()
        self._repository: PriceActionRepository | None = None
        self._notifier: SignalNotifier | None = None
        self._llm_client: LlmClient | None = None
        self._orchestrator: PriceActionOrchestrator | None = None
        self._worker: PaAnalysisWorker | None = None

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
        )
        self._notifier = SignalNotifier(
            self._pg_session_factory,
            config=self.config,
            timeframe=self.timeframe,
        )
        atexit.register(self._cleanup)

        # 初始化默认标的
        self._init_default_pairs()

        logger.info("PG persistence initialized: %s", db_url)

        # 初始化 PA LLM 决策管道
        self._init_pa_pipeline()

        # 启动 /quote HTTP server（供 tg-bot 调用获取 K 线图）
        chart_port = self.config.get("pa_chart_port")
        if chart_port:
            try:
                self._start_chart_http_server(int(chart_port))
            except Exception:
                logger.warning(
                    "Failed to start chart HTTP server on port %s",
                    chart_port,
                    exc_info=True,
                )

    def _init_pa_pipeline(self) -> None:
        """构建 LLM client + Orchestrator + Worker；无 api_key 时优雅降级。"""
        if not self.config.get("pa_agent_enabled", True):
            logger.info("PA LLM pipeline disabled by config (pa_agent_enabled=false)")
            return
        try:
            client = build_llm_client(self.config)
        except Exception:
            logger.warning("PA LLM client build failed; pipeline disabled", exc_info=True)
            return
        # 没有可用 api_key 时不启用管道（DeepSeek 调用会失败）
        if not client.api_key:
            logger.warning(
                "PA LLM pipeline disabled: no api_key (set pa_llm_api_key or DEEPSEEK_API_KEY)"
            )
            return
        self._llm_client = client
        self._orchestrator = PriceActionOrchestrator(
            repository=self._repository,
            llm_client=client,
            config=self.config,
        )
        self._worker = PaAnalysisWorker(
            self._orchestrator,
            max_workers=int(self.config.get("pa_agent_workers", 1)),
        )
        self._worker.start()
        logger.info(
            "PA LLM pipeline started: model=%s base_url=%s",
            client.model,
            client.base_url,
        )

    def _cleanup(self) -> None:
        if self._worker is not None:
            try:
                self._worker.stop()
            except Exception:
                logger.warning("Failed to stop PA worker", exc_info=True)
            self._worker = None
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
        """Optionally seed watch_pair rows for local smoke tests.

        Production monitoring symbols are managed by tg-bot commands and stored
        in watch_pair. Do not insert defaults unless explicitly requested.
        """
        repository = self._get_repository()
        if not repository:
            return
        exchange_name = self.config.get("exchange", {}).get("name", "")
        if exchange_name == "ashare":
            market = "ashare"
        else:
            market = "crypto"
        self._market = market
        repository.market = market
        if not self.config.get("pa_seed_default_pairs", False):
            logger.info(
                "watch_pair seeding disabled; symbols are managed by tg-bot"
            )
            return

        if market == "ashare":
            pairs = self.config.get("exchange", {}).get("pair_whitelist", [])
        else:
            pairs = DEFAULT_PAIRS

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
        """计算价格行为几何指标（特征工程输入）+ 市场背景。

        不再产出信号；信号完全由 LLM 交易决策给出。
        """
        dataframe = self._calc_basic_indicators(dataframe)
        dataframe = self._calc_ema_atr(dataframe)
        dataframe = self._evaluate_background(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """新 K 收盘后：提交 LLM 分析任务 + 持久化 K 线。"""
        if len(dataframe) == 0:
            return dataframe

        # 只在 live/dry_run 模式下运行管道
        if self.dp and self.dp.runmode.value in ("live", "dry_run"):
            pair = metadata.get("pair", "Unknown")
            self._submit_pa_analysis(pair, dataframe)
            self._persist_kline(pair, dataframe)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """空实现，不产生退出信号。"""
        return dataframe

    # ================================================================
    # 指标计算
    # ================================================================

    def _calc_basic_indicators(self, df: DataFrame) -> DataFrame:
        """K 线基础几何指标：body/range/影线/方向。"""
        df = df.copy()
        df["body"] = (df["close"] - df["open"]).abs()
        df["range"] = df["high"] - df["low"]
        return df

    def _calc_ema_atr(self, df: DataFrame) -> DataFrame:
        """EMA20 和 ATR14，使用与特征工程一致的纯 pandas 实现。

        统一这一处指标计算，避免特征工程与策略背景判断用两套
        EMA/ATR 导致不一致。
        """
        df = df.copy()
        df["ema20"] = calculate_ema(df["close"], 20)
        df["atr14"] = calculate_atr(df, 14)
        return df

    def _evaluate_background(self, df: DataFrame) -> DataFrame:
        """PA_AGENT-inspired multi-window market background detection."""
        return self._background.evaluate(df)

    # ================================================================
    # LLM 决策管道
    # ================================================================

    def _submit_pa_analysis(self, pair: str, dataframe: DataFrame) -> None:
        """把最新已收盘 K 提交给后台 worker 跑 PA 分析管道。"""
        if self._worker is None:
            return
        submitted = self._worker.submit(
            symbol=pair,
            dataframe=dataframe,
            timeframe=self.timeframe,
            market=self._market,
            notifier=self._get_notifier(),
            chart_generator=self._generate_chart,
        )
        if submitted:
            logger.info("PA analysis submitted: %s %s", pair, self.timeframe)

    # ================================================================
    # PG 持久化
    # ================================================================

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
