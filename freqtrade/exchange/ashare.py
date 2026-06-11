"""
A 股 Exchange 插件 — 仅监控模式

继承 freqtrade Exchange 基类，用腾讯/新浪 API 替代 ccxt 获取 A 股 OHLCV 数据。
不支持交易（下单/撤单/余额），仅用于 PriceActionMonitor 等监控策略。

数据源:
  - 日线 (1d): 腾讯财经 HTTP API (web.ifzq.gtimg.cn)
  - 分钟线 (1h): 新浪财经 HTTPS API (quotes.sina.cn)

配置示例 (config JSON):
  {
    "exchange": {
      "name": "ashare",
      "pair_whitelist": ["000001/SZ", "600519/SH"]
    },
    "stake_currency": "CNY",
    "dry_run": true
  }

品种格式: 代码/交易所 (如 000001/SZ, 600519/SH)
K 线周期: 1h, 1d
"""

import json
import logging
import time as _time
from datetime import datetime, time, timedelta, timezone
from types import SimpleNamespace

import httpx
import pandas as pd
from pandas import DataFrame

from freqtrade.constants import CandleType
from freqtrade.enums import TradingMode
from freqtrade.exchange.exchange import Exchange

logger = logging.getLogger(__name__)

# 腾讯 K 线 API（日线）
TENCENT_KLINE_URL = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

# 新浪分钟线 API（1h）
SINA_KLINE_URL = (
    "https://quotes.sina.cn/cn/api/jsonp_v2.php/callback/CN_MarketDataService.getKLineData"
)

# 支持的 timeframe
SUPPORTED_TIMEFRAMES = {"1h", "1d"}


class Ashare(Exchange):
    """
    A 股 Exchange 插件 — 仅监控模式。

    用腾讯/新浪 API 替代 ccxt 获取 A 股 OHLCV 数据。
    不支持交易（下单/撤单/余额）。
    """

    _ft_has = {
        "stoploss_on_exchange": False,
        "ws_enabled": False,
        "ohlcv_partial_candle": False,
        "trades_pagination": "no",
        "ccxt_futures_name": "spot",
    }

    _supported_trading_mode_margin_pairs = [
        (TradingMode.SPOT, "cross"),
    ]

    # ------------------------------------------------------------------
    # ccxt 初始化 mock
    # ------------------------------------------------------------------

    def _init_ccxt(self, exchange_config, sync, ccxt_kwargs):
        """Override: 不创建 ccxt 实例，返回满足 freqtrade 类型检查的 mock 对象。"""
        api = SimpleNamespace()
        api.name = "ashare"
        api.id = "ashare"
        api.has = {
            "fetchOHLCV": True,
            "fetchTicker": True,
            "fetchL2OrderBook": False,
            "createOrder": False,
            "cancelOrder": False,
            "fetchOrder": False,
            "fetchBalance": False,
        }
        api.timeframes = {"1h": "1h", "1d": "1d"}
        api.options = {}
        api.precisionMode = 2  # SIGNIFICANT_DIGITS
        api.session = None
        api.close = lambda: None
        return api

    # ------------------------------------------------------------------
    # 配置验证
    # ------------------------------------------------------------------

    def validate_config(self, config):
        """Override: 只验证 timeframe，跳过交易相关验证。"""
        self.validate_timeframes(config.get("timeframe"))

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def reload_markets(self, refresh=False, load_leverage_tiers=False):
        """Override: 从 config.pair_whitelist 构建 markets dict。"""
        exchange_conf = self._config.get("exchange", {})
        pairs = exchange_conf.get("pair_whitelist", [])

        self._markets = {}
        for pair in pairs:
            code, exchange_suffix = _parse_pair(pair)
            self._markets[pair] = {
                "base": code,
                "quote": exchange_suffix,
                "symbol": pair,
                "active": True,
                "spot": True,
                "future": False,
            }

        self._last_markets_refresh = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        logger.info("A-share markets loaded: %d pairs", len(self._markets))

    # ------------------------------------------------------------------
    # OHLCV 数据获取
    # ------------------------------------------------------------------

    def refresh_latest_ohlcv(self, pair_list):
        """Override: 用腾讯/新浪 API 获取 A 股 K 线数据，写入 _klines 缓存。

        含重试机制（最多 3 次，指数退避）和 pair 间请求间隔。
        失败时保留旧缓存。
        """
        max_retries = 3
        for pair, timeframe, candle_type in pair_list:
            if timeframe not in SUPPORTED_TIMEFRAMES:
                logger.warning("Unsupported timeframe for A-share: %s", timeframe)
                continue

            df = None
            for attempt in range(max_retries):
                try:
                    if timeframe == "1d":
                        df = _fetch_tencent_daily(pair)
                    elif timeframe == "1h":
                        df = _fetch_sina_1h(pair)
                    break
                except Exception:
                    if attempt < max_retries - 1:
                        delay = 3 * (attempt + 1)
                        logger.warning(
                            "A-share fetch retry %d/%d for %s %s (wait %ds)",
                            attempt + 1, max_retries, pair, timeframe, delay,
                        )
                        _time.sleep(delay)
                    else:
                        logger.warning(
                            "Failed to fetch A-share OHLCV for %s %s after %d retries",
                            pair, timeframe, max_retries,
                        )

            if df is not None and len(df) > 0:
                self._klines[(pair, timeframe, candle_type)] = df
                self._pairs_last_refresh_time[(pair, timeframe, candle_type)] = int(
                    datetime.now(tz=timezone.utc).timestamp() * 1000
                )
                logger.debug(
                    "A-share OHLCV refreshed: %s %s (%d candles)",
                    pair, timeframe, len(df),
                )
                _time.sleep(1)  # pair 间请求间隔，避免触发反爬

    # ------------------------------------------------------------------
    # exchange_has / 属性
    # ------------------------------------------------------------------

    def exchange_has(self, feature):
        """Override: 报告支持的能力。"""
        has_map = {
            "fetchOHLCV": True,
            "watchOHLCV": False,
            "fetchTicker": True,
            "createOrder": False,
            "cancelOrder": False,
            "fetchOrder": False,
            "fetchBalance": False,
        }
        return has_map.get(feature, False)

    @property
    def name(self):
        return "ashare"

    @property
    def id(self):
        return "ashare"

    @property
    def timeframes(self):
        return list(SUPPORTED_TIMEFRAMES)


# ======================================================================
# 模块级工具函数（方便测试）
# ======================================================================


def _parse_pair(pair: str) -> tuple[str, str]:
    """解析品种标识: '000001/SZ' → ('000001', 'SZ')"""
    parts = pair.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid A-share pair format: {pair}, expected 'CODE/EXCHANGE'")
    return parts[0], parts[1]


def _to_tencent_symbol(pair: str) -> str:
    """将 freqtrade pair 转为腾讯 API 格式: '000001/SZ' → 'sz000001'"""
    code, exchange = _parse_pair(pair)
    return f"{exchange.lower()}{code}"


def _to_sina_symbol(pair: str) -> str:
    """将 freqtrade pair 转为新浪 API 格式: '000001/SZ' → 'sz000001'"""
    code, exchange = _parse_pair(pair)
    return f"{exchange.lower()}{code}"


def _is_market_open() -> bool:
    """判断当前是否在 A 股交易时段内（北京时间）。"""
    now = datetime.now(tz=timezone(timedelta(hours=8)))
    if now.weekday() >= 5:
        return False
    t = now.time()
    morning = time(9, 30) <= t <= time(11, 30)
    afternoon = time(13, 0) <= t <= time(15, 0)
    return morning or afternoon


def _fetch_tencent_daily(pair: str, count: int = 500, adjust: str = "qfq") -> DataFrame | None:
    """
    从腾讯 API 获取日线 K 线数据。

    :param pair: freqtrade pair，如 "000001/SZ"
    :param count: 返回 K 线数量
    :param adjust: 复权类型 "qfq"(前复权), ""(不复权)
    :return: DataFrame with columns [date, open, high, low, close, volume] or None
    """
    symbol = _to_tencent_symbol(pair)
    params = {"param": f"{symbol},day,,,{count},{adjust}"}
    with httpx.Client(follow_redirects=True, timeout=10) as client:
        resp = client.get(TENCENT_KLINE_URL, params=params)
        data = resp.json()

    key = "qfqday" if adjust == "qfq" else "day"
    bars = data.get("data", {}).get(symbol, {}).get(key, [])
    if not bars:
        return None

    df = pd.DataFrame(bars, columns=["date", "open", "close", "high", "low", "volume"])
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fetch_sina_1h(pair: str, count: int = 500) -> DataFrame | None:
    """
    从新浪财经 API 获取 1h 分钟线数据。

    :param pair: freqtrade pair，如 "000001/SZ"
    :param count: 返回 K 线数量
    :return: DataFrame with columns [date, open, high, low, close, volume] or None
    """
    symbol = _to_sina_symbol(pair)
    params = {"symbol": symbol, "scale": "60", "ma": "no", "datalen": str(count)}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.sina.com.cn/",
    }
    with httpx.Client(follow_redirects=True, timeout=10, headers=headers) as client:
        resp = client.get(SINA_KLINE_URL, params=params)
        text = resp.text

    # 新浪返回 JSONP: callback({...})
    json_str = text[text.index("(") + 1 : text.rindex(")")]
    bars = json.loads(json_str)
    if not bars:
        return None

    df = pd.DataFrame(bars)
    df = df[["day", "open", "high", "low", "close", "volume"]]
    df = df.rename(columns={"day": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
