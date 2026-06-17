"""
独立 Telegram Bot 服务 — 价格行为信号盯盘

功能：
1. HTTP API 接收 freqtrade 策略推送的信号通知并转发到 Telegram
2. 处理用户命令：/pa_watch, /pa_add, /pa_remove, /pa_signals, /pa_help
3. 直接读写 PostgreSQL (freqtrade_monitor) 管理标的和查询信号

环境变量：
  TG_TOKEN    — Telegram Bot Token
  TG_CHAT_ID  — 授权的 Chat ID
  DB_URL      — PostgreSQL 连接串
  API_PORT    — HTTP API 端口 (默认 8090)
"""

import json
import logging
import os
import asyncio
import threading
from datetime import datetime, timezone, timedelta, time as dt_time

import httpx
from aiohttp import web
from sqlalchemy import create_engine, text
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT_ID = int(os.environ["TG_CHAT_ID"])
DB_URL = os.environ.get("DB_URL", "postgresql://postgres:postgres@postgres:5432/freqtrade_monitor")
API_PORT = int(os.environ.get("API_PORT", "8090"))

db_engine = create_engine(DB_URL)


# ================================================================
# Database helpers
# ================================================================

def db_get_watch_pairs() -> list[dict]:
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT symbol, enabled, market, display_name FROM watch_pair ORDER BY id")
        ).fetchall()
    return [
        {"symbol": r[0], "enabled": r[1], "market": r[2], "display_name": r[3]}
        for r in rows
    ]


VALID_MARKETS = {"crypto", "ashare", "usstock"}


def _detect_market(symbol: str) -> str:
    """根据标的符号自动判断市场类型。"""
    if symbol.endswith("/SH") or symbol.endswith("/SZ"):
        return "ashare"
    if "/" in symbol:
        return "crypto"
    return "usstock"


def _fetch_ashare_name(symbol: str) -> str | None:
    """获取 A 股中文显示名称，新浪为主、腾讯兜底。

    tg_bot 容器不安装 freqtrade，无法 import ashare.py，按现有 _refresh_trade_days
    直接打 HTTP 接口的模式独立实现。
    """
    parts = symbol.split("/")
    if len(parts) != 2:
        return None
    code, exch = parts[0], parts[1].lower()
    sina_symbol = f"{exch}{code}"

    # --- 新浪 ---
    headers = {"Referer": "https://finance.sina.com.cn/", "User-Agent": "Mozilla/5.0"}
    try:
        with httpx.Client(follow_redirects=True, timeout=10, headers=headers) as client:
            text = client.get(f"https://hq.sinajs.cn/list={sina_symbol}").text
        start = text.index('"') + 1
        end = text.rindex('"')
        content = text[start:end]
        if content:
            name = content.split(",")[0]
            if name:
                return name
    except Exception:
        logger.warning("Sina name lookup failed for %s", symbol, exc_info=True)

    # --- 腾讯兜底 ---
    headers = {"Referer": "https://gu.qq.com/", "User-Agent": "Mozilla/5.0"}
    try:
        with httpx.Client(follow_redirects=True, timeout=10, headers=headers) as client:
            text = client.get(f"https://qt.gtimg.cn/q={sina_symbol}").text
        start = text.index('"') + 1
        end = text.rindex('"')
        content = text[start:end]
        if content:
            fields = content.split("~")
            if len(fields) >= 2 and fields[1]:
                return fields[1]
    except Exception:
        logger.warning("Tencent name lookup failed for %s", symbol, exc_info=True)
    return None


# ================================================================
# A-share trade-day calendar (sourced from sh000001 daily kline)
# ================================================================

ASHARE_INDEX_SYMBOL = "sh000001"
ASHARE_CALENDAR_TZ = timezone(timedelta(hours=8))  # Beijing

# A-share trade-day set, refreshed once per Beijing-day from sh000001 daily kline.
# On fetch failure, keeps the last successful cache (or empty set on first run).
_trade_days_cache: set[str] = set()         # {"YYYY-MM-DD", ...}
_trade_days_fetched_for: str | None = None  # Beijing date we last refreshed for
_trade_days_lock = threading.Lock()


def _refresh_trade_days(force: bool = False) -> set[str]:
    """Refresh and return the A-share trade-day set from Tencent sh000001 daily kline.

    Cached per Beijing-day; thread-safe. The returned dates are the trade
    calendar itself (weekends and holidays are absent, 调休 days are present).
    On error, returns the last successful cache (or empty set if never fetched) —
    callers must treat an empty set as 'unknown, fall back to weekday logic'.

    :param force: bypass the per-day cache and refetch
    :return: set of "YYYY-MM-DD" trade-day strings (may be empty on failure)
    """
    global _trade_days_cache, _trade_days_fetched_for
    today_bj = datetime.now(ASHARE_CALENDAR_TZ).strftime("%Y-%m-%d")
    with _trade_days_lock:
        if _trade_days_fetched_for == today_bj and not force and _trade_days_cache:
            return _trade_days_cache
    try:
        with httpx.Client(follow_redirects=True, timeout=10) as client:
            resp = client.get(
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                params={"param": f"{ASHARE_INDEX_SYMBOL},day,,,500,qfq"},
            )
            data = resp.json().get("data", {}).get(ASHARE_INDEX_SYMBOL, {})
            # 腾讯对不同标的返回的 key 不一致（大盘→qfqday，科创板/创业板→day），与 ashare.py 保持一致
            bars = data.get("qfqday", []) or data.get("day", [])
            days = {b[0] for b in bars if b and b[0]}
        if days:
            with _trade_days_lock:
                _trade_days_cache = days
                _trade_days_fetched_for = today_bj
            logger.info(
                "A-share trade calendar refreshed: %d trading days (last=%s)",
                len(days),
                max(days),
            )
    except Exception:
        logger.warning(
            "Failed to refresh A-share trade calendar, using cached %d days",
            len(_trade_days_cache),
            exc_info=True,
        )
    return _trade_days_cache


def is_ashare_trading_day(now_utc: datetime) -> tuple[bool, str]:
    """Determine whether now (Beijing) falls on an A-share trading day.

    :param now_utc: current time in UTC
    :return: (is_trading_day, reason) where reason is one of:
        "trading-day"     — calendar confirms today is a trade day
        "休市日"           — weekday holiday (calendar knows it's closed)
        "周末"             — Saturday/Sunday (calendar agrees)
        "fallback-weekday" — calendar unavailable, weekday<5 used as best guess
    """
    today_bj = now_utc.astimezone(ASHARE_CALENDAR_TZ).strftime("%Y-%m-%d")
    days = _refresh_trade_days()
    if days and today_bj in days:
        return True, "trading-day"
    if not days:
        # Calendar unavailable — fall back to plain weekday check (old behavior)
        is_wd = now_utc.astimezone(ASHARE_CALENDAR_TZ).weekday() < 5
        return is_wd, "fallback-weekday"
    # Today not in trade-day set
    wd = now_utc.astimezone(ASHARE_CALENDAR_TZ).weekday()
    return False, "休市日" if wd < 5 else "周末"


def _health_threshold(market: str, tf: str, now_utc: datetime) -> tuple[int, str]:
    """根据市场和当前时间返回健康检查阈值（小时）及说明。

    A 股：用 sh000001 交易日历判定是否交易日；非交易日放宽阈值，避免误报。
    crypto 全天交易，使用固定阈值。
    """
    if market == "ashare":
        tz_sh = timezone(timedelta(hours=8))
        local_now = now_utc.astimezone(tz_sh)
        t = local_now.time()
        is_trade_day, day_reason = is_ashare_trading_day(now_utc)
        is_trading = is_trade_day and (
            dt_time(9, 30) <= t <= dt_time(11, 30)
            or dt_time(13, 0) <= t <= dt_time(15, 0)
        )
        if not is_trading:
            # 非交易时段：用日历的 day_reason（休市日 / 周末 / fallback-weekday）
            # 作为说明，方便从告警里看出到底是节假日还是普通盘后
            note = day_reason if day_reason != "trading-day" else "非交易时段"
            return 72, note
        if tf == "1h":
            return 5, ""
        if tf == "1d":
            return 36, ""
        return 48, ""
    # crypto
    if tf == "1h":
        return 2, ""
    if tf == "4h":
        return 9, ""
    return 24, ""


def db_add_pair(symbol: str, market: str) -> str:
    symbol = symbol.upper()
    with db_engine.begin() as conn:
        existing = conn.execute(
            text("SELECT enabled FROM watch_pair WHERE symbol = :s"), {"s": symbol}
        ).fetchone()
        if existing:
            if existing[0]:
                return f"{symbol} already watching"
            conn.execute(text("UPDATE watch_pair SET enabled = true WHERE symbol = :s"), {"s": symbol})
            return f"{symbol} re-enabled"
        display_name = None
        if market == "ashare":
            display_name = _fetch_ashare_name(symbol)
        conn.execute(
            text(
                "INSERT INTO watch_pair (symbol, enabled, market, display_name) "
                "VALUES (:s, true, :m, :dn)"
            ),
            {"s": symbol, "m": market, "dn": display_name},
        )
    suffix = f" ({display_name})" if display_name else f" ({market})"
    return f"{symbol} added{suffix}"


def db_remove_pair(symbol: str) -> str:
    symbol = symbol.upper()
    with db_engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM watch_pair WHERE symbol = :s"), {"s": symbol}
        ).fetchone()
        if not row:
            return f"{symbol} not found"
        conn.execute(text("UPDATE watch_pair SET enabled = false WHERE symbol = :s"), {"s": symbol})
    return f"{symbol} disabled"


def db_get_signals(limit: int = 10) -> list[dict]:
    limit = min(limit, 30)
    with db_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT symbol, timeframe, direction, quality, body_pct, "
                "candle_time, bar_types FROM pa_signal "
                "ORDER BY created_at DESC LIMIT :limit"
            ),
            {"limit": limit},
        ).fetchall()
    return [
        {
            "symbol": r[0],
            "timeframe": r[1],
            "direction": r[2],
            "quality": r[3],
            "body_pct": r[4],
            "candle_time": r[5],
            "bar_types": r[6],
        }
        for r in rows
    ]


# ================================================================
# Signal notification (HTTP POST handler)
# ================================================================

async def handle_signal(request: web.Request) -> web.Response:
    """HTTP POST /signal — 接收 freqtrade 推送的信号 + K线图表并转发到 TG。"""
    chart_bytes = None
    payload_str = None

    # 支持 multipart/form-data（带图表）和 JSON（纯文字，向后兼容）
    content_type = request.content_type
    if content_type and "multipart" in content_type:
        reader = await request.multipart()
        async for part in reader:
            if part.name == "payload":
                payload_str = await part.text()
            elif part.name == "chart":
                chart_bytes = await part.read()
    else:
        try:
            payload_str = await request.text()
        except Exception:
            return web.json_response({"error": "invalid request"}, status=400)

    try:
        data = json.loads(payload_str) if payload_str else {}
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    msg = format_signal_message(data)
    try:
        if chart_bytes:
            # aiohttp multipart part.read() returns bytearray, but python-telegram-bot
            # only treats `bytes` as a file upload. Convert to avoid Telegram treating
            # it as a string file_id ("Wrong remote file identifier" error).
            photo = bytes(chart_bytes)
            try:
                await request.app["tg_bot"].bot.send_photo(
                    chat_id=TG_CHAT_ID,
                    photo=photo,
                    caption=msg,
                )
            except Exception:
                logger.warning("send_photo failed, falling back to text for %s", data.get("symbol"), exc_info=True)
                await request.app["tg_bot"].bot.send_message(
                    chat_id=TG_CHAT_ID,
                    text=msg,
                )
        else:
            await request.app["tg_bot"].bot.send_message(
                chat_id=TG_CHAT_ID,
                text=msg,
            )
        logger.info("Signal pushed to TG: %s %s %s", data.get("symbol"), data.get("direction"), data.get("quality"))
    except Exception:
        logger.exception("Failed to push signal to TG")

    return web.json_response({"ok": True})


def format_signal_message(data: dict) -> str:
    """格式化信号推送消息（中文）。"""
    direction = data.get("direction", "?").lower()
    quality = data.get("quality", "?")
    quality_map = {"good": "Good", "acceptable": "Acceptable", "fair": "Fair"}
    q_str = quality_map.get(quality, quality)

    dir_cn = "做多" if direction == "long" else "做空"
    arrow = "+" if direction == "long" else "-"
    symbol = data.get("symbol", "?")
    tf = data.get("timeframe", "?")
    display_name = data.get("display_name")
    label = f"{display_name}({symbol})" if display_name else symbol
    # 信号 K 线时间 → 北京时间显示
    time_str = ""
    raw_time = data.get("signal_time")
    if raw_time:
        try:
            ct = datetime.fromisoformat(str(raw_time))
            if ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            time_str = ct.astimezone(timezone(timedelta(hours=8))).strftime("%m-%d %H:%M")
        except (ValueError, TypeError):
            time_str = ""
    header = f"{arrow} {label} {tf}"
    if time_str:
        header += f" @{time_str}"
    price = data.get("entry_price", 0)
    sl = data.get("stop_loss", 0)
    tp = data.get("target_price", 0)

    # 价格精度：根据价格大小自动选择小数位
    def fmt_price(v):
        if v >= 1000:
            return f"{v:.2f}"
        elif v >= 1:
            return f"{v:.4f}"
        else:
            return f"{v:.6f}"

    lines = [
        header,
        f"{dir_cn} [{q_str}] 当前价格: {fmt_price(price)}",
        f"实体占比={data.get('body_pct', 0):.2f} "
        f"收盘位置={data.get('close_location', 0):.2f} "
        f"实体比={data.get('body_ratio', 0):.1f}",
    ]

    if sl and tp:
        risk = abs(price - sl)
        if risk > 0:
            rr = abs(tp - price) / risk
            lines.append(f"止损 {fmt_price(sl)} | 目标 {fmt_price(tp)} (盈亏比 {rr:.1f})")

    bar_types = data.get("bar_types", [])
    if bar_types:
        type_map = {"surprise": "惊喜", "engulfing": "吞噬", "inside": "内包", "2k_reversal": "2K反转", "doji": "十字星"}
        type_names = [type_map.get(t, t) for t in bar_types]
        lines.append("特殊K线: " + " | ".join(type_names))

    above = data.get("ema20_above", None)
    if above is not None:
        ema_str = "均线上方" if above else "均线下方"
        lines.append(f"EMA20: {ema_str} (偏离={data.get('ema_gap', 0):.1f}x ATR)")

    bs = data.get("bull_strength_5", 0.5)
    if bs > 0.6:
        bias = "多头"
    elif bs < 0.4:
        bias = "空头"
    else:
        bias = "中性"
    lines.append(f"近5K偏向: {bias} ({bs:.0%})")

    return "\n".join(lines)


# ================================================================
# TG Command handlers
# ================================================================

def authorized(update: Update) -> bool:
    return update.effective_chat and update.effective_chat.id == TG_CHAT_ID


# ---- /quote 路由表: (market, timeframe) -> (容器名, HTTP 端口) ----
# 每个 freqtrade 容器只缓存自己 timeframe 的 OHLCV。
CHART_ROUTES = {
    ("crypto", "1h"): ("price-action-1h", 8091),
    ("crypto", "4h"): ("price-action-4h", 8092),
    ("ashare", "1h"): ("ashare-1h", 8093),
    ("ashare", "1d"): ("ashare-1d", 8094),
}


def _route_chart(symbol: str, timeframe: str) -> tuple[str, int] | None:
    """根据 symbol 后缀 + timeframe 找到对应的 freqtrade 容器。

    :return: (host, port) 或 None(不支持的路由)
    """
    is_ashare = symbol.endswith("/SH") or symbol.endswith("/SZ")
    market = "ashare" if is_ashare else "crypto"
    return CHART_ROUTES.get((market, timeframe))


async def quote(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /quote <pair> [timeframe] [n] — 从 freqtrade 容器拉 K 线图。"""
    if not authorized(update):
        return
    if not context.args:
        await update.message.reply_text(
            "用法: /quote <标的> [周期] [数量]\n"
            "默认: 周期=1h, 数量=20 根\n"
            "例: /quote BTC/USDT\n"
            "    /quote BTC/USDT 4h 50\n"
            "    /quote 510300/SH 1d\n"
            "支持: crypto(1h/4h) · A股(1h/1d)"
        )
        return

    pair = context.args[0].upper()
    timeframe = context.args[1] if len(context.args) > 1 else "1h"
    try:
        n = int(context.args[2]) if len(context.args) > 2 else 20
    except ValueError:
        await update.message.reply_text("数量必须是整数")
        return
    if n < 1 or n > 200:
        await update.message.reply_text("数量必须在 1-200 之间")
        return

    route = _route_chart(pair, timeframe)
    if route is None:
        await update.message.reply_text(
            f"不支持的周期: {timeframe}\n"
            "crypto 支持 1h/4h, A股 支持 1h/1d"
        )
        return
    host, port = route
    url = f"http://{host}:{port}/quote"
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, params={"pair": pair, "tf": timeframe, "n": n})
    except Exception as e:
        await update.message.reply_text(f"图表服务连接失败: {e}")
        return

    if resp.status_code != 200:
        try:
            err = resp.json().get("error", resp.text)
        except Exception:
            err = resp.text
        await update.message.reply_text(f"{pair} {timeframe}: {err}")
        return

    caption = f"{pair} {timeframe} · {n} 根 K 线"
    try:
        await update.message.reply_photo(photo=resp.content, caption=caption)
    except Exception:
        logger.warning("reply_photo failed for %s, falling back to text", pair, exc_info=True)
        await update.message.reply_text(f"{caption}\n(图片发送失败,见日志)")


async def pa_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    pairs = db_get_watch_pairs()
    if not pairs:
        await update.message.reply_text("暂无监控标的")
        return
    lines = ["监控标的列表:"]
    for p in pairs:
        status = "开" if p["enabled"] else "关"
        market = p.get("market") or "crypto"
        name = p.get("display_name")
        label = f"{name}({p['symbol']})" if name else p["symbol"]
        lines.append(f"  {label} [{status}] ({market})")
    await update.message.reply_text("\n".join(lines))


async def pa_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    if not context.args:
        await update.message.reply_text(
            "用法: /pa_add <标的> [市场]\n"
            "市场: crypto(默认) / ashare / usstock\n"
            "例: /pa_add BTC/USDT\n"
            "    /pa_add 510300/SH\n"
            "    /pa_add AAPL"
        )
        return
    symbol = context.args[0].upper()
    if len(context.args) > 1:
        market = context.args[1].lower()
        if market not in VALID_MARKETS:
            await update.message.reply_text(f"无效市场: {market}\n可选: {', '.join(sorted(VALID_MARKETS))}")
            return
    else:
        market = _detect_market(symbol)
    msg = db_add_pair(symbol, market)
    await update.message.reply_text(msg + "\n提示: 等待缓存刷新生效，或发送 /reload_config 立即生效")


async def pa_remove(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    if not context.args:
        await update.message.reply_text("用法: /pa_remove BTC/USDT")
        return
    symbol = context.args[0].upper()
    msg = db_remove_pair(symbol)
    await update.message.reply_text(msg)


async def pa_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    limit = 10
    if context.args:
        try:
            limit = int(context.args[0])
        except ValueError:
            await update.message.reply_text("用法: /pa_signals [数量]")
            return
    signals = db_get_signals(limit)
    if not signals:
        await update.message.reply_text("暂无信号记录")
        return
    tz_shanghai = timezone(timedelta(hours=8))
    lines = [f"最近 {len(signals)} 条信号:"]
    for s in signals:
        ct = s["candle_time"]
        if ct:
            if ct.tzinfo is None:
                ct = ct.replace(tzinfo=timezone.utc)
            time_str = ct.astimezone(tz_shanghai).strftime("%m/%d %H:%M")
        else:
            time_str = "?"
        dir_cn = "多" if s["direction"] == "long" else "空"
        lines.append(
            f"{time_str} {s['symbol']} {s['timeframe']} "
            f"{dir_cn} [{s['quality']}] "
            f"实体={s['body_pct']:.2f}"
        )
    await update.message.reply_text("\n".join(lines))


async def pa_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    await update.message.reply_text(
        "价格行为信号盯盘 命令:\n"
        "/pa_watch — 查看监控标的\n"
        "/pa_add <标的> [市场] — 添加标的\n"
        "  自动识别: BTC/USDT→crypto, 510300/SH→ashare, AAPL→usstock\n"
        "  手动指定: /pa_add AAPL usstock\n"
        "/pa_remove <标的> — 禁用标的\n"
        "/pa_signals [N] — 最近信号 (默认10条)\n"
        "/quote <标的> [周期] [N] — K线图 (默认 1h、20 根)\n"
        "  crypto: 1h/4h · A股: 1h/1d\n"
        "  例: /quote BTC/USDT 4h 50\n"
        "/pa_status — 服务健康状态\n"
        "/pa_help — 帮助信息"
    )


async def pa_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """检查服务健康状态：各 timeframe 最后信号时间，超时告警。"""
    if not authorized(update):
        return
    now = datetime.now(timezone.utc)
    lines = ["服务状态:"]
    has_alert = False

    # 按 market + timeframe 分组检查
    with db_engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT market, timeframe, MAX(candle_time) as last_candle "
            "FROM pa_signal GROUP BY market, timeframe ORDER BY market, timeframe"
        )).fetchall()

    tz_shanghai = timezone(timedelta(hours=8))
    for r in rows:
        market, tf = r[0], r[1]
        last_candle = r[2]
        if last_candle.tzinfo is None:
            last_candle = last_candle.replace(tzinfo=timezone.utc)
        age_hours = (now - last_candle).total_seconds() / 3600
        local_str = last_candle.astimezone(tz_shanghai).strftime("%m/%d %H:%M")

        # A 股非交易时段不算异常
        if market == "ashare":
            local_now = now.astimezone(tz_shanghai)
            weekday = local_now.weekday()
            t = local_now.time()
            is_trading = weekday < 5 and (
                dt_time(9, 30) <= t <= dt_time(11, 30)
                or dt_time(13, 0) <= t <= dt_time(15, 0)
            )
            if not is_trading:
                threshold = 72  # 非交易时段放宽到 72 小时
            elif tf == "1h":
                threshold = 5
            elif tf == "1d":
                threshold = 36
            else:
                threshold = 48
        else:
            # crypto 阈值
            if tf == "1h":
                threshold = 3
            elif tf == "4h":
                threshold = 12
            else:
                threshold = 24

        if age_hours > threshold:
            status = f"⚠ 超过 {age_hours:.0f}h 无信号"
            has_alert = True
        else:
            status = "正常"
        market_tag = "A股" if market == "ashare" else "Crypto"
        lines.append(f"  [{market_tag}] {tf}: 最后信号 {local_str} ({status})")

    # 检查信号总数
    with db_engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM pa_signal")).scalar()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    with db_engine.connect() as conn:
        today_count = conn.execute(text(
            "SELECT COUNT(*) FROM pa_signal WHERE created_at >= :ts"
        ), {"ts": today_start}).scalar()
    lines.append(f"\n今日信号: {today_count} | 总计: {total}")

    if has_alert:
        lines.append("\n建议检查 freqtrade 容器是否正常运行")

    await update.message.reply_text("\n".join(lines))


# ================================================================
# Main — start TG polling + HTTP API
# ================================================================

async def main() -> None:
    app_tg = Application.builder().token(TG_TOKEN).build()

    app_tg.add_handler(CommandHandler("pa_watch", pa_watch))
    app_tg.add_handler(CommandHandler("pa_add", pa_add))
    app_tg.add_handler(CommandHandler("pa_remove", pa_remove))
    app_tg.add_handler(CommandHandler("pa_signals", pa_signals))
    app_tg.add_handler(CommandHandler("pa_status", pa_status))
    app_tg.add_handler(CommandHandler("pa_help", pa_help))
    app_tg.add_handler(CommandHandler("quote", quote))

    # HTTP API
    app_http = web.Application()
    app_http["tg_bot"] = app_tg
    app_http.router.add_post("/signal", handle_signal)

    runner = web.AppRunner(app_http)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", API_PORT)
    await site.start()
    logger.info("HTTP API listening on port %d", API_PORT)

    # Start TG polling with retry on Conflict
    logger.info("Starting TG bot polling...")
    for attempt in range(10):
        try:
            await app_tg.initialize()
            await app_tg.start()
            await app_tg.updater.start_polling(drop_pending_updates=True)
            # 注册 Bot Commands 菜单
            from telegram import BotCommand
            await app_tg.bot.set_my_commands([
                BotCommand("pa_watch", "查看监控标的"),
                BotCommand("pa_add", "添加标的"),
                BotCommand("pa_remove", "禁用标的"),
                BotCommand("pa_signals", "最近信号"),
                BotCommand("quote", "查 K 线图"),
                BotCommand("pa_status", "服务健康状态"),
                BotCommand("pa_help", "帮助信息"),
            ])
            logger.info("TG bot polling started, commands registered")
            break
        except Exception as e:
            wait = min(30, 5 * (attempt + 1))
            logger.warning("TG polling failed (attempt %d): %s — retrying in %ds", attempt + 1, e, wait)
            try:
                await app_tg.updater.stop()
                await app_tg.stop()
                await app_tg.shutdown()
            except Exception:
                pass
            await asyncio.sleep(wait)
    else:
        logger.error("Failed to start TG polling after 10 attempts, running HTTP API only")

    # Keep running + health monitor
    # Warm the A-share trade-day calendar so the first hourly check has it populated
    _refresh_trade_days(force=True)
    last_health_alert = None
    try:
        while True:
            await asyncio.sleep(1)
            # Health check: 每小时 05 分检查上一根 K 线是否已落入 pa_kline
            now = datetime.now(timezone.utc)
            if now.minute == 5 and now.second < 2 and (
                last_health_alert is None or (now - last_health_alert).total_seconds() > 1800
            ):
                alert_items = []
                with db_engine.connect() as conn:
                    rows = conn.execute(text(
                        "SELECT "
                        "  CASE WHEN symbol LIKE '%/SH' OR symbol LIKE '%/SZ' "
                        "       THEN 'ashare' ELSE 'crypto' END AS market, "
                        "  timeframe, MAX(candle_time) as last_candle, "
                        "  COUNT(DISTINCT symbol) as pair_count "
                        "FROM pa_kline "
                        "GROUP BY market, timeframe"
                    )).fetchall()
                for r in rows:
                    market, tf, last_candle, pair_count = r[0], r[1], r[2], r[3]
                    if last_candle.tzinfo is None:
                        last_candle = last_candle.replace(tzinfo=timezone.utc)
                    age_hours = (now - last_candle).total_seconds() / 3600
                    threshold, note = _health_threshold(market, tf, now)
                    if age_hours > threshold:
                        tag = "A股" if market == "ashare" else "Crypto"
                        suffix = f", {note}" if note else ""
                        alert_items.append(
                            f"[{tag}] {tf} K线已 {age_hours:.1f}h 未更新 "
                            f"({pair_count}个标的{suffix}, 阈值 {threshold}h)"
                        )
                if alert_items:
                    last_health_alert = now
                    tz_shanghai = timezone(timedelta(hours=8))
                    local_str = now.astimezone(tz_shanghai).strftime("%H:%M")
                    detail = "\n".join(alert_items)
                    try:
                        await app_tg.bot.send_message(
                            chat_id=TG_CHAT_ID,
                            text=f"⚠ [{local_str}] 健康检查:\n{detail}\n请检查 freqtrade 容器状态",
                        )
                    except Exception:
                        logger.exception("Failed to send health alert")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            await app_tg.updater.stop()
            await app_tg.stop()
            await app_tg.shutdown()
        except Exception:
            pass
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
