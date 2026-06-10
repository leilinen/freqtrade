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
        rows = conn.execute(text("SELECT symbol, enabled, market FROM watch_pair ORDER BY id")).fetchall()
    return [{"symbol": r[0], "enabled": r[1], "market": r[2]} for r in rows]


def db_add_pair(symbol: str) -> str:
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
        conn.execute(
            text("INSERT INTO watch_pair (symbol, enabled, market) VALUES (:s, true, 'crypto')"),
            {"s": symbol},
        )
    return f"{symbol} added"


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
    """HTTP POST /signal — 接收 freqtrade 推送的信号并转发到 TG。"""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    msg = format_signal_message(data)
    try:
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
        f"{arrow} {symbol} {tf}",
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
        lines.append(f"  {p['symbol']} [{status}]")
    await update.message.reply_text("\n".join(lines))


async def pa_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    if not context.args:
        await update.message.reply_text("用法: /pa_add BTC/USDT")
        return
    symbol = context.args[0].upper()
    msg = db_add_pair(symbol)
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
    lines = [f"最近 {len(signals)} 条信号:"]
    for s in signals:
        ct = s["candle_time"]
        time_str = ct.strftime("%m/%d %H:%M") if ct else "?"
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
        "/pa_add <标的> — 添加标的 (例: /pa_add DOGE/USDT)\n"
        "/pa_remove <标的> — 禁用标的\n"
        "/pa_signals [N] — 最近信号 (默认10条)\n"
        "/pa_help — 帮助信息"
    )


# ================================================================
# Main — start TG polling + HTTP API
# ================================================================

async def main() -> None:
    app_tg = Application.builder().token(TG_TOKEN).build()

    app_tg.add_handler(CommandHandler("pa_watch", pa_watch))
    app_tg.add_handler(CommandHandler("pa_add", pa_add))
    app_tg.add_handler(CommandHandler("pa_remove", pa_remove))
    app_tg.add_handler(CommandHandler("pa_signals", pa_signals))
    app_tg.add_handler(CommandHandler("pa_help", pa_help))

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
            logger.info("TG bot polling started successfully")
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

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
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
