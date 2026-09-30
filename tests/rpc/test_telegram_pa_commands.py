"""PA watch telegram commands — /list /add /remove /enable /disable /signal.

Run against sqlite-backed stores (schema is dialect-neutral); handlers are
driven through the authorized_only wrapper with a stubbed _send_msg.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from freqtrade.rpc.telegram import (
    Telegram,
    _bot_command_menu,
    _format_signals,
    _format_watch_pairs,
    _split_market_arg,
    _validate_pair_timeframe,
)
from freqtrade.persistence import init_db
from pa_core.records.pg_store import PgRecordStore
from tests.pa_core.unit.test_pg_store import _make_record

CHAT_ID = 258711369


def _update() -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(chat_id=CHAT_ID, message_thread_id=None),
        callback_query=None,
        effective_user=None,
    )


def _ctx(args: list[str]) -> SimpleNamespace:
    return SimpleNamespace(args=args)


@pytest.fixture()
def tg(default_conf, tmp_path):
    """Telegram instance with pa stores on sqlite and a capturing send stub."""
    default_conf["telegram"] = {"enabled": True, "token": "t", "chat_id": CHAT_ID}
    default_conf["timeframe"] = "1h"
    default_conf["pa_db_url"] = f"sqlite:///{tmp_path}/pa_records.db"
    init_db(default_conf["db_url"])  # authorized_only touches Trade.session

    instance = Telegram.__new__(Telegram)
    instance._config = default_conf
    sent: list[str] = []

    async def _capture(msg: str, **kwargs: Any) -> None:
        sent.append(msg)

    instance._send_msg = _capture
    instance._sent = sent
    return instance


def _run(coro):
    asyncio.run(coro)


def test_split_market_arg():
    assert _split_market_arg("SOL/USDT") == ("SOL/USDT", "crypto")
    assert _split_market_arg(" ashare:600519 ") == ("600519", "ashare")


def test_format_watch_pairs():
    assert "为空" in _format_watch_pairs([])
    rows = [
        {"symbol": "BTC/USDT", "enabled": True, "display_name": "Bitcoin",
         "market": "crypto", "timeframe": None},
        {"symbol": "SOL/USDT", "enabled": False, "display_name": "SOL/USDT",
         "market": "crypto", "timeframe": "4h"},
    ]
    text = _format_watch_pairs(rows, "1h")
    assert "2 行" in text and "默认 1h" in text
    assert "✅ BTC/USDT · 1h · Bitcoin" in text
    sol_line = next(line for line in text.splitlines() if "SOL/USDT" in line)
    assert "⏸ SOL/USDT · 4h" in sol_line and "· SOL/USDT" not in sol_line


def test_validate_pair_timeframe():
    _validate_pair_timeframe("1h", {"timeframe": "1h"})  # equal is allowed
    _validate_pair_timeframe("1d", {"timeframe": "1h"})
    _validate_pair_timeframe("1d", {})  # missing config -> 1h default
    with pytest.raises(ValueError, match="低于主周期"):
        _validate_pair_timeframe("5m", {"timeframe": "1h"})
    with pytest.raises(ValueError, match="低于主周期"):
        _validate_pair_timeframe("15m", {"timeframe": "1h"})
    with pytest.raises(ValueError, match="无效周期"):
        _validate_pair_timeframe("4x", {"timeframe": "1h"})


def test_format_signals():
    assert "signal 表为空" in _format_signals([])
    rows = [
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "created_at": "2026-09-30T11:15:27.700000",
            "order_direction": "做多",
            "order_type": "限价单",
            "entry_price": 83384.68,
            "stop_loss_price": 82000.0,
            "take_profit_price": 85000.0,
            "trade_confidence": 65,
        }
    ]
    text = _format_signals(rows)
    assert "BTC/USDT 1h 2026-09-30 11:15" in text
    assert "做多 限价单 @ 83384.7" in text
    assert "SL 82000" in text and "TP 85000" in text
    assert "置信 65" in text


def test_watch_add_list_flow(tg):
    _run(tg._watch_add(_update(), _ctx(["SOL/USDT", "Solana"])))
    assert any("已添加 SOL/USDT" in m for m in tg._sent)

    _run(tg._watch_list(_update(), _ctx([])))
    listing = next(m for m in tg._sent if "Watch pairs" in m)
    assert "✅ SOL/USDT · 1h · Solana" in listing  # no tf -> default shown
    assert "默认 1h" in listing


def test_watch_add_with_timeframe(tg):
    _run(tg._watch_add(_update(), _ctx(["DOGE/USDT", "4h", "Doge"])))
    assert any("已添加 DOGE/USDT · 周期 4h" in m for m in tg._sent)
    rows = {r["symbol"]: r for r in tg._pa_watch_store().list_pairs()}
    assert rows["DOGE/USDT"]["timeframe"] == "4h"
    assert rows["DOGE/USDT"]["display_name"] == "Doge"


def test_watch_add_rejects_timeframe_below_main(tg):
    _run(tg._watch_add(_update(), _ctx(["DOGE/USDT", "5m"])))
    assert any("❌" in m and "5m" in m for m in tg._sent)
    assert tg._pa_watch_store().list_pairs() == []  # nothing inserted


def test_watch_add_duplicate_reports_error(tg):
    _run(tg._watch_add(_update(), _ctx(["BTC/USDT"])))
    _run(tg._watch_add(_update(), _ctx(["BTC/USDT"])))
    assert any("❌" in m and "BTC/USDT" in m for m in tg._sent)


def test_watch_disable_enable_remove(tg):
    up, ctx = _update(), _ctx(["ETH/USDT"])
    _run(tg._watch_add(up, _ctx(["ETH/USDT"])))  # fresh sqlite: no seeded rows
    store = tg._pa_watch_store()

    _run(tg._watch_disable(up, ctx))
    assert any("已停用 ETH/USDT" in m for m in tg._sent)
    assert [r["symbol"] for r in store.list_pairs(enabled_only=True)] == []

    _run(tg._watch_enable(up, ctx))
    assert any("已启用 ETH/USDT" in m for m in tg._sent)
    assert [r["symbol"] for r in store.list_pairs(enabled_only=True)] == ["ETH/USDT"]

    _run(tg._watch_remove(up, ctx))
    assert any("已删除 ETH/USDT" in m for m in tg._sent)
    assert "ETH/USDT" not in [r["symbol"] for r in store.list_pairs()]


def test_watch_remove_unknown_reports_error(tg):
    _run(tg._watch_remove(_update(), _ctx(["DOGE/USDT"])))
    assert any("❌" in m for m in tg._sent)


def test_watch_usage_hints_without_args(tg):
    for handler in (tg._watch_add, tg._watch_remove, tg._watch_disable, tg._watch_enable):
        _run(handler(_update(), _ctx([])))
    assert len(tg._sent) == 4
    assert all("用法" in m for m in tg._sent)


def test_watch_signal_reads_ledger(tg):
    store = PgRecordStore(tg._config["pa_db_url"])
    record = _make_record(
        decision={
            "order_type": "限价单",
            "order_direction": "做多",
            "entry_price": 83384.68,
            "stop_loss_price": 82000.0,
            "take_profit_price": 85000.0,
            "take_profit_price_2": 88000.0,
            "estimated_win_rate": 55,
        }
    )
    assert store.save_full(record)
    _run(tg._watch_signal(_update(), _ctx([])))
    assert any("BTC/USDT" in m and "做多 限价单" in m for m in tg._sent)


def test_watch_signal_empty_ledger(tg):
    _run(tg._watch_signal(_update(), _ctx(["3"])))
    assert any("signal 表为空" in m for m in tg._sent)


def test_watch_store_unavailable_reply(mocker, default_conf):
    default_conf["telegram"] = {"enabled": True, "token": "t", "chat_id": CHAT_ID}
    default_conf.pop("pa_db_url", None)  # no storage configured
    init_db(default_conf["db_url"])
    mocker.patch.object(Telegram, "_pa_watch_store", return_value=None)

    instance = Telegram.__new__(Telegram)
    instance._config = default_conf
    sent: list[str] = []

    async def _capture(msg: str, **kwargs: Any) -> None:
        sent.append(msg)

    instance._send_msg = _capture
    _run(instance._watch_list(_update(), _ctx([])))
    assert any("不可用" in m for m in sent)


def test_bot_command_menu():
    """'/' menu: PA watch commands present, no stale pa_* entries, curated."""
    menu = _bot_command_menu()
    names = [c.command for c in menu]
    assert {"list", "add", "remove", "enable", "disable", "signal"} <= set(names)
    assert not any(n.startswith("pa_") for n in names)
    assert all(c.description for c in menu)
    assert len(menu) <= 20
