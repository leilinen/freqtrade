"""
Unit tests for tg_bot market detection and symbol normalization.

Focus: /pa_add 自动识别市场规则 (纯数字→A股, 字母→美股, 含/或稳定币→加密),
以及 A 股代码按前缀补 /SH 或 /SZ 的归一化逻辑。

Run from repo root:
  .venv/bin/pytest tests/price_action/test_tg_bot_market.py -v
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# tg_bot.py 在模块顶层 import httpx/aiohttp/sqlalchemy/telegram。
# 这些是项目运行依赖，测试使用真实包，避免污染 sys.modules 影响全量 pytest。
# ---------------------------------------------------------------------------
pytest.importorskip("aiohttp")
pytest.importorskip("sqlalchemy")
pytest.importorskip("telegram")
pytest.importorskip("telegram.ext")

# tg_bot 在 import 时读 TG_TOKEN/TG_CHAT_ID 环境变量
os.environ.setdefault("TG_TOKEN", "x")
os.environ.setdefault("TG_CHAT_ID", "1")

# 为了 import tg_bot,把 services/ 加入 sys.path
# (tests/price_action/ -> ../../ = repo root -> services/)
_SERVICES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "services"))
if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)

import tg_bot  # noqa: E402


# ===================================================================
# Tests: /decision HTTP notification flow
# ===================================================================


class TestDecisionHttpFlow:
    """主流程：freqtrade notifier -> tg-bot /decision -> Telegram 文本/图片推送。"""

    def _decision_payload(self):
        return {
            "symbol": "BTC/USDT",
            "signal_time": "2026-06-01T10:00:00+00:00",
            "timeframe": "1h",
            "decision_type": "enter_long",
            "direction": "long",
            "order_type": "stop",
            "entry": 42000.0,
            "stop_loss": 41800.0,
            "take_profit_1": 42400.0,
            "risk_reward": 2.0,
            "confidence": 0.7,
            "reason": "EMA20 pullback reversal",
            "decision_trace": ["trend up", "pullback to EMA20"],
            "validation": {"valid": True},
        }

    @pytest.mark.asyncio
    async def test_decision_form_payload_without_chart_sends_text(self):
        """图表生成失败时 notifier 会发 form payload；tg-bot 应文本兜底推送。"""
        bot = SimpleNamespace(send_message=AsyncMock(), send_photo=AsyncMock())
        payload = self._decision_payload()

        class Request:
            content_type = "application/x-www-form-urlencoded"
            app = {"tg_bot": SimpleNamespace(bot=bot)}

            async def post(self):
                return {"payload": json.dumps(payload)}

        response = await tg_bot.handle_decision(Request())

        assert response.status == 200
        bot.send_message.assert_awaited_once()
        bot.send_photo.assert_not_awaited()
        text = bot.send_message.await_args.kwargs["text"]
        assert "PA决策 BTC/USDT 1h" in text
        assert "做多进场" in text

    @pytest.mark.asyncio
    async def test_decision_multipart_payload_with_chart_sends_photo(self):
        """正常带图路径应发 photo，并把格式化后的决策放在 caption。"""
        bot = SimpleNamespace(send_message=AsyncMock(), send_photo=AsyncMock())
        payload = self._decision_payload()

        class Part:
            def __init__(self, name, value):
                self.name = name
                self._value = value

            async def text(self):
                return self._value

            async def read(self):
                return self._value

        class Multipart:
            def __aiter__(self):
                self._parts = iter(
                    [
                        Part("payload", json.dumps(payload)),
                        Part("chart", bytearray(b"\x89PNG_fake")),
                    ]
                )
                return self

            async def __anext__(self):
                try:
                    return next(self._parts)
                except StopIteration:
                    raise StopAsyncIteration

        class Request:
            content_type = "multipart/form-data"
            app = {"tg_bot": SimpleNamespace(bot=bot)}

            async def multipart(self):
                return Multipart()

        response = await tg_bot.handle_decision(Request())

        assert response.status == 200
        bot.send_photo.assert_awaited_once()
        bot.send_message.assert_not_awaited()
        assert bot.send_photo.await_args.kwargs["photo"] == b"\x89PNG_fake"
        assert "做多进场" in bot.send_photo.await_args.kwargs["caption"]


# ===================================================================
# Tests: format_decision_message 精简渲染（理由摘要 + trace 紧凑一行）
# ===================================================================


class TestDecisionMessageCompact:
    """理由长文应截断为摘要；decision_trace 节点应渲染成紧凑一行而非 dict 字面量。"""

    def _long_reason(self) -> str:
        return (
            "K1 inside bear方向 body_pct=0.3046 close=63140.60低于K2 close=63299.40 "
            "overlap=0.7613高重叠 volume=312.79较K2大幅萎缩58.9%。"
            "K2突破63090.9的follow_through更新为no——突破后缺乏跟进。"
            "但K1 close=63140.60仍高于63090.9突破位50点 突破尚未完全失败 "
            "三重突破栈底层(62407.0)和中层(62964.6)仍有效。"
        )

    def _trace(self) -> list:
        return [
            {
                "node_id": "3.1",
                "question": "当前是否有有效的信号棒？",
                "answer": "中性",
                "reason": "K2 trend_bull body_pct=0.525 close=63299.40突破63090.9创新高63469.30 是上一根信号K线。",
                "branch": "weak",
                "section": "信号棒评估",
                "bar_range": "K2-K1",
            },
            {
                "node_id": "4.1",
                "question": "K2 close=63299.40突破63090.9是否仍为有效突破？",
                "answer": "中性",
                "reason": "顶层突破处于危险，中底层仍有效。",
                "branch": "breakout_danger",
                "section": "突破评估",
                "bar_range": "K7-K1",
            },
        ]

    def test_reason_is_truncated_with_ellipsis(self):
        msg = tg_bot.format_decision_message({
            "symbol": "BTC/USDT",
            "timeframe": "4h",
            "decision_type": "enter_long",
            "reason": self._long_reason(),
        })
        reason_line = next(ln for ln in msg.splitlines() if ln.startswith("理由:"))
        # 摘要应远短于原文，且以省略号结尾
        assert reason_line.endswith("…")
        assert len(reason_line) < len(self._long_reason())

    def test_trace_items_rendered_as_compact_line_not_dict(self):
        msg = tg_bot.format_decision_message({
            "symbol": "BTC/USDT",
            "timeframe": "4h",
            "decision_type": "enter_long",
            "decision_trace": self._trace(),
        })
        # 不应再出现 dict 字面量的痕迹
        assert "{'node_id'" not in msg
        assert "node_id" not in msg.split("•")[1] if "•" in msg else True
        # 第一条 trace 应紧凑呈现 node_id · section · answer
        first_trace = next(ln for ln in msg.splitlines() if ln.startswith("• 3.1"))
        assert "信号棒评估" in first_trace
        assert "中性" in first_trace

    def test_string_trace_item_still_handled(self):
        """历史纯字符串 trace 也应被截断处理，不报错。"""
        msg = tg_bot.format_decision_message({
            "symbol": "BTC/USDT",
            "timeframe": "4h",
            "decision_type": "enter_long",
            "decision_trace": ["trend up", "pullback to EMA20"],
        })
        assert "• trend up" in msg

    def test_format_trace_item_unit(self):
        item = {
            "node_id": "4.1",
            "section": "突破评估",
            "answer": "中性",
            "reason": "顶层突破处于危险，中底层仍有效。",
        }
        out = tg_bot._format_trace_item(item)
        assert out == "4.1 · 突破评估 · 中性 — 顶层突破处于危险，中底层仍有效。"

    def test_format_trace_item_truncates_long_reason(self):
        item = {
            "node_id": "3.1",
            "section": "信号棒评估",
            "answer": "中性",
            "reason": self._long_reason(),
        }
        out = tg_bot._format_trace_item(item)
        assert out.endswith("…")
        # 头部仍保留三段 + 分隔
        assert out.startswith("3.1 · 信号棒评估 · 中性 — ")


# ===================================================================
# Tests: /quote chart routing
# ===================================================================


class TestQuoteChartRoutes:
    """图表路由必须支持新旧 price-action 服务并行部署。"""

    def test_default_routes_keep_legacy_container_names(self):
        routes = tg_bot._load_chart_routes("")

        assert routes[("crypto", "1h")] == ("price-action-1h", 8091)
        assert routes[("crypto", "4h")] == ("price-action-4h", 8092)
        assert routes[("ashare", "1h")] == ("ashare-1h", 8093)
        assert routes[("ashare", "1d")] == ("ashare-1d", 8094)

    def test_env_routes_can_target_freqtrade_priceaction_stack(self):
        raw = json.dumps({
            "crypto:1h": {
                "host": "freqtrade_priceaction_crypto_1h",
                "port": 8091,
            },
            "ashare:1d": [
                "freqtrade_priceaction_ashare_1d",
                "8094",
            ],
        })

        routes = tg_bot._load_chart_routes(raw)

        assert routes[("crypto", "1h")] == (
            "freqtrade_priceaction_crypto_1h",
            8091,
        )
        assert routes[("ashare", "1d")] == (
            "freqtrade_priceaction_ashare_1d",
            8094,
        )


# ===================================================================
# Tests: Telegram command menu
# ===================================================================


class TestTelegramCommandMenu:
    """菜单只展示当前 PA 决策盯盘的主交互入口。"""

    def test_menu_excludes_legacy_signal_history_commands(self):
        commands = [command for command, _ in tg_bot.BOT_COMMAND_SPECS]

        assert commands == [
            "pa_watch",
            "pa_add",
            "pa_remove",
            "quote",
            "pa_status",
            "pa_help",
        ]
        assert "pa_signals" not in commands
        assert "pa_history" not in commands

    @pytest.mark.asyncio
    async def test_sync_bot_commands_clears_stale_scopes(self):
        bot = SimpleNamespace(
            delete_my_commands=AsyncMock(),
            set_my_commands=AsyncMock(),
        )

        await tg_bot.sync_bot_commands(bot)

        assert bot.delete_my_commands.await_count == 5
        deleted_scopes = [
            call.kwargs["scope"]
            for call in bot.delete_my_commands.await_args_list
        ]
        assert any(isinstance(scope, tg_bot.BotCommandScopeAllPrivateChats) for scope in deleted_scopes)
        assert any(isinstance(scope, tg_bot.BotCommandScopeChat) for scope in deleted_scopes)

        assert bot.set_my_commands.await_count == 2
        commands = bot.set_my_commands.await_args_list[0].args[0]
        assert [command.command for command in commands] == [
            "pa_watch",
            "pa_add",
            "pa_remove",
            "quote",
            "pa_status",
            "pa_help",
        ]


# ===================================================================
# Tests: _normalize_ashare_code
# ===================================================================

class TestNormalizeAshareCode:
    """A 股 6 位代码按首位数字补沪/深后缀。"""

    @pytest.mark.parametrize(
        "code,expected",
        [
            # 沪市主板 6xxxxx
            ("600519", "600519/SH"),
            ("601318", "601318/SH"),
            # 科创板 688xxx
            ("688981", "688981/SH"),
            # 沪市 ETF/基金 5xxxxx (含科创板 ETF 588xxx)
            ("510300", "510300/SH"),
            ("588290", "588290/SH"),
            ("513650", "513650/SH"),
            # 深市主板 0xxxxx
            ("000001", "000001/SZ"),
            ("000858", "000858/SZ"),
            # 创业板 30xxxx
            ("300750", "300750/SZ"),
            ("300059", "300059/SZ"),
            # 深市 ETF/基金 15xxxx/16xxxx
            ("159934", "159934/SZ"),
            ("160644", "160644/SZ"),
            # 北交所 8xxxxx / 4xxxxx
            ("830879", "830879/BJ"),
            ("430139", "430139/BJ"),
        ],
    )
    def test_valid_codes(self, code, expected):
        assert tg_bot._normalize_ashare_code(code) == expected

    @pytest.mark.parametrize(
        "code",
        [
            "",
            "123",          # 位数不对
            "1234567",      # 太长
            "ABCDEF",       # 非数字
            "58829O",       # 含字母 O
            "2XXXXX",       # 2 开头不合法(A 股无此前缀段)
            "712345",       # 7 开头(仅用于特定品种,不归 A 股 ETF/股票)
            "912345",       # 9 开头(B 股,不在本次自动判定范围)
        ],
    )
    def test_invalid_codes_return_none(self, code):
        assert tg_bot._normalize_ashare_code(code) is None


# ===================================================================
# Tests: _detect_market
# ===================================================================

class TestDetectMarket:
    """_detect_market 返回 (market, normalized_symbol)。"""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # 纯数字 → ashare,自动补后缀
            ("588290", ("ashare", "588290/SH")),
            ("510300", ("ashare", "510300/SH")),
            ("600519", ("ashare", "600519/SH")),
            ("000001", ("ashare", "000001/SZ")),
            ("159934", ("ashare", "159934/SZ")),
            # 小写也行,应当归一化大写
            ("588290", ("ashare", "588290/SH")),
            # 已带后缀 → ashare
            ("510300/SH", ("ashare", "510300/SH")),
            ("159934/SZ", ("ashare", "159934/SZ")),
            ("510300/sh", ("ashare", "510300/SH")),  # 小写后缀
            ("830879/BJ", ("ashare", "830879/BJ")),  # 北交所
            # 含 / 且 quote 是稳定币 → crypto
            ("BTC/USDT", ("crypto", "BTC/USDT")),
            ("BTC/USDC", ("crypto", "BTC/USDC")),
            # 以稳定币结尾无斜杠 → crypto
            ("BTCUSDT", ("crypto", "BTC/USDT")),
            ("BTCUSDC", ("crypto", "BTC/USDC")),
            # 纯字母 → usstock
            ("AAPL", ("usstock", "AAPL")),
            ("TSLA", ("usstock", "TSLA")),
            ("brk.b", ("usstock", "BRK.B")),    # 含点,伯克希尔 B 类
            ("brk-b", ("usstock", "BRK-B")),    # 含横线
        ],
    )
    def test_known_markets(self, raw, expected):
        assert tg_bot._detect_market(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "123",           # 3 位数字
            "1234567",       # 7 位数字
            "ABC123",        # 字母数字混合(不以字母开头,也不全是数字)
            "588290/SS",     # 非法 A 股后缀,quote 非稳定币
            "eth/btc",       # crypto 但 quote(BTC)不是稳定币 → 要求显式声明
        ],
    )
    def test_unknown_inputs(self, raw):
        market, _ = tg_bot._detect_market(raw)
        assert market == "unknown"


# ===================================================================
# Tests: _verify_* 函数 (用 patch 避免真实 HTTP)
# ===================================================================

class TestVerifyAshareSymbol:
    """_verify_ashare_symbol 委托给 _fetch_ashare_name,有名字即存在。"""

    def test_returns_true_when_name_found(self):
        with patch.object(tg_bot, "_fetch_ashare_name", return_value="贵州茅台"):
            assert tg_bot._verify_ashare_symbol("600519/SH") is True

    def test_returns_false_when_name_none(self):
        with patch.object(tg_bot, "_fetch_ashare_name", return_value=None):
            assert tg_bot._verify_ashare_symbol("999999/SH") is False


class TestVerifyCryptoSymbol:
    """_verify_crypto_symbol 调 OKX spot instruments,找到 instId 即视为存在。"""

    def test_returns_true_on_okx_200(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "code": "0",
            "data": [{"instId": "BTC-USDT", "state": "live"}],
        }
        fake_client = MagicMock()
        fake_client.get.return_value = fake_resp
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        with patch.object(tg_bot.httpx, "Client", return_value=fake_client):
            assert tg_bot._verify_crypto_symbol("BTC/USDT") is True
        fake_client.get.assert_called_once_with(
            "https://www.okx.com/api/v5/public/instruments",
            params={"instType": "SPOT", "instId": "BTC-USDT"},
        )

    def test_returns_false_when_okx_data_empty(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"code": "0", "data": []}
        fake_client = MagicMock()
        fake_client.get.return_value = fake_resp
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        with patch.object(tg_bot.httpx, "Client", return_value=fake_client):
            assert tg_bot._verify_crypto_symbol("FAKE/USDT") is False

    def test_returns_false_on_okx_http_error(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 400
        fake_resp.text = '{"code":"51000","msg":"Parameter instId error"}'
        fake_client = MagicMock()
        fake_client.get.return_value = fake_resp
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        with patch.object(tg_bot.httpx, "Client", return_value=fake_client):
            assert tg_bot._verify_crypto_symbol("FAKE/USDT") is False

    def test_returns_false_on_exception(self):
        with patch.object(tg_bot.httpx, "Client", side_effect=Exception("network")):
            assert tg_bot._verify_crypto_symbol("BTC/USDT") is False


class TestVerifyUsstockSymbol:
    """_verify_usstock_symbol 调 Yahoo chart 接口。"""

    def test_returns_true_on_yahoo_200(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"chart": {"result": [{"meta": {"symbol": "AAPL"}}]}}
        fake_client = MagicMock()
        fake_client.get.return_value = fake_resp
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        with patch.object(tg_bot.httpx, "Client", return_value=fake_client):
            assert tg_bot._verify_usstock_symbol("AAPL") is True

    def test_returns_false_on_yahoo_404(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 404
        fake_client = MagicMock()
        fake_client.get.return_value = fake_resp
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        with patch.object(tg_bot.httpx, "Client", return_value=fake_client):
            assert tg_bot._verify_usstock_symbol("FAKE") is False


class TestVerifySymbolDispatch:
    """_verify_symbol 按 market 分发到对应校验函数。"""

    def test_dispatches_ashare(self):
        with patch.object(tg_bot, "_verify_ashare_symbol", return_value=True) as m:
            assert tg_bot._verify_symbol("ashare", "600519/SH") is True
            m.assert_called_once_with("600519/SH")

    def test_dispatches_crypto(self):
        with patch.object(tg_bot, "_verify_crypto_symbol", return_value=True) as m:
            assert tg_bot._verify_symbol("crypto", "BTC/USDT") is True
            m.assert_called_once_with("BTC/USDT")

    def test_dispatches_usstock(self):
        with patch.object(tg_bot, "_verify_usstock_symbol", return_value=True) as m:
            assert tg_bot._verify_symbol("usstock", "AAPL") is True
            m.assert_called_once_with("AAPL")

    def test_unknown_market_returns_false(self):
        assert tg_bot._verify_symbol("unknown", "WHATEVER") is False


# ===================================================================
# Tests: A-share health-check threshold
# ===================================================================


class TestHealthThreshold:
    """A 股未开市时不应触发 K 线健康检查告警。"""

    def test_ashare_weekend_skips_check(self):
        # 2026-06-13 10:05 Beijing is Saturday.
        now = datetime(2026, 6, 13, 2, 5, tzinfo=timezone.utc)
        with patch.object(tg_bot, "_refresh_trade_days", return_value={"2026-06-12"}):
            threshold, note = tg_bot._health_threshold("ashare", "1h", now)
        assert threshold is None
        assert note == "周末"

    def test_ashare_weekday_holiday_skips_check(self):
        # Calendar knows this weekday is not a trading day.
        now = datetime(2026, 6, 9, 2, 5, tzinfo=timezone.utc)
        with patch.object(tg_bot, "_refresh_trade_days", return_value={"2026-06-08"}):
            threshold, note = tg_bot._health_threshold("ashare", "1h", now)
        assert threshold is None
        assert note == "休市日"

    def test_ashare_trade_day_before_open_skips_check(self):
        # 2026-06-08 09:05 Beijing: trade day, but no new intraday K-line is expected yet.
        now = datetime(2026, 6, 8, 1, 5, tzinfo=timezone.utc)
        with patch.object(tg_bot, "_refresh_trade_days", return_value={"2026-06-08"}):
            threshold, note = tg_bot._health_threshold("ashare", "1h", now)
        assert threshold is None
        assert note == "非交易时段"

    def test_ashare_trade_day_lunch_break_skips_check(self):
        # 2026-06-08 12:05 Beijing: lunch break.
        now = datetime(2026, 6, 8, 4, 5, tzinfo=timezone.utc)
        with patch.object(tg_bot, "_refresh_trade_days", return_value={"2026-06-08"}):
            threshold, note = tg_bot._health_threshold("ashare", "1h", now)
        assert threshold is None
        assert note == "非交易时段"

    def test_ashare_trading_session_uses_timeframe_threshold(self):
        # 2026-06-08 10:05 Beijing: actively trading.
        now = datetime(2026, 6, 8, 2, 5, tzinfo=timezone.utc)
        with patch.object(tg_bot, "_refresh_trade_days", return_value={"2026-06-08"}):
            assert tg_bot._health_threshold("ashare", "1h", now) == (5, "")
            assert tg_bot._health_threshold("ashare", "1d", now) == (36, "")


class TestKlineHealthRows:
    """Health check should only count enabled watch pairs."""

    def test_query_joins_enabled_watch_pairs(self):
        result = MagicMock()
        result.fetchall.return_value = [("crypto", "1h", datetime(2026, 6, 26, 21), 4)]
        conn = MagicMock()
        conn.execute.return_value = result

        with patch.object(tg_bot, "text", side_effect=lambda sql: sql):
            rows = tg_bot._fetch_kline_health_rows(conn)

        sql = conn.execute.call_args[0][0]
        assert "JOIN watch_pair wp ON wp.symbol = k.symbol" in sql
        assert "WHERE wp.enabled = true" in sql
        assert "COUNT(DISTINCT k.symbol)" in sql
        assert rows == result.fetchall.return_value


# ===================================================================
# Tests: K-line backfill after /pa_add
# ===================================================================


class TestKlineBackfill:
    """Backfill should fetch historical OHLCV and upsert into pa_kline."""

    def _http_client(self, response):
        fake_client = MagicMock()
        fake_client.get.return_value = response
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        return fake_client

    def test_timeframes_match_running_pa_services(self):
        assert tg_bot._backfill_timeframes_for_market("crypto") == ("1h", "4h")
        assert tg_bot._backfill_timeframes_for_market("ashare") == ("1h", "1d")
        assert tg_bot._backfill_timeframes_for_market("usstock") == ()

    def test_fetch_crypto_klines_okx_filters_open_candle_and_sorts(self):
        closed_ts = int(datetime(2026, 6, 30, 10, tzinfo=timezone.utc).timestamp() * 1000)
        older_ts = int(datetime(2026, 6, 30, 9, tzinfo=timezone.utc).timestamp() * 1000)
        future_ts = int((datetime.now(timezone.utc) + timedelta(days=10)).timestamp() * 1000)
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "code": "0",
            "data": [
                [str(future_ts), "9", "10", "8", "9.5", "99", "", "", "0"],
                [str(closed_ts), "2", "3", "1", "2.5", "20", "", "", "1"],
                [str(older_ts), "1", "2", "0.5", "1.5", "10", "", "", "1"],
            ],
        }
        fake_client = self._http_client(response)

        with patch.object(tg_bot.httpx, "Client", return_value=fake_client):
            rows = tg_bot._fetch_crypto_klines_okx("BTC/USDT", "1h", 120)

        fake_client.get.assert_called_once_with(
            tg_bot.OKX_CANDLES_URL,
            params={"instId": "BTC-USDT", "bar": "1H", "limit": "121"},
        )
        assert [r["candle_time"] for r in rows] == [
            datetime(2026, 6, 30, 9),
            datetime(2026, 6, 30, 10),
        ]
        assert rows[0]["open"] == 1.0
        assert rows[1]["volume"] == 20.0

    def test_fetch_ashare_daily_tencent_parses_beijing_date_as_utc_naive(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "data": {
                "sh588290": {
                    "qfqday": [
                        ["2026-06-29", "1.00", "1.10", "1.20", "0.90", "1000"],
                        ["2026-06-30", "1.10", "1.15", "1.30", "1.05", "1200"],
                    ]
                }
            }
        }
        fake_client = self._http_client(response)

        with patch.object(tg_bot.httpx, "Client", return_value=fake_client):
            rows = tg_bot._fetch_ashare_daily_tencent("588290/SH", 120)

        assert rows[0]["candle_time"] == datetime(2026, 6, 28, 16)
        assert rows[1]["candle_time"] == datetime(2026, 6, 29, 16)
        assert rows[1]["high"] == 1.30
        assert rows[1]["close"] == 1.15

    def test_fetch_ashare_1h_sina_parses_jsonp(self):
        response = MagicMock()
        response.status_code = 200
        response.text = (
            'callback([{"day":"2026-06-30 10:30:00","open":"1.00","high":"1.20",'
            '"low":"0.90","close":"1.10","volume":"1000"}])'
        )
        fake_client = self._http_client(response)

        with patch.object(tg_bot.httpx, "Client", return_value=fake_client):
            rows = tg_bot._fetch_ashare_1h_sina("588290/SH", 120)

        assert rows == [{
            "candle_time": datetime(2026, 6, 30, 2, 30),
            "open": 1.0,
            "high": 1.2,
            "low": 0.9,
            "close": 1.1,
            "volume": 1000.0,
        }]

    def test_upsert_pa_klines_uses_batch_conflict_update(self):
        rows = [{
            "candle_time": datetime(2026, 6, 30, 10),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 10.0,
        }]
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        engine = MagicMock()
        engine.begin.return_value = conn

        with patch.object(tg_bot, "db_engine", engine):
            count = tg_bot._upsert_pa_klines("BTC/USDT", "1h", rows)

        assert count == 1
        sql, params = conn.execute.call_args[0]
        assert "INSERT INTO pa_kline" in str(sql)
        assert "ON CONFLICT (symbol, timeframe, candle_time) DO UPDATE" in str(sql)
        assert params == [{
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "candle_time": datetime(2026, 6, 30, 10),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 10.0,
        }]

    def test_backfill_continues_when_one_timeframe_fails(self):
        rows = [{
            "candle_time": datetime(2026, 6, 30, 10),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 10.0,
        }]
        with (
            patch.object(
                tg_bot,
                "_fetch_crypto_klines_okx",
                side_effect=[rows, RuntimeError("network")],
            ),
            patch.object(tg_bot, "_upsert_pa_klines", return_value=1) as upsert,
        ):
            results = tg_bot._backfill_pair_klines("BTC/USDT", "crypto")

        assert results[0] == {"timeframe": "1h", "ok": True, "rows": 1}
        assert results[1]["timeframe"] == "4h"
        assert results[1]["ok"] is False
        assert "network" in results[1]["error"]
        upsert.assert_called_once_with("BTC/USDT", "1h", rows)

    def test_format_backfill_summary_for_success_failure_and_unsupported_market(self):
        assert tg_bot._format_backfill_summary("usstock", []) == "回填: usstock 暂无 PA 回填周期"
        summary = tg_bot._format_backfill_summary(
            "crypto",
            [
                {"timeframe": "1h", "ok": True, "rows": 120},
                {"timeframe": "4h", "ok": False, "error": "timeout"},
            ],
        )
        assert summary == "回填: 1h=120, 4h失败(timeout)"


# ===================================================================
# Tests: db_get_signals_by_symbol
# ===================================================================


class TestDbGetSignalsBySymbol:
    """Verify db_get_signals_by_symbol passes correct SQL and params."""

    def _mock_engine(self, rows):
        """Create a mock db_engine whose .connect() returns a context manager.
        `rows` is the list of tuples that fetchall() should return."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        return mock_engine, mock_conn

    def test_passes_symbol_and_limit_params(self):
        rows = [("BTC/USDT", "1h", "long", "good", 0.85,
                 MagicMock(tzinfo=None), "strong_bar", 65000.0)]
        mock_engine, mock_conn = self._mock_engine(rows)
        with patch.object(tg_bot, "db_engine", mock_engine):
            result = tg_bot.db_get_signals_by_symbol("BTC/USDT", 5)
        # Verify execute was called with 2 positional args (text() + params dict)
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        # Verify params dict contains correct symbol and limit
        params = call_args[0][1]
        assert params["s"] == "BTC/USDT"
        assert params["limit"] == 5

    def test_clamps_limit_to_30(self):
        rows = [("BTC/USDT", "1h", "long", "good", 0.85,
                 MagicMock(tzinfo=None), "strong_bar", 65000.0)]
        mock_engine, mock_conn = self._mock_engine(rows)
        with patch.object(tg_bot, "db_engine", mock_engine):
            tg_bot.db_get_signals_by_symbol("BTC/USDT", 100)
        params = mock_conn.execute.call_args[0][1]
        assert params["limit"] == 30

    def test_returns_empty_when_no_rows(self):
        mock_engine, _ = self._mock_engine([])
        with patch.object(tg_bot, "db_engine", mock_engine):
            result = tg_bot.db_get_signals_by_symbol("NONEXIST", 10)
        assert result == []

    def test_returns_dicts_from_rows(self):
        candle_time = MagicMock()
        candle_time.tzinfo = None
        rows = [
            ("588290/SH", "1h", "long", "good", 0.85, candle_time, "strong_bar", 1.234),
            ("588290/SH", "1h", "short", "fair", 0.42, candle_time, "doji", 1.100),
        ]
        mock_engine, _ = self._mock_engine(rows)
        with patch.object(tg_bot, "db_engine", mock_engine):
            result = tg_bot.db_get_signals_by_symbol("588290/SH", 10)
        assert len(result) == 2
        assert result[0]["symbol"] == "588290/SH"
        assert result[0]["direction"] == "long"
        assert result[0]["quality"] == "good"
        assert result[0]["entry_price"] == 1.234
        assert result[1]["direction"] == "short"
        assert result[1]["entry_price"] == 1.100


# ===================================================================
# Tests: format_signal_message (EMA20 cross branch)
# ===================================================================


class TestFormatSignalMessageCross:
    """Verify EMA20 cross signals use the full format (same as signal bar)."""

    def test_cross_up_uses_full_format(self):
        """上穿EMA20 → 方向文本'上穿EMA20做多' + 完整指标行。"""
        data = {
            "symbol": "BTC/USDT",
            "display_name": None,
            "timeframe": "1h",
            "direction": "long",
            "quality": "cross",
            "signal_time": "2026-06-20T10:00:00+00:00",
            "entry_price": 60000.0,
            "body_pct": 0.8,
            "close_location": 0.9,
            "body_ratio": 1.5,
            "stop_loss": 58000.0,
            "target_price": 64000.0,
        }
        msg = tg_bot.format_signal_message(data)
        assert "上穿EMA20做多" in msg
        assert "[Cross]" in msg
        assert "当前价格: 60000.00" in msg
        assert "实体占比=0.80" in msg
        assert "止损 58000.00" in msg

    def test_cross_down_uses_full_format(self):
        """下穿EMA20 → 方向文本'下穿EMA20做空' + 完整指标行。"""
        data = {
            "symbol": "ETH/USDT",
            "display_name": None,
            "timeframe": "4h",
            "direction": "short",
            "quality": "cross",
            "signal_time": "2026-06-20T10:00:00+00:00",
            "entry_price": 3000.0,
            "body_pct": 0.6,
            "close_location": 0.7,
            "body_ratio": 1.2,
            "stop_loss": 3100.0,
            "target_price": 2800.0,
        }
        msg = tg_bot.format_signal_message(data)
        assert "下穿EMA20做空" in msg
        assert "[Cross]" in msg
        assert "实体占比=0.60" in msg

    def test_cross_message_includes_time(self):
        """穿越消息应该带北京时间。"""
        data = {
            "symbol": "588290/SH",
            "display_name": "科创ETF",
            "timeframe": "1d",
            "direction": "long",
            "quality": "cross",
            "signal_time": "2026-06-20T02:00:00+00:00",  # 北京 10:00
            "entry_price": 1.05,
            "body_pct": 0.5,
            "close_location": 0.6,
            "body_ratio": 1.0,
        }
        msg = tg_bot.format_signal_message(data)
        assert "@06-20 10:00" in msg
        # 带 display_name 时应展开成 中文(symbol)
        assert "科创ETF(588290/SH)" in msg

    def test_normal_signal_message_unchanged(self):
        """quality=good 仍走原完整路径,不应触发精简分支。"""
        data = {
            "symbol": "BTC/USDT",
            "display_name": None,
            "timeframe": "1h",
            "direction": "long",
            "quality": "good",
            "signal_time": "2026-06-20T10:00:00+00:00",
            "entry_price": 60000.0,
            "body_pct": 0.8,
            "close_location": 0.9,
            "body_ratio": 1.5,
            "stop_loss": 58000.0,
            "target_price": 64000.0,
        }
        msg = tg_bot.format_signal_message(data)
        assert "做多" in msg
        assert "实体占比" in msg
        assert "止损" in msg

    def test_new_context_bar_type_labels_are_rendered(self):
        """新增结构标签应该映射为中文展示。"""
        data = {
            "symbol": "BTC/USDT",
            "display_name": None,
            "timeframe": "1h",
            "direction": "long",
            "quality": "good",
            "entry_price": 60000.0,
            "body_pct": 0.8,
            "close_location": 0.9,
            "body_ratio": 1.5,
            "bar_types": ["ioi", "mdb", "breakout_up", "range_edge"],
        }
        msg = tg_bot.format_signal_message(data)
        assert "内外内" in msg
        assert "微双底" in msg
        assert "上破近5K" in msg
        assert "区间边界" in msg


# ===================================================================
# Tests: fmt_price
# ===================================================================


class TestFmtPrice:
    """Verify price formatting precision by magnitude."""

    def test_large_price(self):
        assert tg_bot.fmt_price(65000.0) == "65000.00"

    def test_medium_price(self):
        assert tg_bot.fmt_price(3000.0) == "3000.00"

    def test_small_price(self):
        assert tg_bot.fmt_price(1.2345) == "1.2345"

    def test_tiny_price(self):
        assert tg_bot.fmt_price(0.000123) == "0.000123"
