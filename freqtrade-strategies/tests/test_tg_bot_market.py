"""
Unit tests for tg_bot market detection and symbol normalization.

Focus: /pa_add 自动识别市场规则 (纯数字→A股, 字母→美股, 含/或稳定币→加密),
以及 A 股代码按前缀补 /SH 或 /SZ 的归一化逻辑。

Run from repo root:
  .venv/bin/pytest freqtrade-strategies/tests/test_tg_bot_market.py -v
"""
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# tg_bot.py 在模块顶层 import httpx/aiohttp/sqlalchemy/telegram,测试时必须 stub。
# 跟同目录 test_price_action_monitor.py 的风格一致(sys.modules.setdefault)。
# ---------------------------------------------------------------------------
for name in ("aiohttp", "sqlalchemy", "telegram", "telegram.ext"):
    if name not in sys.modules:
        sys.modules[name] = MagicMock()

# aiohttp.web 在模块内被用作类型注解,给个 SimpleNamespace 避免属性错误
sys.modules["aiohttp"].web = types.SimpleNamespace(
    Request=object, Response=object, Application=object, run_app=lambda *a, **kw: None
)

# tg_bot 在 import 时读 TG_TOKEN/TG_CHAT_ID 环境变量
os.environ.setdefault("TG_TOKEN", "x")
os.environ.setdefault("TG_CHAT_ID", "1")

# 为了 import tg_bot,把 freqtrade-strategies/ 加入 sys.path
_STRAT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _STRAT_DIR not in sys.path:
    sys.path.insert(0, _STRAT_DIR)

import tg_bot  # noqa: E402


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
            ("BTCUSDT", ("crypto", "BTCUSDT")),
            ("BTCUSDC", ("crypto", "BTCUSDC")),
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
    """_verify_crypto_symbol 调 Binance ticker/price,200+symbol 字段即视为存在。"""

    def test_returns_true_on_binance_200(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"symbol": "BTCUSDT", "price": "60000.0"}
        fake_client = MagicMock()
        fake_client.get.return_value = fake_resp
        fake_client.__enter__ = MagicMock(return_value=fake_client)
        fake_client.__exit__ = MagicMock(return_value=False)
        with patch.object(tg_bot.httpx, "Client", return_value=fake_client):
            assert tg_bot._verify_crypto_symbol("BTC/USDT") is True

    def test_returns_false_on_binance_400(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 400
        fake_resp.text = '{"code":-1121,"msg":"Invalid symbol"}'
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
                 MagicMock(tzinfo=None), "strong_bar")]
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
                 MagicMock(tzinfo=None), "strong_bar")]
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
            ("588290/SH", "1h", "long", "good", 0.85, candle_time, "strong_bar"),
            ("588290/SH", "1h", "short", "fair", 0.42, candle_time, "doji"),
        ]
        mock_engine, _ = self._mock_engine(rows)
        with patch.object(tg_bot, "db_engine", mock_engine):
            result = tg_bot.db_get_signals_by_symbol("588290/SH", 10)
        assert len(result) == 2
        assert result[0]["symbol"] == "588290/SH"
        assert result[0]["direction"] == "long"
        assert result[0]["quality"] == "good"
        assert result[1]["direction"] == "short"


# ===================================================================
# Tests: format_signal_message (EMA20 cross branch)
# ===================================================================


class TestFormatSignalMessageCross:
    """Verify the concise rendering path for ema20_cross signals."""

    def test_cross_up_message(self):
        """上穿 → ↗ 符号 + EMA20 上穿 + 价格。"""
        data = {
            "symbol": "BTC/USDT",
            "display_name": None,
            "timeframe": "1h",
            "direction": "long",
            "quality": "cross",  # 触发精简分支
            "signal_time": "2026-06-20T10:00:00+00:00",
            "entry_price": 60000.0,
        }
        msg = tg_bot.format_signal_message(data)
        assert "↗" in msg
        assert "EMA20 上穿" in msg
        assert "BTC/USDT" in msg
        assert "当前价格" in msg
        # 精简分支不应出现实体占比/收盘位置等形态细节
        assert "实体占比" not in msg
        assert "收盘位置" not in msg

    def test_cross_down_message(self):
        """下穿 → ↘ 符号 + EMA20 下穿 + 价格。"""
        data = {
            "symbol": "ETH/USDT",
            "display_name": None,
            "timeframe": "4h",
            "direction": "short",
            "quality": "cross",
            "signal_time": "2026-06-20T10:00:00+00:00",
            "entry_price": 3000.0,
        }
        msg = tg_bot.format_signal_message(data)
        assert "↘" in msg
        assert "EMA20 下穿" in msg
        assert "ETH/USDT" in msg
        assert "实体占比" not in msg

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
