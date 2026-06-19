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
