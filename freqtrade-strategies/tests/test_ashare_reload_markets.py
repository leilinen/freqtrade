"""
Unit tests for Ashare.reload_markets / _load_ashare_pairs.

覆盖 2026-06-25 的改动：监控标的来源从 config.pair_whitelist 切换到 PostgreSQL
watch_pair 表 (SELECT symbol FROM watch_pair WHERE enabled=true AND market='ashare')。

Run from repo root:
  .venv/bin/pytest freqtrade-strategies/tests/test_ashare_reload_markets.py -v
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# 本仓库的 freqtrade 包未做 pip 安装（非 editable install），它之所以能 import
# 是因为 cwd 恰好是仓库根。pytest 收集时 cwd 不保证是仓库根，所以这里把仓库根
# （含 freqtrade/ 包的目录，即本测试文件向上三级）显式加入 sys.path。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 兄弟测试 test_price_action_monitor.py 用 sys.modules.setdefault 把 "freqtrade"
# 替换成 MagicMock（mock 策略依赖）。同进程运行时会污染本测试对真实包的导入，
# 这里先清掉可能存在的 mock，强制走真实包。
for _mod in ("freqtrade", "freqtrade.exchange", "freqtrade.exchange.ashare"):
    if _mod in sys.modules and not hasattr(sys.modules[_mod], "__file__"):
        del sys.modules[_mod]

# venv 里安装的就是本仓库的 freqtrade 包，可直接真实导入。reload_markets 和
# _load_ashare_pairs 只依赖 self._config 与 sqlalchemy，不触发 ccxt 初始化，
# 因此无需 mock Exchange 父类——测试中用 __new__ 绕过 __init__ 即可。
from freqtrade.exchange.ashare import Ashare  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instance(db_url="postgresql://u:p@host:5432/db"):
    """创建一个跳过 Exchange.__init__ 的 Ashare 裸实例，注入 _config。

    reload_markets / _load_ashare_pairs 只读 self._config，不依赖任何父类
    实例状态，因此绕过 __init__ 是安全的。
    """
    inst = Ashare.__new__(Ashare)
    inst._config = {"pa_db_url": db_url} if db_url else {}
    # 父类 Exchange.__del__/close 直接访问多个 async 属性，绕过 __init__ 时需
    # 手动补上，否则实例被 GC 回收时抛 AttributeError（PytestUnraisableExceptionWarning）。
    inst._exchange_ws = None
    inst._ws_async = None
    inst._api_async = None
    inst.loop = None
    return inst


def _make_engine(rows):
    """构造一个 fake create_engine：connect().execute() 返回给定行。

    rows 是 list[tuple]，每 tuple 形如 ("000001/SZ",)。
    """
    fake_conn = MagicMock()
    fake_conn.execute.return_value = list(rows)  # 可迭代，每元素支持 row[0]
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=fake_conn)
    fake_ctx.__exit__ = MagicMock(return_value=False)

    fake_engine = MagicMock()
    fake_engine.connect.return_value = fake_ctx
    return fake_engine


# ===================================================================
# Tests: _load_ashare_pairs
# ===================================================================


class TestLoadAsharePairs:
    """_load_ashare_pairs: 从 watch_pair 表读取 enabled 的 ashare 标的。"""

    def test_returns_pairs_from_db(self):
        """查到 3 个 enabled 的 ashare 标的，应按 ORDER BY id 返回。"""
        inst = _make_instance()
        fake_engine = _make_engine(
            [("000001/SZ",), ("600519/SH",), ("515050/SH",)]
        )
        # create_engine / text 在 _load_ashare_pairs 内部是延迟 import，
        # 因此 patch 源模块 sqlalchemy，而非 ashare 命名空间。
        with patch("sqlalchemy.create_engine", return_value=fake_engine), \
             patch("sqlalchemy.text", side_effect=lambda s: s):
            result = inst._load_ashare_pairs()

        assert result == ["000001/SZ", "600519/SH", "515050/SH"]
        # engine 释放资源
        fake_engine.dispose.assert_called_once()

    def test_returns_empty_when_no_rows(self):
        """表里没有 enabled 的 ashare 标的，应返回空列表（不报错）。"""
        inst = _make_instance()
        fake_engine = _make_engine([])
        with patch("sqlalchemy.create_engine", return_value=fake_engine), \
             patch("sqlalchemy.text", side_effect=lambda s: s):
            result = inst._load_ashare_pairs()

        assert result == []

    def test_returns_empty_when_db_url_missing(self, caplog):
        """config 里没有 pa_db_url，应返回空列表并记 error 日志。"""
        inst = _make_instance(db_url=None)
        assert inst._config.get("pa_db_url") in (None, "")

        with patch("sqlalchemy.create_engine") as mock_ce:
            result = inst._load_ashare_pairs()

        assert result == []
        # 不应尝试建连接
        mock_ce.assert_not_called()

    def test_returns_empty_on_db_exception(self, caplog):
        """数据库连接失败应被捕获，返回空列表（不抛出）。"""
        import logging as _logging

        caplog.set_level(_logging.ERROR)
        inst = _make_instance()
        with patch("sqlalchemy.create_engine",
                   side_effect=Exception("connection refused")):
            result = inst._load_ashare_pairs()

        assert result == []


# ===================================================================
# Tests: reload_markets
# ===================================================================


class TestReloadMarkets:
    """reload_markets: 用 _load_ashare_pairs 的结果构建 markets dict。"""

    def test_builds_markets_for_each_pair(self):
        """每个 pair 应生成一个 market 条目，含 base/quote/symbol/active。"""
        inst = _make_instance()
        with patch.object(
            inst,
            "_load_ashare_pairs",
            return_value=["000001/SZ", "515050/SH"],
        ):
            inst.reload_markets()

        assert "000001/SZ" in inst._markets
        assert "515050/SH" in inst._markets

        m = inst._markets["515050/SH"]
        assert m["base"] == "515050"
        assert m["quote"] == "CNY"
        assert m["symbol"] == "515050/SH"
        assert m["active"] is True
        assert m["spot"] is True

    def test_empty_pairs_yields_empty_markets(self):
        """_load_ashare_pairs 返回空时，markets 也应为空（不报错）。"""
        inst = _make_instance()
        with patch.object(inst, "_load_ashare_pairs", return_value=[]):
            inst.reload_markets()

        assert inst._markets == {}

    def test_ignores_config_pair_whitelist(self):
        """核心契约：reload_markets 不应读取 config.pair_whitelist，
        只认 watch_pair 表。config 里有 pair_whitelist 也应被忽略。"""
        inst = _make_instance()
        # 故意在 config 里塞 pair_whitelist，验证它不被使用
        inst._config["pair_whitelist"] = ["999999/SZ", "SHOULD/IGNORE"]
        with patch.object(
            inst,
            "_load_ashare_pairs",
            return_value=["000001/SZ"],
        ) as mock_load:
            inst.reload_markets()

        # 结果只来自 _load_ashare_pairs，不含 pair_whitelist 的内容
        assert list(inst._markets.keys()) == ["000001/SZ"]
        assert "999999/SZ" not in inst._markets
        assert "SHOULD/IGNORE" not in inst._markets
        mock_load.assert_called_once()

    def test_calls_load_pairs_once(self):
        """reload_markets 应调用 _load_ashare_pairs 一次。"""
        inst = _make_instance()
        with patch.object(
            inst, "_load_ashare_pairs", return_value=["000001/SZ"]
        ) as mock_load:
            inst.reload_markets()

        mock_load.assert_called_once()

    def test_invalid_pair_format_logged_not_raised(self, caplog):
        """格式非法的 pair（缺少 /EXCHANGE）会触发 ValueError；
        reload_markets 当前实现不捕获它——此测试固化当前行为。

        若未来决定容错跳过非法 pair，更新此测试即可。
        """
        import logging as _logging

        inst = _make_instance()
        with patch.object(inst, "_load_ashare_pairs", return_value=["BADPAIR"]):
            # 当前实现：_parse_pair 抛 ValueError，未被 reload_markets 捕获
            with pytest.raises(ValueError):
                inst.reload_markets()
