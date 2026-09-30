"""DatabasePairList tests — reads watch_pair via sqlite (dialect-neutral SQL)."""
from __future__ import annotations

import pytest

from freqtrade.plugins.pairlist.DatabasePairList import DatabasePairList
from freqtrade.plugins.pairlistmanager import PairListManager
from pa_core.records.watch_pair_store import WatchPairStore
from tests.conftest import get_patched_exchange, log_has, log_has_re


def _markets() -> dict:
    def mkt(base: str) -> dict:
        return {
            "id": f"{base.lower()}usdt",
            "symbol": f"{base}/USDT",
            "base": base,
            "quote": "USDT",
            "active": True,
            "spot": True,
            "swap": False,
            "linear": None,
            "type": "spot",
        }

    return {"BTC/USDT": mkt("BTC"), "ETH/USDT": mkt("ETH")}


@pytest.fixture()
def dpl_env(mocker, default_conf, tmp_path):
    default_conf["stake_currency"] = "USDT"
    default_conf["exchange"]["pair_whitelist"] = []
    default_conf["exchange"]["pair_blacklist"] = []
    db_url = f"sqlite:///{tmp_path}/watch_pair.db"
    store = WatchPairStore(db_url)
    default_conf["pairlists"] = [
        {"method": "DatabasePairList", "db_url": db_url, "refresh_period": 1800}
    ]
    exchange = get_patched_exchange(mocker, default_conf, mock_markets=_markets())
    return {
        "store": store,
        "config": default_conf,
        "exchange": exchange,
        "db_url": db_url,
        "tmp_path": tmp_path,
    }


def _make_pairlist(env) -> DatabasePairList:
    pairlistmanager = PairListManager(env["exchange"], env["config"])
    return DatabasePairList(
        env["exchange"], pairlistmanager, env["config"], env["config"]["pairlists"][0], 0
    )


def test_gen_pairlist_reads_enabled_rows_in_id_order(dpl_env, caplog):
    store: WatchPairStore = dpl_env["store"]
    store.add("BTC/USDT")
    store.add("ETH/USDT")
    store.add("SOL/USDT")
    store.set_enabled("SOL/USDT", enabled=False)
    store.add("DOGE/USDT")  # enabled but not an active market on the exchange

    pairlist = _make_pairlist(dpl_env)
    assert pairlist.gen_pairlist([]) == ["BTC/USDT", "ETH/USDT"]
    assert log_has_re("Pair DOGE/USDT is not compatible with exchange", caplog)


def test_ttl_cache_blocks_reload_until_expiry(dpl_env):
    store: WatchPairStore = dpl_env["store"]
    store.add("BTC/USDT")

    pairlist = _make_pairlist(dpl_env)
    assert pairlist.gen_pairlist([]) == ["BTC/USDT"]

    store.add("ETH/USDT")
    # within refresh_period and pairlist non-empty → cached result
    assert pairlist.gen_pairlist([]) == ["BTC/USDT"]

    pairlist._last_refresh = 0  # simulate TTL expiry
    assert pairlist.gen_pairlist([]) == ["BTC/USDT", "ETH/USDT"]


def test_empty_pairlist_retries_every_call(dpl_env):
    store: WatchPairStore = dpl_env["store"]
    pairlist = _make_pairlist(dpl_env)
    assert pairlist.gen_pairlist([]) == []

    store.add("BTC/USDT")
    # empty list bypasses the TTL — the row is visible on the next call
    assert pairlist.gen_pairlist([]) == ["BTC/USDT"]


def test_missing_table_returns_empty_and_logs(dpl_env, caplog):
    dpl_env["config"]["pairlists"] = [
        {
            "method": "DatabasePairList",
            # fresh file without watch_pair — engine stays read-only, never creates
            "db_url": f"sqlite:///{dpl_env['tmp_path']}/no_table.db",
            "refresh_period": 1800,
        }
    ]
    pairlist = _make_pairlist(dpl_env)
    assert pairlist.gen_pairlist([]) == []
    assert log_has("Failed to load pairs from database", caplog)


def test_db_url_falls_back_to_pa_db_url(mocker, default_conf, tmp_path):
    default_conf["stake_currency"] = "USDT"
    default_conf["exchange"]["pair_whitelist"] = []
    default_conf["exchange"]["pair_blacklist"] = []
    default_conf["pa_db_url"] = f"sqlite:///{tmp_path}/fallback.db"
    default_conf["pairlists"] = [{"method": "DatabasePairList", "refresh_period": 1800}]

    store = WatchPairStore(default_conf["pa_db_url"])
    store.add("BTC/USDT")

    exchange = get_patched_exchange(mocker, default_conf, mock_markets=_markets())
    pairlistmanager = PairListManager(exchange, default_conf)
    pairlist = DatabasePairList(
        exchange, pairlistmanager, default_conf, default_conf["pairlists"][0], 0
    )
    assert pairlist.gen_pairlist([]) == ["BTC/USDT"]
