"""Tests for the watch_pair store (run on SQLite; schema is dialect-neutral)."""
from __future__ import annotations

import pytest

from pa_core.records.pg_store import PaBase
from pa_core.records.watch_pair_store import (
    MARKET_ASHARE,
    MARKET_CRYPTO,
    WatchPairDuplicateError,
    WatchPairNotFoundError,
    WatchPairStore,
)


@pytest.fixture()
def store(tmp_path):
    return WatchPairStore(f"sqlite:///{tmp_path}/watch_pair_test.db")


def test_init_creates_watch_pair_table(store: WatchPairStore):
    assert "watch_pair" in PaBase.metadata.tables
    assert store.list_pairs() == []


def test_add_and_list_roundtrip(store: WatchPairStore):
    store.add("BTC/USDT", display_name="Bitcoin")
    store.add("ETH/USDT")
    rows = store.list_pairs(market=MARKET_CRYPTO)
    assert [r["symbol"] for r in rows] == ["BTC/USDT", "ETH/USDT"]  # insertion order
    assert rows[0]["enabled"] is True
    assert rows[0]["display_name"] == "Bitcoin"
    assert rows[1]["display_name"] == "ETH/USDT"  # defaults to symbol
    assert rows[0]["created_at"]  # server_default populated


def test_add_duplicate_raises(store: WatchPairStore):
    store.add("BTC/USDT")
    with pytest.raises(WatchPairDuplicateError, match="BTC/USDT"):
        store.add("BTC/USDT")


def test_market_isolation_allows_same_symbol(store: WatchPairStore):
    store.add("600519", market=MARKET_ASHARE)
    store.add("600519", market=MARKET_CRYPTO)  # unique key is (symbol, market)
    assert len(store.list_pairs()) == 2
    assert len(store.list_pairs(market=MARKET_ASHARE)) == 1


def test_disable_filters_enabled_only_listing(store: WatchPairStore):
    store.add("BTC/USDT")
    store.add("ETH/USDT")
    store.set_enabled("BTC/USDT", enabled=False)
    enabled = store.list_pairs(enabled_only=True)
    assert [r["symbol"] for r in enabled] == ["ETH/USDT"]
    all_rows = store.list_pairs()
    assert [r["symbol"] for r in all_rows] == ["BTC/USDT", "ETH/USDT"]
    assert all_rows[0]["enabled"] is False
    store.set_enabled("BTC/USDT", enabled=True)
    assert len(store.list_pairs(enabled_only=True)) == 2


def test_remove_hard_deletes(store: WatchPairStore):
    store.add("BTC/USDT")
    store.remove("BTC/USDT")
    assert store.list_pairs() == []
    with pytest.raises(WatchPairNotFoundError):
        store.remove("BTC/USDT")


def test_missing_symbol_raises_not_found(store: WatchPairStore):
    with pytest.raises(WatchPairNotFoundError):
        store.set_enabled("BTC/USDT", enabled=False)
    with pytest.raises(WatchPairNotFoundError):
        store.set_display_name("BTC/USDT", display_name="x")


def test_set_display_name(store: WatchPairStore):
    store.add("BTC/USDT")
    store.set_display_name("BTC/USDT", display_name="Bitcoin")
    assert store.list_pairs()[0]["display_name"] == "Bitcoin"


def test_seed_idempotent(store: WatchPairStore):
    added = store.seed(["BTC/USDT", "ETH/USDT", " BTC/USDT "])
    assert added == ["BTC/USDT", "ETH/USDT"]  # stripped duplicate skipped
    assert len(store.list_pairs()) == 2
    assert store.seed(["BTC/USDT", "ETH/USDT"]) == []
    assert len(store.list_pairs()) == 2
    # seed never touches other markets
    assert store.list_pairs(market=MARKET_ASHARE) == []


def test_ensure_defaults_seeds_when_empty(store: WatchPairStore):
    assert store.ensure_defaults() == ["BTC/USDT", "ETH/USDT"]
    rows = store.list_pairs()
    assert [r["symbol"] for r in rows] == ["BTC/USDT", "ETH/USDT"]
    assert all(r["enabled"] for r in rows)


def test_ensure_defaults_noop_when_rows_exist(store: WatchPairStore):
    store.add("SOL/USDT")
    assert store.ensure_defaults() == []
    # existing rows are never extended with the defaults
    assert [r["symbol"] for r in store.list_pairs()] == ["SOL/USDT"]


def test_ensure_defaults_market_isolated(store: WatchPairStore):
    assert store.ensure_defaults(market=MARKET_ASHARE, symbols=["600519"]) == ["600519"]
    assert store.list_pairs(market=MARKET_CRYPTO) == []


def test_add_rejects_empty_symbol(store: WatchPairStore):
    with pytest.raises(ValueError, match="empty"):
        store.add("   ")


def test_watch_pair_metadata_separate_from_freqtrade():
    # watch_pair joins the pa_core metadata, not freqtrade's.
    pytest.importorskip("humanize", reason="freqtrade deps not installed")
    from freqtrade.persistence.base import ModelBase

    assert "watch_pair" in set(PaBase.metadata.tables)
    assert "watch_pair" not in set(ModelBase.metadata.tables)


def test_add_and_list_timeframe(store: WatchPairStore):
    store.add("BTC/USDT")  # NULL -> strategy default
    store.add("SOL/USDT", timeframe="4h")
    rows = store.list_pairs()
    assert rows[0]["timeframe"] is None
    assert rows[1]["timeframe"] == "4h"


def test_timeframe_migration_from_old_schema(tmp_path):
    """A table created before the timeframe column gets it added in place."""
    from sqlalchemy import create_engine, text

    db_url = f"sqlite:///{tmp_path}/old_schema.db"
    engine = create_engine(db_url, future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE watch_pair ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "symbol VARCHAR(32) NOT NULL, "
                "market VARCHAR(16) NOT NULL DEFAULT 'crypto', "
                "enabled BOOLEAN NOT NULL DEFAULT 1, "
                "display_name VARCHAR(64), "
                "created_at DATETIME, "
                "updated_at DATETIME"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO watch_pair (symbol, market, enabled, display_name) "
                "VALUES ('BTC/USDT', 'crypto', 1, 'BTC/USDT')"
            )
        )
    engine.dispose()

    store = WatchPairStore(db_url)  # __init__ runs the migration
    rows = store.list_pairs()
    assert rows[0]["symbol"] == "BTC/USDT"
    assert rows[0]["timeframe"] is None  # existing rows keep the default
    store.add("SOL/USDT", timeframe="1d")
    assert store.list_pairs()[1]["timeframe"] == "1d"
