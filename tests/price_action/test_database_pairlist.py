"""Tests for the PostgreSQL-backed price-action pairlist."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from freqtrade.plugins.pairlist.DatabasePairList import DatabasePairList


def _make_engine(rows):
    fake_conn = MagicMock()
    fake_conn.execute.return_value = list(rows)
    fake_ctx = MagicMock()
    fake_ctx.__enter__ = MagicMock(return_value=fake_conn)
    fake_ctx.__exit__ = MagicMock(return_value=False)

    fake_engine = MagicMock()
    fake_engine.connect.return_value = fake_ctx
    return fake_engine


def test_empty_watch_pair_returns_empty_without_market_validation():
    pairlist = DatabasePairList.__new__(DatabasePairList)
    pairlist._db_url = "postgresql://postgres:postgres@postgres:5432/freqtrade_priceaction"
    pairlist._config = {"exchange": {"name": "ashare"}}
    pairlist._whitelist_for_active_markets = MagicMock()

    fake_engine = _make_engine([])
    with patch(
        "freqtrade.plugins.pairlist.DatabasePairList.create_engine",
        return_value=fake_engine,
    ), patch(
        "freqtrade.plugins.pairlist.DatabasePairList.text",
        side_effect=lambda value: value,
    ):
        result = pairlist._load_from_db()

    assert result == []
    pairlist._whitelist_for_active_markets.assert_not_called()
    fake_engine.dispose.assert_called_once()
