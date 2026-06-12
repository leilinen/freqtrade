"""
Database Pair List provider

Reads pair whitelist from PostgreSQL watch_pair table.
Supports TTL-based caching to avoid querying the database on every bot iteration.
"""

import logging
import time

from sqlalchemy import create_engine, text

from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting


logger = logging.getLogger(__name__)


class DatabasePairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._db_url = self._pairlistconfig.get("db_url") or self._config.get("pa_db_url")
        if not self._db_url:
            raise ValueError("DatabasePairList requires 'db_url' in pairlist config or 'pa_db_url' in root config")

        self._refresh_period = self._pairlistconfig.get("refresh_period", 3600)
        self._pairlist: list[str] = []
        self._last_refresh: float = 0

    def short_desc(self) -> str:
        return f"DatabasePairList ({len(self._pairlist)} pairs)"

    @staticmethod
    def description() -> str:
        return "Reads pair whitelist from a PostgreSQL database table."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "db_url": {
                "type": "string",
                "default": "",
                "description": "PostgreSQL connection URL",
                "help": "SQLAlchemy connection string for the database containing watch_pair table.",
            },
            "refresh_period": {
                "type": "number",
                "default": 3600,
                "description": "Cache TTL in seconds",
                "help": "How often to re-query the database. Default: 3600 (1 hour).",
            },
        }

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        Generate the pairlist from database with TTL caching.
        """
        now = time.time()
        if now - self._last_refresh > self._refresh_period or not self._pairlist:
            self._pairlist = self._load_from_db()
            self._last_refresh = now
            logger.info("DatabasePairList refreshed: %d pairs", len(self._pairlist))

        return self._pairlist

    def _load_from_db(self) -> list[str]:
        """Query watch_pair table for enabled symbols, filtered by market."""
        exchange_name = self._config.get("exchange", {}).get("name", "")
        market = "ashare" if exchange_name == "ashare" else "crypto"

        try:
            engine = create_engine(self._db_url)
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT symbol FROM watch_pair "
                        "WHERE enabled = true AND market = :market ORDER BY id"
                    ),
                    {"market": market},
                )
                pairs = [row[0] for row in result]
            engine.dispose()
        except Exception:
            logger.exception("Failed to load pairs from database")
            return []

        return self._whitelist_for_active_markets(pairs)

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        return pairlist
