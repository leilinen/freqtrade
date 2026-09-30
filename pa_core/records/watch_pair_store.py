"""PostgreSQL persistence for the ``watch_pair`` table (monitored symbols).

The watch system's monitored-symbol table — read by freqtrade's
DatabasePairList and the ashare exchange loader (raw SQL, strictly read-only),
written by ops tooling (``tools/watch_pairs.py``) and later the telegram
plugin's ``/add`` / ``/remove`` handlers via this store.

Error philosophy is the opposite of PgRecordStore: this is the storage layer
of an *interactive* tool, so failures raise. Callers (CLI now, telegram
handlers later) decide how to present or swallow them.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    String,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from pa_core.records.pg_store import PaBase

MARKET_CRYPTO = "crypto"
MARKET_ASHARE = "ashare"

# Cold-start defaults: seeded into an empty market at bot startup. The
# whitelist lives entirely in PG — watch_config.json carries no pair_whitelist.
DEFAULT_WATCH_PAIRS = ("BTC/USDT", "ETH/USDT")


class WatchPairRow(PaBase):
    """One monitored symbol — freqtrade reads enabled rows filtered by market."""

    __tablename__ = "watch_pair"
    __table_args__ = (
        UniqueConstraint("symbol", "market", name="uq_watch_pair_symbol_market"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Exchange pair symbol, e.g. "BTC/USDT" (crypto) or "600519" (ashare).
    symbol: Mapped[str] = mapped_column(String(32))
    market: Mapped[str] = mapped_column(String(16), default=MARKET_CRYPTO, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Human-facing label for telegram cards; defaults to the symbol itself.
    display_name: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class WatchPairDuplicateError(ValueError):
    """(symbol, market) already exists in watch_pair."""


class WatchPairNotFoundError(KeyError):
    """No row for (symbol, market) in watch_pair."""


def _row_to_dict(row: WatchPairRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "symbol": row.symbol,
        "market": row.market,
        "enabled": bool(row.enabled),
        "display_name": row.display_name,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


class WatchPairStore:
    """CRUD access to watch_pair. Raises on failure (see module docstring).

    Thread model matches PgRecordStore: one engine, short-lived sessions per
    call — safe to share between CLI runs, the strategy process and the
    telegram plugin.
    """

    def __init__(self, db_url: str, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._engine = create_engine(db_url, future=True)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        # Idempotent: creates watch_pair (plus any missing PaBase tables).
        PaBase.metadata.create_all(self._engine)

    # ── Reads ──────────────────────────────────────────────────────────────────

    def list_pairs(
        self, market: str | None = None, enabled_only: bool = False
    ) -> list[dict[str, Any]]:
        """Rows ordered by id (insertion order); optionally filtered."""
        with self._session_factory() as session:
            stmt = select(WatchPairRow).order_by(WatchPairRow.id)
            if market:
                stmt = stmt.where(WatchPairRow.market == market)
            if enabled_only:
                stmt = stmt.where(WatchPairRow.enabled.is_(True))
            return [_row_to_dict(row) for row in session.execute(stmt).scalars().all()]

    # ── Writes ─────────────────────────────────────────────────────────────────

    def add(
        self,
        symbol: str,
        market: str = MARKET_CRYPTO,
        display_name: str | None = None,
    ) -> int:
        """Insert one monitored symbol; returns the row id."""
        symbol = symbol.strip()
        if not symbol:
            raise ValueError("symbol must not be empty")
        with self._session_factory() as session, session.begin():
            stmt = select(WatchPairRow).where(
                WatchPairRow.symbol == symbol, WatchPairRow.market == market
            )
            if session.execute(stmt).scalar_one_or_none() is not None:
                raise WatchPairDuplicateError(
                    f"{symbol} already exists in market {market}"
                )
            row = WatchPairRow(
                symbol=symbol,
                market=market,
                enabled=True,
                display_name=display_name or symbol,
            )
            session.add(row)
            session.flush()
            return row.id

    def remove(self, symbol: str, market: str = MARKET_CRYPTO) -> None:
        """Hard-delete the row. To stop watching a symbol, prefer
        set_enabled(False) — deleting every row of a market leaves the
        whitelist empty until rows are added back."""
        with self._session_factory() as session, session.begin():
            row = self._get(session, symbol, market)
            session.delete(row)

    def set_enabled(
        self, symbol: str, market: str = MARKET_CRYPTO, enabled: bool = True
    ) -> None:
        with self._session_factory() as session, session.begin():
            row = self._get(session, symbol, market)
            row.enabled = enabled

    def set_display_name(
        self, symbol: str, market: str = MARKET_CRYPTO, display_name: str | None = None
    ) -> None:
        with self._session_factory() as session, session.begin():
            row = self._get(session, symbol, market)
            row.display_name = display_name

    def seed(
        self, symbols: list[str], market: str = MARKET_CRYPTO
    ) -> list[str]:
        """Insert the symbols that don't exist yet; returns the added ones.

        Idempotent — used by ``watch_pairs.py init``.
        """
        added: list[str] = []
        with self._session_factory() as session, session.begin():
            existing = set(
                session.execute(
                    select(WatchPairRow.symbol).where(WatchPairRow.market == market)
                ).scalars().all()
            )
            for raw in symbols:
                symbol = str(raw).strip()
                if not symbol or symbol in existing:
                    continue
                session.add(
                    WatchPairRow(
                        symbol=symbol, market=market, enabled=True, display_name=symbol
                    )
                )
                existing.add(symbol)
                added.append(symbol)
        if added:
            self._logger.info("Seeded %d watch_pair rows: %s", len(added), added)
        return added

    def ensure_defaults(
        self, market: str = MARKET_CRYPTO, symbols: list[str] | None = None
    ) -> list[str]:
        """Cold-start seeding: insert *symbols* (default DEFAULT_WATCH_PAIRS)
        only when the market has no rows at all; returns the added symbols.

        Contract: deleting every row of a market re-seeds the defaults on the
        next bot start; to stop watching a single symbol use set_enabled(False).
        """
        syms = [s for s in (symbols if symbols is not None else DEFAULT_WATCH_PAIRS) if s.strip()]
        with self._session_factory() as session, session.begin():
            count = session.execute(
                select(func.count())
                .select_from(WatchPairRow)
                .where(WatchPairRow.market == market)
            ).scalar_one()
            if count:
                return []
            session.add_all(
                WatchPairRow(
                    symbol=s.strip(), market=market, enabled=True, display_name=s.strip()
                )
                for s in syms
            )
        if syms:
            self._logger.info(
                "Cold start: seeded %d default watch_pair rows: %s", len(syms), syms
            )
        return syms

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get(self, session, symbol: str, market: str) -> WatchPairRow:
        stmt = select(WatchPairRow).where(
            WatchPairRow.symbol == symbol.strip(), WatchPairRow.market == market
        )
        row = session.execute(stmt).scalar_one_or_none()
        if row is None:
            raise WatchPairNotFoundError(f"{symbol} not found in market {market}")
        return row
