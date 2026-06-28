"""Experience retrieval helpers."""
from __future__ import annotations

from typing import Any

from .repository import PriceActionRepository


def retrieve_experience_cases(
    repository: PriceActionRepository | None,
    *,
    market: str,
    timeframe: str,
    diagnosis: dict[str, Any],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Fetch simple cycle/direction/pattern matched cases from PostgreSQL."""
    if repository is None:
        return []
    market_state = diagnosis.get("market_state") or {}
    signal_chain = diagnosis.get("signal_chain") or {}
    return repository.query_experience(
        market=market,
        timeframe=timeframe,
        cycle_position=str(market_state.get("cycle", "")) or None,
        direction=str(signal_chain.get("direction") or market_state.get("direction") or "") or None,
        patterns=[str(p) for p in signal_chain.get("patterns", []) if p],
        limit=limit,
    )
