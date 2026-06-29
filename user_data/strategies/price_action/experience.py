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
    return repository.query_experience(
        market=market,
        timeframe=timeframe,
        cycle_position=str(diagnosis.get("cycle_position", "")) or None,
        direction=str(diagnosis.get("direction", "")) or None,
        patterns=[str(p) for p in diagnosis.get("detected_patterns", []) if p],
        limit=limit,
    )
