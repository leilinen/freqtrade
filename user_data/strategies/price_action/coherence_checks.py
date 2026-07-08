"""Cross-field coherence checks for Stage 1 AI JSON.

This is a freqtrade adapter for the Stage1 portions of
``PA_Agent/pa_agent/ai/coherence_checks.py``.  The upstream code consumes a
``KlineFrame`` and recomputes geometry features; this service already passes
``feature_rows`` containing the program-computed ``bar_type`` values, so the
adapter keeps the same behavior while reading from those rows directly.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

BAR_BY_BAR_TARGET_COUNT = 5

_BAR_FIELD_RE = re.compile(r"K\s*(\d+)", re.IGNORECASE)


def _feature_bar_types(feature_rows: list[dict[str, Any]] | None) -> dict[int, str]:
    features: dict[int, str] = {}
    for row in feature_rows or []:
        match = _BAR_FIELD_RE.search(str(row.get("k", "") or ""))
        if not match:
            continue
        bar_type = str(row.get("bar_type", "") or "").strip().lower()
        if bar_type:
            features[int(match.group(1))] = bar_type
    return features


def _feature_count(feature_rows: list[dict[str, Any]] | None) -> int | None:
    seqs = set(_feature_bar_types(feature_rows))
    return max(seqs) if seqs else None


def auto_fix_bar_by_bar_types(
    stage1: dict[str, Any],
    *,
    feature_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Auto-correct bar_type when it contradicts program features.

    Ported from upstream ``auto_fix_bar_by_bar_types``.  Mutates ``stage1``
    in-place: replaces genuine bull/bear contradictions with the program value.
    The freqtrade adapter also syncs ``bar_analysis.bar_type`` for K1 because
    the program feature table is authoritative here.
    """
    features = _feature_bar_types(feature_rows)
    if not features:
        return []
    corrections: list[str] = []
    _opposites = frozenset(
        {
            ("trend_bull", "trend_bear"),
            ("trend_bear", "trend_bull"),
            ("outside_bull", "outside_bear"),
            ("outside_bear", "outside_bull"),
        }
    )

    summary = stage1.get("bar_by_bar_summary")
    if isinstance(summary, list):
        for item in summary:
            if not isinstance(item, dict):
                continue
            match = _BAR_FIELD_RE.search(str(item.get("bar", "") or ""))
            if not match:
                continue
            seq = int(match.group(1))
            computed = features.get(seq)
            declared = str(item.get("bar_type", "") or "").strip().lower()
            if declared and computed and (declared, computed) in _opposites:
                item["bar_type"] = computed
                corrections.append(
                    "auto-fixed bar_by_bar_summary "
                    f"K{seq}.bar_type: {declared!r} -> {computed!r}"
                )

    bar_analysis = stage1.get("bar_analysis")
    if isinstance(bar_analysis, dict):
        computed = features.get(1)
        declared = str(bar_analysis.get("bar_type", "") or "").strip().lower()
        if declared and computed and declared != computed:
            bar_analysis["bar_type"] = computed
            corrections.append(
                f"auto-fixed bar_analysis.bar_type: {declared!r} -> {computed!r}"
            )
    return corrections


def validate_bar_by_bar_vs_features(
    stage1: dict[str, Any],
    *,
    feature_rows: list[dict[str, Any]] | None = None,
    strict: bool = False,
) -> list[str]:
    """Validate ``bar_by_bar_summary`` against program geometry features.

    Ported from upstream ``validate_bar_by_bar_vs_features``.  In lenient mode
    it only rejects genuine directional contradictions.  With ``strict=True``
    it rejects every mismatch, matching the old freqtrade behavior.
    """
    features = _feature_bar_types(feature_rows)
    if not features:
        return []
    summary = stage1.get("bar_by_bar_summary")
    if not isinstance(summary, list):
        return []

    structural_types = frozenset({"inside", "outside_bull", "outside_bear"})
    threshold_sensitive_types = frozenset({"doji", "trend_bull", "trend_bear", "other"})
    compatible_pairs = frozenset(
        {
            ("outside_bull", "trend_bull"),
            ("trend_bull", "outside_bull"),
            ("outside_bear", "trend_bear"),
            ("trend_bear", "outside_bear"),
            ("inside", "doji"),
            ("doji", "inside"),
        }
    )

    errors: list[str] = []
    for i, item in enumerate(summary):
        if not isinstance(item, dict):
            continue
        match = _BAR_FIELD_RE.search(str(item.get("bar", "") or ""))
        if not match:
            continue
        seq = int(match.group(1))
        computed = features.get(seq)
        if computed is None:
            errors.append(
                f"bar_by_bar_summary[{i}].bar K{seq} not in geometry feature table"
            )
            continue
        declared = str(item.get("bar_type", "") or "").strip().lower()
        if not declared or not computed or declared == computed:
            continue
        if strict:
            errors.append(
                f"bar_by_bar_summary[{i}].bar_type={declared!r} contradicts "
                f"program feature K{seq} bar_type={computed!r}"
            )
            continue
        if (declared, computed) in compatible_pairs:
            continue
        all_bar_types = structural_types | threshold_sensitive_types
        if declared in all_bar_types and computed in all_bar_types:
            opposites = (
                ("trend_bull", "trend_bear"),
                ("outside_bull", "outside_bear"),
            )
            if (declared, computed) in opposites or (computed, declared) in opposites:
                errors.append(
                    f"bar_by_bar_summary[{i}].bar_type={declared!r} contradicts "
                    f"program feature K{seq} bar_type={computed!r}"
                )
    return errors


def validate_bar_by_bar_count(
    stage1: dict[str, Any],
    *,
    feature_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return upstream-style count errors for the five-bar Stage1 summary."""
    summary = stage1.get("bar_by_bar_summary")
    if not isinstance(summary, list):
        return []
    n_bars = _feature_count(feature_rows)
    if n_bars is None:
        return []
    expected = min(BAR_BY_BAR_TARGET_COUNT, n_bars)
    if len(summary) != expected:
        return [
            f"bar_by_bar_summary has {len(summary)} items; "
            f"expected exactly {expected} for {n_bars} bars"
        ]
    return []


def normalize_bar_range(item: dict[str, Any]) -> str:
    """Return canonical bar_range text for duplicate-range checks."""
    return str(item.get("bar_range", "") or "").strip().upper().replace(" ", "")


def validate_duplicate_bar_ranges(
    trace: list[dict[str, Any]] | None,
    *,
    path_prefix: str,
    min_items: int = 4,
) -> list[str]:
    """Flag when too many gate/decision nodes share the same bar_range.

    Ported from upstream.  The main validator keeps this behind strict mode;
    PA_Agent defaults do not hard-fail Stage1 on this check.
    """
    ranges: list[str] = []
    for item in trace or []:
        if not isinstance(item, dict):
            continue
        if item.get("skipped") and item.get("answer") == "不适用":
            continue
        bar_range = normalize_bar_range(item)
        if bar_range and bar_range not in ("不适用", "—", "全局", "GLOBAL"):
            ranges.append(bar_range)
    if len(ranges) < min_items:
        return []
    if len(set(ranges)) == 1:
        return [
            f"{path_prefix}: {len(ranges)} nodes share identical bar_range "
            f"{ranges[0]!r}; each node should cite the K-lines it actually used"
        ]
    return []
