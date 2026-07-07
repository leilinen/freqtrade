"""Pattern-routing coherence layer (ported from PA_Agent pattern_routing).

The freqtrade :mod:`router` already ports the routing logic (which strategy
files to load). This module ports the *normalize* and *validate* halves
that keep ``detected_patterns`` in sync with the narrative text:

* :func:`ensure_detected_patterns_coherent` (called from stage1 normalize)
  merges model tags + entry_setup_type + cycle_position + key_signals
  keyword heuristics into ``detected_patterns`` so downstream routing
  sees a complete tag set.
* :func:`validate_detected_patterns_vs_key_signals` (called from stage1
  validate) emits warnings when narrative mentions a pattern keyword
  that's missing from ``detected_patterns``.

Barbwire auto-tagging depends on upstream's ``compute_simple_market_features``
which is not ported here; we read ``program_features.barbwire_candidate``
when present and otherwise treat barbwire as not-auto-detected.
"""
from __future__ import annotations

import re
from typing import Any

from .router import (
    CYCLE_PATTERN_TAGS,
    ENTRY_SETUP_PATTERN_OVERLAY,
    HL_COUNT_RE,
    PATTERN_KEYWORD_TAGS,
    merge_detected_patterns,
)


_HL_WORD_RE = re.compile(r"(?i)(?:high|low)\s*([12])(?![0-9])")


def _entry_setup_type(stage1: dict[str, Any]) -> str:
    bar = stage1.get("bar_analysis")
    if not isinstance(bar, dict):
        return ""
    return str(bar.get("entry_setup_type") or "").strip().lower()


def _barbwire_blocked(stage1: dict[str, Any]) -> bool:
    """True when program_features explicitly forbids barbwire auto-tagging."""
    pf = stage1.get("program_features")
    return isinstance(pf, dict) and pf.get("barbwire_candidate") is False


def _barbwire_candidate(stage1: dict[str, Any]) -> bool:
    """Return True when program_features indicates barbwire structure present.

    Limited to the ``program_features.barbwire_candidate`` flag since the
    upstream fallback to ``compute_simple_market_features`` requires a
    kline_frame + market_features module not ported to freqtrade.
    """
    if _barbwire_blocked(stage1):
        return False
    pf = stage1.get("program_features")
    if isinstance(pf, dict) and "barbwire_candidate" in pf:
        return bool(pf["barbwire_candidate"])
    return False


def _barbwire_keyword_negated(text: str) -> bool:
    """Skip keyword sync when narrative discusses sub-threshold barbwire score."""
    lowered = text.lower()
    if "未达阈值" in lowered and "铁丝网" in lowered:
        return True
    if re.search(r"铁丝网分数\s*0\.\d+", lowered) and "未达" in lowered:
        return True
    return False


def _should_sync_barbwire_tag(stage1: dict[str, Any], text: str) -> bool:
    if _barbwire_blocked(stage1):
        return False
    if _barbwire_keyword_negated(text):
        return False
    return _barbwire_candidate(stage1)


def _mentions_hl_count_setup(text: str) -> bool:
    return bool(HL_COUNT_RE.search(text))


def _hl_tags_from_text(text: str) -> list[str]:
    tags: list[str] = []
    for m in _HL_WORD_RE.finditer(text):
        prefix = "h" if m.group(0).lower().startswith("h") else "l"
        tags.append(f"{prefix}{m.group(1)}")
    for hl in ("h1", "h2", "l1", "l2"):
        if re.search(rf"(?<![a-z]){hl}(?![a-z])", text) and hl not in tags:
            tags.append(hl)
    return tags


def sync_detected_patterns_field(stage1: dict[str, Any]) -> list[str]:
    """Write merged tags back into ``stage1['detected_patterns']`` in place.

    Thin wrapper around :func:`router.merge_detected_patterns` (already
    ported) plus the barbwire gate from ENTRY_SETUP_TYPE_PATTERN_OVERLAY.
    """
    patterns = merge_detected_patterns(stage1)
    # router.merge_detected_patterns does not apply the barbwire gate;
    # we filter it here to mirror upstream behavior. Barbwire is only
    # kept when program_features.barbwire_candidate confirms it (absent
    # → drop, since compute_simple_market_features is not ported).
    if "barbwire" in patterns and not _barbwire_candidate(stage1):
        patterns = [p for p in patterns if p != "barbwire"]
    stage1["detected_patterns"] = patterns
    return patterns


def ensure_detected_patterns_coherent(stage1: dict[str, Any]) -> bool:
    """Auto-add detected_patterns tags implied by key_signals / entry_setup_type.

    Mirrors upstream ``ensure_detected_patterns_coherent``. Runs the
    initial sync, then walks keyword tables again to catch patterns
    whose keywords appear in narrative text but were not in the initial
    merge.
    """
    before = list(stage1.get("detected_patterns") or [])
    sync_detected_patterns_field(stage1)
    patterns: list[str] = list(stage1.get("detected_patterns") or [])
    seen = {str(p).strip().lower() for p in patterns}

    blob = " ".join(str(s) for s in (stage1.get("key_signals") or [])).lower()
    risk = str(stage1.get("risk_warning") or "").lower()
    text = f"{blob} {risk}"
    changed = patterns != before

    if _mentions_hl_count_setup(text) or _hl_tags_from_text(text):
        for hl in _hl_tags_from_text(text) or ("h1", "h2", "l1", "l2"):
            if hl not in seen:
                seen.add(hl)
                patterns.append(hl)
                changed = True

    for keywords, required in PATTERN_KEYWORD_TAGS:
        if required == "h1":
            continue
        if required == "barbwire":
            if not any(k.lower() in text for k in keywords):
                continue
            if not _should_sync_barbwire_tag(stage1, text):
                continue
        if any(k.lower() in text for k in keywords) and required not in seen:
            seen.add(required)
            patterns.append(required)
            changed = True

    est = _entry_setup_type(stage1)
    if est == "wedge" and "wedge" not in seen:
        patterns.append("wedge")
        changed = True
    if est == "breakout_pullback" and "breakout_pullback" not in seen:
        patterns.append("breakout_pullback")
        changed = True
    if est == "tr_boundary":
        for required in ("middle_range", "barbwire"):
            if required == "barbwire" and not _barbwire_candidate(stage1):
                continue
            if required not in seen:
                seen.add(required)
                patterns.append(required)
                changed = True

    if changed:
        stage1["detected_patterns"] = patterns
    return changed


def validate_detected_patterns_vs_key_signals(stage1: dict[str, Any]) -> list[str]:
    """Warn when narrative mentions a pattern but detected_patterns omits it.

    Mirrors upstream. Returns a list of human-readable warning strings
    (empty when coherent). Does not mutate ``stage1``.
    """
    errors: list[str] = []
    patterns = {str(p).strip().lower() for p in (stage1.get("detected_patterns") or [])}
    blob = " ".join(str(s) for s in (stage1.get("key_signals") or [])).lower()
    risk = str(stage1.get("risk_warning") or "").lower()
    text = f"{blob} {risk}"

    for keywords, required in PATTERN_KEYWORD_TAGS:
        if required == "h1":
            if _mentions_hl_count_setup(text):
                if not patterns.intersection({"h1", "h2", "l1", "l2"}):
                    errors.append(
                        "key_signals mentions H1/H2/L1/L2 count setup but "
                        "detected_patterns lacks h1/h2/l1/l2"
                    )
            continue
        if required == "barbwire":
            if not any(k.lower() in text for k in keywords):
                continue
            if not _should_sync_barbwire_tag(stage1, text):
                continue
        if any(k.lower() in text for k in keywords):
            if required not in patterns:
                errors.append(
                    f"key_signals/risk_warning mentions pattern related to {required!r} "
                    f"but detected_patterns lacks {required!r}"
                )

    est = _entry_setup_type(stage1)
    if est == "wedge" and "wedge" not in patterns:
        errors.append(
            "bar_analysis.entry_setup_type=wedge requires detected_patterns to include 'wedge'"
        )
    if est == "breakout_pullback" and "breakout_pullback" not in patterns:
        errors.append(
            "bar_analysis.entry_setup_type=breakout_pullback requires "
            "detected_patterns to include 'breakout_pullback'"
        )
    if est == "tr_boundary":
        for required in ("middle_range", "barbwire"):
            if required == "barbwire" and not _barbwire_candidate(stage1):
                continue
            if required not in patterns:
                errors.append(
                    f"bar_analysis.entry_setup_type=tr_boundary requires "
                    f"detected_patterns to include {required!r}"
                )

    return errors
