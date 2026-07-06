"""Trace normalize layer (ported from PA_Agent ``trace_normalize.py``).

Stage1 (gate_trace) and Stage2 (decision_trace) often carry LLM quirks:
bar_range fields in wrong order (K1-K8 instead of K8-K1), comma-separated
shorthand (K7,K1), placeholder values (pending / tbd / waiting), or
bar_range entirely missing while the reason text cites K-lines. Upstream
runs this normalize layer BEFORE the validator so the validator can rely
on canonical strings.

This module ports the **bar_range subset** (Batch A) of upstream
``trace_normalize.py``:

* :func:`infer_max_bar_seq_from_trace`
* :func:`fix_bar_range_string`
* :func:`normalize_trace_item` (bar_range subset only; answer-alias
  resolution lives in :mod:`normalize` via ``_resolve_trace_answers``)
* :func:`normalize_trace_list` (bar_range subset)

Subsequent batches will port chapter ordering, gate_12/23 sync, gate_result
repair, terminal repair, and program-reference block stripping.
"""
from __future__ import annotations

import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

# ── Regexes ──
_BAR_RANGE_RE = re.compile(r"^K(\d+)-K(\d+)$", re.IGNORECASE)
_SINGLE_BAR_RE = re.compile(r"^K(\d+)$", re.IGNORECASE)

# ── Aliases / placeholders ──
_BAR_RANGE_ALIASES = frozenset({"全局", "全图", "整体", "全部", "all"})
_PENDING_BAR_RANGE_VALUES = frozenset(
    {
        "pending",
        "tbd",
        "n/a",
        "na",
        "等待触发",
        "待触发",
        "未触发",
        "尚无",
        "等待",
    }
)
_NULLISH_STRINGS = frozenset({"", "null", "none", "nil", "n/a"})


def _is_nullish(value: Any) -> bool:
    """Return True for None / "" / "null" / "n/a" etc."""
    if value is None:
        return True
    return str(value).strip().lower() in _NULLISH_STRINGS


def ensure_trace_string_fields(item: dict[str, Any]) -> None:
    """Coerce missing / null JSON values to strings before schema validation.

    Mirrors upstream ``_ensure_trace_string_fields``. Schema requires
    node_id/question/answer/reason to be non-null strings; LLMs sometimes
    emit ``null`` or skip keys entirely.
    """
    for key in ("node_id", "question", "answer", "reason"):
        if key not in item or _is_nullish(item.get(key)):
            if key == "answer" and item.get("skipped"):
                item[key] = "不适用"
            elif key in ("question", "reason"):
                item[key] = "—"
            elif key == "node_id":
                item[key] = ""
            elif key == "answer":
                item[key] = "不适用" if item.get("skipped") else "否"
            else:
                item[key] = ""


def infer_max_bar_seq_from_trace(trace: list[Any]) -> int | None:
    """Infer largest K index mentioned in trace bar_range or reason text.

    Used as ``default_max_seq`` fallback when feature_rows is unavailable
    (e.g. replay from stored JSON without kline context).
    """
    max_seq = 0
    for item in trace:
        if not isinstance(item, dict):
            continue
        for field in ("bar_range", "reason"):
            raw = str(item.get(field, "") or "")
            for m in re.finditer(r"K(\d+)", raw, re.IGNORECASE):
                max_seq = max(max_seq, int(m.group(1)))
    return max_seq or None


def _comma_separated_bar_range(compact: str) -> str | None:
    """Turn "K7,K1" or "K1、K7" into "K7-K1" when two or more K refs present."""
    # Need at least one comma-like separator to attempt parsing.
    if not any(sep in compact for sep in (",", "，", "、")):  # noqa: RUF001
        return None
    parts = re.split(r"[,，、]", compact)  # noqa: RUF001  ASCII / fullwidth / ideographic
    seqs: list[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = _SINGLE_BAR_RE.match(part)
        if m:
            seqs.append(int(m.group(1)))
    if len(seqs) < 2:
        return None
    older, newer = max(seqs), min(seqs)
    return f"K{older}" if older == newer else f"K{older}-K{newer}"


def _bar_range_is_canonical(text: str) -> bool:
    compact = str(text or "").strip().upper().replace(" ", "")
    if not compact or compact in ("不适用", "—", "-"):
        return True
    return bool(_BAR_RANGE_RE.match(compact) or _SINGLE_BAR_RE.match(compact))


def _bar_seqs_from_range_text(bar_range: str) -> set[int]:
    """Return all K indices covered by a bar_range string (e.g. K8-K1 → {1..8})."""
    text = (bar_range or "").strip().upper().replace(" ", "")
    if not text or text in ("不适用", "—", "全局", "GLOBAL"):
        return set()
    m = _BAR_RANGE_RE.match(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = min(a, b), max(a, b)
        return set(range(lo, hi + 1))
    single = _SINGLE_BAR_RE.match(text)
    if single:
        return {int(single.group(1))}
    return set()


def _bar_seqs_from_reason_text(reason: str) -> set[int]:
    """Extract all K-index citations from a free-text reason."""
    return {
        int(m.group(1))
        for m in re.finditer(r"K\s*(\d+)", reason or "", re.IGNORECASE)
    }


def _bar_range_from_reason(
    item: dict[str, Any],
    *,
    default_max_seq: int | None = None,
) -> str | None:
    """Infer a bar_range string from K citations in the reason text."""
    cited = _bar_seqs_from_reason_text(str(item.get("reason", "") or ""))
    if not cited:
        return None
    if default_max_seq and default_max_seq >= 1:
        cited = {s for s in cited if 1 <= s <= default_max_seq}
        if not cited:
            return None
    older, newer = max(cited), min(cited)
    return f"K{older}" if older == newer else f"K{older}-K{newer}"


def _cap_bar_seq(seq: int, max_seq: int | None) -> int:
    if max_seq is not None and max_seq >= 1:
        return max(1, min(seq, max_seq))
    return seq


def fix_bar_range_string(text: str, *, default_max_seq: int | None = None) -> str:
    """Canonicalize bar_range: order, aliases, spacing.

    Handles:
    * ``"K1-K8"`` (reversed) → ``"K8-K1"`` (K1 = newest, K8 = oldest)
    * ``"全局" / "all"`` → ``"K{N}-K1"`` when default_max_seq known, else ``"不适用"``
    * ``"pending" / "tbd" / "等待"`` → ``""`` (placeholder, must be inferred)
    * Comma/顿号 shorthand ``"K7,K1"`` → ``"K7-K1"``
    * Cap seqs at ``default_max_seq`` to keep within frame
    """
    raw = str(text).strip()
    if not raw:
        return ""

    if raw.lower() in _PENDING_BAR_RANGE_VALUES:
        return ""

    if raw in _BAR_RANGE_ALIASES or raw.lower() in {"global", "all"}:
        if default_max_seq and default_max_seq > 1:
            return f"K{default_max_seq}-K1"
        return "不适用"

    compact = raw.upper().replace(" ", "")
    comma_range = _comma_separated_bar_range(compact)
    if comma_range:
        compact = comma_range.replace(" ", "")
    m = _BAR_RANGE_RE.match(compact)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a == b:
            capped = _cap_bar_seq(a, default_max_seq)
            return f"K{capped}"
        if a < b:
            logger.debug(
                "bar_range=%r has reversed order (K%d before K%d); "
                "K1=newest, K{N}=older. Auto-fixing to K%d-K%d.",
                text,
                a,
                b,
                b,
                a,
            )
            a, b = b, a
        a = _cap_bar_seq(a, default_max_seq)
        b = _cap_bar_seq(b, default_max_seq)
        if a == b:
            return f"K{a}"
        return f"K{a}-K{b}"

    single = _SINGLE_BAR_RE.match(compact)
    if single:
        return f"K{_cap_bar_seq(int(single.group(1)), default_max_seq)}"

    return raw


def _expand_bar_range_for_reason_citations(
    item: dict[str, Any],
    *,
    default_max_seq: int | None = None,
) -> None:
    """Widen bar_range when reason cites K-lines outside the declared window.

    Avoids validator complaints like ``decision_trace_bar_range_out_of_frame``
    when the reason legitimately references context bars beyond the bar_range.
    """
    reason = str(item.get("reason", "") or "")
    br = str(item.get("bar_range", "") or "").strip()
    cited = _bar_seqs_from_reason_text(reason)
    if not cited or br in ("不适用", "—", "全局", "GLOBAL", ""):
        return

    allowed = _bar_seqs_from_range_text(br)
    if not allowed:
        return
    if cited.issubset(allowed):
        return

    merged = allowed | cited
    # Clip only when frame cap is known; always keep cited seqs from reason.
    if default_max_seq and default_max_seq >= 1:
        merged = {s for s in merged if 1 <= s <= default_max_seq} | cited
        if not merged:
            return

    older, newer = max(merged), min(merged)
    expanded = f"K{older}" if older == newer else f"K{older}-K{newer}"
    if expanded != br:
        logger.debug(
            "bar_range expanded %s -> %s (node %s, cited=%s)",
            br,
            expanded,
            item.get("node_id"),
            sorted(cited),
        )
        item["bar_range"] = expanded


def _resolve_missing_bar_range(
    item: dict[str, Any],
    *,
    nid: str,
    skipped: bool,
    ans: str,
    note: str,
    default_max_seq: int | None,
    prior_bar_range: str | None,
) -> bool:
    """Try to fill a missing/placeholder bar_range. Return True when set.

    Shared by the placeholder branch and the null branch of
    :func:`_coerce_bar_range`. ``note`` is included in debug logs so we can
    distinguish "null" vs "placeholder" origins.
    """
    inferred = _bar_range_from_reason(item, default_max_seq=default_max_seq)
    if inferred:
        item["bar_range"] = inferred
        logger.debug("bar_range %s -> %s (node %s, from reason)", note, inferred, nid)
        _expand_bar_range_for_reason_citations(item, default_max_seq=default_max_seq)
        return True
    if skipped or ans == "不适用":
        item["bar_range"] = "不适用"
        return True
    if prior_bar_range and prior_bar_range not in ("不适用", "—"):
        item["bar_range"] = prior_bar_range
        logger.debug("bar_range %s -> copied %s (node %s)", note, prior_bar_range, nid)
        return True
    if default_max_seq and default_max_seq > 1:
        item["bar_range"] = f"K{default_max_seq}-K1"
        logger.debug("bar_range %s -> K%s-K1 (node %s)", note, default_max_seq, nid)
        return True
    item["bar_range"] = "不适用"
    return True


def _coerce_bar_range(
    item: dict[str, Any],
    *,
    default_max_seq: int | None = None,
    prior_bar_range: str | None = None,
) -> None:
    """Ensure bar_range is a non-null canonical string.

    Decision tree (mirrors upstream ``_coerce_bar_range``):

    1. If skipped/answer=不适用 → ``"不适用"``
    2. If bar_range is placeholder (pending/tbd/...) or null:
       a. try infer from reason text
       b. fall back to prior node's bar_range (lenient mode)
       c. fall back to ``K{N}-K1`` when default_max_seq known
       d. last resort: ``"不适用"``
    3. Otherwise canonicalize via :func:`fix_bar_range_string` and expand
       when reason cites bars outside the window.
    """
    nid = str(item.get("node_id", ""))
    skipped = bool(item.get("skipped"))
    ans = str(item.get("answer", "") or "").strip()

    if skipped and not ans:
        item["answer"] = "不适用"
        ans = "不适用"

    br = item.get("bar_range")
    br_text = str(br or "").strip()
    if br_text.lower() in _PENDING_BAR_RANGE_VALUES:
        if _resolve_missing_bar_range(
            item,
            nid=nid,
            skipped=skipped,
            ans=ans,
            note=str(br),
            default_max_seq=default_max_seq,
            prior_bar_range=prior_bar_range,
        ):
            return

    if _is_nullish(br):
        _resolve_missing_bar_range(
            item,
            nid=nid,
            skipped=skipped,
            ans=ans,
            note="null",
            default_max_seq=default_max_seq,
            prior_bar_range=prior_bar_range,
        )
        return

    fixed = fix_bar_range_string(str(br), default_max_seq=default_max_seq)
    if fixed != br_text:
        logger.debug("bar_range %s -> %s (node %s)", br, fixed, nid)
    item["bar_range"] = fixed
    if not _bar_range_is_canonical(item["bar_range"]):
        inferred = _bar_range_from_reason(item, default_max_seq=default_max_seq)
        if inferred:
            item["bar_range"] = inferred
            logger.debug(
                "bar_range %r -> %s (node %s, repaired non-canonical)",
                br,
                inferred,
                nid,
            )
        elif skipped or ans == "不适用":
            item["bar_range"] = "不适用"
    _expand_bar_range_for_reason_citations(item, default_max_seq=default_max_seq)


def normalize_trace_item_bar_range(
    item: dict[str, Any],
    *,
    default_max_seq: int | None = None,
    prior_bar_range: str | None = None,
) -> None:
    """Mutate one trace item's bar_range (Batch A subset of upstream).

    Answer-alias resolution is handled separately by ``_resolve_trace_answers``
    in :mod:`normalize`; this function focuses purely on bar_range repair.
    """
    if not isinstance(item, dict):
        return
    ensure_trace_string_fields(item)
    _coerce_bar_range(
        item,
        default_max_seq=default_max_seq,
        prior_bar_range=prior_bar_range,
    )


def normalize_trace_list_bar_range(
    trace: list[Any] | None,
    *,
    default_max_seq: int | None = None,
) -> list[Any] | None:
    """Canonicalize bar_range across a trace list (Batch A subset).

    * Computes ``default_max_seq`` from the trace itself when not given
    * Walks trace in order, carrying the last valid bar_range forward
      so subsequent placeholder/null bar_ranges can inherit it
    """
    if not isinstance(trace, list):
        return trace

    max_seq = default_max_seq or infer_max_bar_seq_from_trace(trace)
    last_br: str | None = None
    for item in trace:
        if not isinstance(item, dict):
            continue
        normalize_trace_item_bar_range(
            item,
            default_max_seq=max_seq,
            prior_bar_range=last_br,
        )
        br = str(item.get("bar_range", "") or "")
        if br and br not in ("不适用", "—", "-"):
            last_br = br
    return trace
