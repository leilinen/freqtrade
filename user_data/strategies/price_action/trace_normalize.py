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

# Program-injected reference blocks like 【程序参考数据（市场特征）：...】
_PROG_REF_BLOCK_RE = re.compile(r"【程序参考数据（[^】]*）：.*?】", re.DOTALL)

# Model-invented summary nodes; gate_result belongs in gate_result field.
_GATE_END_NODE_IDS = frozenset({"gate_end", "gate_summary", "summary"})

# Tokens that mark a "proceed" rationale as complete.
_PROCEED_FINAL_TOKENS: tuple[str, ...] = (
    "进入阶段二",
    "可进入阶段二",
    "闸门通过",
    "继续阶段二",
    "进入策略",
    "可继续分析",
)

# When the AI writes the gate_result token as a trace answer
# (e.g. node 2.5 answer="proceed"), map it back to the schema enum.
_GATE_RESULT_ANSWER_ALIASES: dict[str, str] = {
    "proceed": "是",
    "wait": "等待",
    "unknown": "中性",
}

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
    # Clip to the valid bar window [1, default_max_seq]. Out-of-window
    # citations in the reason text (e.g. K0 for the forming bar, or K10
    # when only 8 bars are in frame) are dropped — they would push
    # bar_range into validator-rejected state. The narrative mention
    # itself is fine, but bar_range must stay within frame.
    if default_max_seq and default_max_seq >= 1:
        merged = {s for s in merged if 1 <= s <= default_max_seq}
        if not merged:
            return
    else:
        # No frame cap known: at minimum drop K0 (validator hard-rejects it).
        merged = {s for s in merged if s >= 1}
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
    """Canonicalize a trace list (Batch A + Batch B).

    Batch A: bar_range canonicalization per item, carrying last valid
    bar_range forward so placeholder/null entries can inherit it.
    Batch B: sort items by ``node_id`` chapter prefix before per-item
    processing so the ``decision_trace_chapter_order_invalid`` /
    ``market_diagnosis_gate_trace_node_order_invalid`` validator checks
    pass for traces the AI emitted out of order.
    """
    if not isinstance(trace, list):
        return trace

    sort_trace_by_chapter(trace)

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


# ── Batch B: chapter ordering + terminal repair + strip AI §14.1 ──

_CHAPTER_ORDER: dict[str, int] = {
    "1.": 10,
    "2.": 20,
    "3.": 30,
    "4.": 40,
    "5.": 50,
    "6.": 60,
    "7.": 70,
    "8.": 80,
    "9.": 90,
    "10.": 100,
    "11.": 110,
    "12.": 120,
    "13.": 130,
    "14": 140,
}


def _chapter_rank(item: Any) -> int:
    """Sort rank for a trace item based on its ``node_id`` chapter prefix.

    Mirrors upstream ``normalize_trace_list._chapter_rank``. Unrecognized
    nodes land in the middle (500) so they don't displace real chapters.
    """
    if not isinstance(item, dict):
        return 999
    nid = str(item.get("node_id", "") or "").strip()
    for prefix, rank in _CHAPTER_ORDER.items():
        if nid.startswith(prefix) or nid == prefix.rstrip("."):
            return rank
    return 500


def sort_trace_by_chapter(trace: list[Any]) -> None:
    """Reorder trace items by ``node_id`` chapter (in-place).

    AI may output nodes in any order; the canonical order is by chapter
    prefix (1.x → 2.x → ... → 14). Mirrors upstream
    ``normalize_trace_list`` sort step.
    """
    if not isinstance(trace, list):
        return
    trace.sort(key=_chapter_rank)


def strip_ai_gate_14(gate_trace: list[Any]) -> int:
    """Remove duplicate AI-written §14.1 nodes from gate_trace.

    The model sometimes emits node 14.1 (禁止行为扫描) in gate_trace.
    We keep the first occurrence and drop subsequent duplicates so the
    validator's node-ordering check passes. Returns the number removed.
    """
    if not isinstance(gate_trace, list) or not gate_trace:
        return 0
    kept: list[Any] = []
    seen_14 = False
    removed = 0
    for item in gate_trace:
        if isinstance(item, dict) and str(item.get("node_id", "") or "").strip() == "14.1":
            if not seen_14:
                seen_14 = True
                kept.append(item)
            else:
                removed += 1
        else:
            kept.append(item)
    if removed:
        gate_trace[:] = kept
        logger.debug("Stripped %s duplicate AI-written 14.1 from gate_trace", removed)
    return removed


def repair_stage2_terminal(obj: dict[str, Any]) -> bool:
    """When 10.3 is 否 on a no-order path, terminal must cite node 10.3.

    Mirrors upstream ``_repair_stage2_terminal``. Returns True when
    ``terminal.node_id`` was adjusted.
    """
    trace = obj.get("decision_trace")
    terminal = obj.get("terminal")
    decision = obj.get("decision")
    if not isinstance(trace, list) or not isinstance(terminal, dict):
        return False
    if not isinstance(decision, dict) or decision.get("order_type") != "不下单":
        return False
    if terminal.get("outcome") not in ("wait", "reject"):
        return False

    for item in trace:
        if not isinstance(item, dict) or str(item.get("node_id", "") or "").strip() != "10.3":
            continue
        if str(item.get("answer", "") or "").strip() != "否":
            return False
        old_nid = str(terminal.get("node_id", "") or "").strip()
        if old_nid != "10.3":
            terminal["node_id"] = "10.3"
            logger.debug(
                "stage2 terminal.node_id %r -> 10.3 (10.3 answer=否, order_type=不下单)",
                old_nid,
            )
            return True
        return False
    return False


# ── Batch C: program-reference stripping, gate-end removal, answer aliases ──


def strip_program_reference_blocks(text: str) -> str:
    """Remove merged program-metric blocks from trace reason (§2.5 cleanup).

    Mirrors upstream ``_strip_program_reference_blocks``. The orchestrator
    injects 【程序参考数据（市场特征）：...】 blocks into node 2.5 reason; the
    model sometimes echoes them back into the trace, where the validator's
    "no program-reference leakage" rule would reject them. Strip before
    validation.
    """
    cleaned = _PROG_REF_BLOCK_RE.sub("", text or "")
    return " ".join(cleaned.split()).strip()


def repair_gate_trace_answer_aliases(gate_trace: list[Any]) -> None:
    """Map gate_result tokens mistakenly written as trace answer.

    Mirrors upstream ``_repair_gate_trace_answer_aliases``. When the AI
    writes ``answer="proceed"`` (a gate_result enum value, not a valid
    node answer), translate to 是. proceed with empty branch → fill
    ``branch="proceed"`` so downstream direction sync treats it as
    "passed".
    """
    for item in gate_trace:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("answer", "") or "").strip()
        mapped = _GATE_RESULT_ANSWER_ALIASES.get(raw.lower())
        if mapped:
            item["answer"] = mapped
            branch = str(item.get("branch", "") or "").strip()
            if raw.lower() == "proceed" and not branch:
                item["branch"] = "proceed"


def strip_gate_end_nodes(gate_trace: list[Any]) -> int:
    """Drop model-invented gate_end / summary nodes from gate_trace.

    Mirrors upstream ``_strip_gate_end_nodes``. ``gate_result`` lives
    in its own field; nodes named gate_end/gate_summary/summary are
    hallucinations. Returns count removed.
    """
    if not isinstance(gate_trace, list) or not gate_trace:
        return 0
    kept = [
        item
        for item in gate_trace
        if not (
            isinstance(item, dict)
            and str(item.get("node_id", "") or "").strip().lower() in _GATE_END_NODE_IDS
        )
    ]
    removed = len(gate_trace) - len(kept)
    if removed:
        gate_trace[:] = kept
        logger.debug("Stripped %s gate_end/summary nodes from gate_trace", removed)
    return removed


def repair_stage1_gate_trace(obj: dict[str, Any]) -> bool:
    """Format-only stage1 gate_trace repairs before validation.

    Mirrors upstream ``_repair_stage1_gate_trace``. Runs:
    1. answer alias repair (proceed/wait/unknown → enum)
    2. drop gate_end/summary nodes
    3. strip program-reference blocks from §1.3/§2.5 reasons
    4. (delegates _sync_gate_23_with_direction / _repair_gate_result
       to validation.py — already called there)

    Returns True if any mutation happened.
    """
    gate = obj.get("gate_trace")
    if not isinstance(gate, list) or not gate:
        return False

    mutated = False

    repair_gate_trace_answer_aliases(gate)
    if strip_gate_end_nodes(gate):
        mutated = True

    for item in gate:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("node_id", "") or "").strip()
        if nid in ("1.3", "2.5"):
            reason = str(item.get("reason", "") or "")
            stripped = strip_program_reference_blocks(reason)
            if stripped != reason.strip():
                item["reason"] = stripped
                mutated = True

    # Proceed-final-token injection: when gate_result=proceed, ensure the
    # last gate node reason ends with a token marking the gate as cleared.
    if str(obj.get("gate_result", "") or "").strip().lower() == "proceed" and gate:
        last = gate[-1]
        if isinstance(last, dict):
            blob = str(last.get("reason", "") or "")
            if not any(tok in blob for tok in _PROCEED_FINAL_TOKENS):
                last["reason"] = (blob.rstrip("。") + "，闸门通过，可进入阶段二。").strip()
                mutated = True

    return mutated


# ── Canonical question repair (Batch D2) ──
#
# Models often paraphrase the binary decision tree questions
# ("数据是否足够?" → "数据是否充足？"). The canonical wording lives in
# ``二元决策.txt``; these helpers overwrite AI paraphrases with the
# spec text so downstream consumers see consistent node questions.


def _canonical_gate_questions() -> dict[str, str]:
    """Return ``{node_id: canonical_question}`` from the decision tree spec.

    Mirrors upstream ``_canonical_gate_questions``. Same dict serves
    gate_trace (stage1) and decision_trace (stage2).
    """
    # Local import to avoid module-load cycle on missing asset.
    from .decision_tree import canonical_tree_questions

    return canonical_tree_questions()


def repair_stage1_gate_trace_questions(gate_trace: list[Any]) -> bool:
    """Overwrite gate_trace node questions with canonical wording.

    Mirrors upstream behavior inside ``_repair_stage1_gate_trace``
    (the canonical question sync was inline there). Returns True when
    any item was modified.
    """
    canonical_q = _canonical_gate_questions()
    if not canonical_q:
        return False
    changed = False
    for item in gate_trace:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("node_id", "") or "").strip()
        if nid in canonical_q:
            canonical = canonical_q[nid]
            if str(item.get("question", "") or "").strip() != canonical:
                item["question"] = canonical
                changed = True
    return changed


def repair_stage2_decision_trace_questions(trace: list[Any]) -> bool:
    """Align decision_trace question text with the decision tree (format-only).

    Mirrors upstream ``_repair_stage2_decision_trace_questions``.
    Returns True when any item was modified.
    """
    canonical_q = _canonical_gate_questions()
    if not canonical_q:
        return False
    changed = False
    for item in trace:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("node_id", "") or "").strip()
        if nid in canonical_q:
            canonical = canonical_q[nid]
            if str(item.get("question", "") or "").strip() != canonical:
                item["question"] = canonical
                changed = True
    return changed
