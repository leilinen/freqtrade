"""Previous-decision continuity: invalidation checks, flip cooldown, Stage-2 prompt block.

Ported from ``PA_Agent/pa_agent/ai/decision_continuity.py``. The logic is
preserved verbatim. Adaptations:

* Upstream reads the previous decision from a ``trade_records`` CSV *or* an
  analysis record. The freqtrade port has only the DB-backed repository
  (see :meth:`repository.PriceActionRepository.get_previous_successful_analysis`),
  so the CSV fallback (``load_last_trade_csv_row`` / ``audit_relation_fields``
  / ``_TRADE_RECORDS_DIR``) is dropped — the caller always supplies the DB
  record dict ``{"candle_time": iso, "decision": <stage2 JSON>, ...}``.

* Upstream operates on a ``kline_frame`` object with ``.bars[0]`` being K1
  and ``getattr(bar, "close")`` style access. The port operates on
  ``feature_rows: list[dict]`` where ``rows[0]`` is K1 and fields are dict
  keys (``row["close"]``). See :mod:`.features`.

* ``build_continuity_context`` no longer takes a ``frame`` object; instead it
  takes the explicit scalars it needs (``feature_rows``, ``symbol``,
  ``timeframe``, ``candle_time_iso``) so callers that already hold those
  values don't need to wrap them in a frame object.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .price_tick import infer_price_tick_from_rows

# Default: no opposite-direction plan at the same structure within N closed bars.
DEFAULT_STRUCTURE_FLIP_COOLDOWN_BARS = 3
_STRUCTURE_TOLERANCE_TICKS = 3

_REL_SAME = "same_direction"
_REL_FLIP = "flip"
_REL_INVALIDATED = "invalidated"
_REL_FIRST = "first"
_REL_NO_ORDER_PREV = "no_order_prev"
_REL_WAIT = "wait_continued"

_REL_LABELS_ZH = {
    _REL_SAME: "同向",
    _REL_FLIP: "反手",
    _REL_INVALIDATED: "已失效",
    _REL_FIRST: "首单",
    _REL_NO_ORDER_PREV: "上轮无单",
    _REL_WAIT: "延续等待",
}


def _parse_price(raw: object) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        v = float(raw)
        return v if v == v else None  # NaN guard
    except (TypeError, ValueError):
        return None


def order_direction_sign(direction: str | None) -> int:
    if not direction:
        return 0
    d = str(direction).strip().lower()
    if "多" in d or d in ("long", "buy", "bullish"):
        return 1
    if "空" in d or d in ("short", "sell", "bearish"):
        return -1
    return 0


def order_direction_label(sign: int) -> str:
    if sign > 0:
        return "做多"
    if sign < 0:
        return "做空"
    return "—"


def extract_always_in_branch(stage1_json: dict | None) -> str | None:
    """Return AIL / AIS from gate_trace §2.4, or None if not Always In."""
    if not stage1_json:
        return None
    for node in stage1_json.get("gate_trace") or []:
        if not isinstance(node, dict):
            continue
        nid = str(node.get("node_id", "")).replace("§", "")
        if nid not in ("2.4", "2.4.0"):
            continue
        if str(node.get("answer", "")).strip() not in ("是", "yes", "true"):
            continue
        branch = str(node.get("branch") or "").strip().upper()
        if branch in ("AIL", "AIS"):
            return branch
    return None


def _timeframe_minutes(timeframe: str) -> int:
    tf = (timeframe or "").strip().lower()
    m = re.fullmatch(r"(\d+)\s*m", tf)
    if m:
        return max(1, int(m.group(1)))
    h = re.fullmatch(r"(\d+)\s*h", tf)
    if h:
        return max(1, int(h.group(1))) * 60
    d = re.fullmatch(r"(\d+)\s*d", tf)
    if d:
        return max(1, int(d.group(1))) * 1440
    return 5


def bars_elapsed_between(
    prev_time_iso: str | None,
    current_ms: int | None,
    timeframe: str,
    *,
    fallback: int = 1,
) -> int:
    if not prev_time_iso or not current_ms:
        return fallback
    try:
        prev_dt = datetime.strptime(prev_time_iso[:19], "%Y-%m-%d %H:%M:%S")
        prev_ms = int(prev_dt.timestamp() * 1000)
    except (TypeError, ValueError, OSError):
        return fallback
    bar_ms = _timeframe_minutes(timeframe) * 60 * 1000
    if bar_ms <= 0:
        return fallback
    return max(1, round((int(current_ms) - prev_ms) / bar_ms))


def _iso_to_ms(iso: str | None) -> int | None:
    """Parse an ISO timestamp string (as emitted by the port) into epoch ms.

    The port stores ``candle_time`` as a UTC-naive ISO string
    (``features._to_utc_naive(...).isoformat()``). Handle the trailing ``Z``
    and ``+00:00`` variants the repository / AI may emit.
    """
    if not iso:
        return None
    text = str(iso)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        # Fallback to a space-separated strptime (upstream format).
        try:
            dt = datetime.strptime(str(iso)[:19], "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError, OSError):
            return None
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return int(dt.timestamp() * 1000)


def entries_same_structure(
    entry_a: float | None,
    entry_b: float | None,
    *,
    tick: float,
    tolerance_ticks: int = _STRUCTURE_TOLERANCE_TICKS,
) -> bool:
    if entry_a is None or entry_b is None:
        return False
    tol = max(float(tick) * tolerance_ticks, float(tick))
    return abs(entry_a - entry_b) <= tol


def is_order_plan(decision: dict | None) -> bool:
    if not decision:
        return False
    ot = str(decision.get("order_type") or "").strip()
    return ot not in ("", "不下单", "none", "null")


# ── Plan invalidation (adapted to feature_rows) ─────────────────────────────


def assess_plan_invalidation(
    decision: dict | None,
    feature_rows: list[dict[str, Any]] | None,
) -> tuple[bool, str]:
    """True if latest closed bar (K1) invalidated the previous plan (stop touched).

    Adapted from upstream: K1 is ``feature_rows[0]`` and OHLC are dict keys.
    """
    if not is_order_plan(decision):
        return False, ""
    sign = order_direction_sign(str(decision.get("order_direction") or ""))
    stop = _parse_price(decision.get("stop_loss_price"))
    if not feature_rows or stop is None or sign == 0:
        return False, ""

    k1 = feature_rows[0]
    close = float(k1["close"])
    high = float(k1["high"])
    low = float(k1["low"])

    if sign < 0:
        if close >= stop or high >= stop:
            return True, f"K1 触及/突破止损 {stop}（做空方案失效）"
    elif sign > 0:
        if close <= stop or low <= stop:
            return True, f"K1 触及/跌破止损 {stop}（做多方案失效）"
    return False, ""


# ── Previous-record adapters (DB record shape) ──────────────────────────────


def decision_from_previous_record(previous_record: Any) -> dict[str, Any] | None:
    """Extract the stage-2 ``decision`` dict from a port DB record.

    The port's ``repository.get_previous_successful_analysis`` returns
    ``{"candle_time": iso, "decision": <stage2 JSON>, ...}`` where the
    stage-2 JSON itself nests the decision under its own ``decision`` key.
    Upstream read ``previous_record.stage2_decision.decision``; here we read
    ``previous_record["decision"]["decision"]``.
    """
    if previous_record is None:
        return None
    s2 = (
        previous_record.get("decision")
        if isinstance(previous_record, dict)
        else None
    )
    if not isinstance(s2, dict):
        return None
    dec = s2.get("decision")
    return dec if isinstance(dec, dict) else None


def previous_record_time_iso(previous_record: Any) -> str | None:
    """Return the candle-time ISO string of a port DB record."""
    if previous_record is None:
        return None
    if isinstance(previous_record, dict):
        return previous_record.get("candle_time")
    return None


def classify_vs_previous(
    prev_decision: dict | None,
    curr_decision: dict | None,
    *,
    feature_rows: list[dict[str, Any]] | None,
    prev_invalidated: bool,
    same_structure: bool,
) -> str:
    if prev_decision is None or not is_order_plan(prev_decision):
        return _REL_FIRST if is_order_plan(curr_decision) else _REL_NO_ORDER_PREV
    if not is_order_plan(curr_decision):
        return _REL_WAIT if not prev_invalidated else _REL_INVALIDATED

    prev_sign = order_direction_sign(str(prev_decision.get("order_direction") or ""))
    curr_sign = order_direction_sign(str(curr_decision.get("order_direction") or ""))
    if prev_invalidated:
        return _REL_INVALIDATED
    if prev_sign == curr_sign:
        return _REL_SAME
    if prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
        return _REL_FLIP if same_structure else _REL_FLIP
    return _REL_SAME


def build_continuity_context(
    *,
    feature_rows: list[dict[str, Any]] | None,
    stage1_json: dict,
    symbol: str = "",
    timeframe: str = "",
    candle_time_iso: str | None = None,
    previous_record: Any | None = None,
    cooldown_bars: int = DEFAULT_STRUCTURE_FLIP_COOLDOWN_BARS,
) -> dict[str, Any]:
    """Assemble continuity facts for prompt injection.

    Adapted from upstream ``build_continuity_context``: takes explicit
    scalars instead of a ``frame`` object, and drops the CSV fallback (the
    port only has the DB repository). ``current_ms`` is derived from
    ``candle_time_iso``.
    """
    tick = infer_price_tick_from_rows(feature_rows) if feature_rows else 0.01

    prev_decision = decision_from_previous_record(previous_record)
    prev_time = previous_record_time_iso(previous_record)

    current_ms = _iso_to_ms(candle_time_iso)
    bars_since = bars_elapsed_between(prev_time, current_ms, timeframe)

    invalidated, invalidation_reason = assess_plan_invalidation(prev_decision, feature_rows)
    prev_entry = _parse_price((prev_decision or {}).get("entry_price"))
    direction = str(stage1_json.get("direction") or "neutral")
    always_in = extract_always_in_branch(stage1_json)

    return {
        "has_previous_plan": is_order_plan(prev_decision),
        "previous_decision": prev_decision or {},
        "previous_time": prev_time,
        "previous_source": "db_record",
        "bars_since": bars_since,
        "cooldown_bars": max(1, int(cooldown_bars)),
        "invalidated": invalidated,
        "invalidation_reason": invalidation_reason,
        "previous_entry": prev_entry,
        "tick": tick,
        "direction": direction,
        "always_in_branch": always_in,
        "timeframe": timeframe,
        "symbol": symbol,
    }


def render_continuity_prompt_block(ctx: dict[str, Any]) -> str:
    if not ctx.get("has_previous_plan"):
        direction = ctx.get("direction", "neutral")
        always_in = ctx.get("always_in_branch")
        neutral_lines = [
            "## 方案连续性规则（程序强制，阶段二必须遵守）",
            "",
            "### A. direction=neutral 时的方向约束",
            "- 阶段一 `direction=neutral` 时，**禁止**在上下边界双向同时给刮头皮方案。",
        ]
        if always_in == "AIL":
            neutral_lines.append(
                "- 当前 §2.4=**AIL** → **仅允许做多侧** setup（回踩支撑/下边界/顺势回撤做多）。"
                "禁止 §9.0P 阻力做空。"
            )
        elif always_in == "AIS":
            neutral_lines.append(
                "- 当前 §2.4=**AIS** → **仅允许做空侧** setup（反弹阻力/上边界/顺势回撤做空）。"
                "禁止 §9.0P 支撑做多。"
            )
        else:
            neutral_lines.append(
                "- 当前 §2.4 **非** Always In → §9.0P 计划型限价默认 **wait**；"
                "仅当出现与 §2.4 方向一致的强信号棒（§9.0=是）才可下单。"
            )
        neutral_lines.extend([
            "",
            "### B. 同结构位反手冷却",
            f"- 若上一轮有可执行方案且**未失效**，{ctx.get('cooldown_bars', 3)} 根已收盘 K 线内，"
            "禁止在**同一结构位**（entry 相差≤3跳）提出**反向**新方案；"
            "除非 K1 **收盘**突破上一轮 `invalidation_condition` / 止损结构位。",
            "",
            "（本轮无上一轮下单方案记录，仅适用 A/B 通用规则。）",
        ])
        return "\n".join(neutral_lines)

    prev = ctx.get("previous_decision") or {}
    prev_dir = str(prev.get("order_direction") or "—")
    prev_type = str(prev.get("order_type") or "—")
    prev_entry = prev.get("entry_price", "—")
    prev_stop = prev.get("stop_loss_price", "—")
    prev_time = ctx.get("previous_time") or "—"
    bars_since = ctx.get("bars_since", 1)
    cooldown = ctx.get("cooldown_bars", 3)
    inv = ctx.get("invalidated", False)
    inv_reason = ctx.get("invalidation_reason") or ""

    status = "**已失效**" if inv else "**未失效（仍有效）**"
    if inv and inv_reason:
        status += f"：{inv_reason}"

    direction = ctx.get("direction", "neutral")
    always_in = ctx.get("always_in_branch")

    lines = [
        "## 上一轮交易方案连续性（程序评估，阶段二必须遵守）",
        "",
        f"上一轮（{prev_time}，约 {bars_since} 根 {ctx.get('timeframe', '')} K 线前）方案：",
        f"- **{prev_dir}** {prev_type} @ {prev_entry}，止损 {prev_stop}",
        f"- 程序判定：{status}",
        "",
        "### 裁定（按优先级）",
        f"1. **未失效** → 默认 `order_type=不下单`、`terminal.outcome=wait`，"
        "在 watch_points 说明仍等待上一轮 setup 触发；"
        "**禁止**立即在相近结构位反手，除非 K1 收盘已触发失效。",
        f"2. **同结构位反手冷却**：{cooldown} 根 K 线内，"
        "若新 entry 与上一轮 entry 相差≤3跳，**禁止反向**新单（失效后除外）。",
        "3. **direction=neutral** 时仅顺 §2.4：",
    ]
    if always_in == "AIL":
        lines.append("   - 当前 **AIL** → 只允许做多侧；禁止做空限价/突破。")
    elif always_in == "AIS":
        lines.append("   - 当前 **AIS** → 只允许做空侧；禁止做多限价/突破。")
    else:
        lines.append("   - §2.4 非 Always In → 无强信号则 wait，禁止边界双向刮头皮。")

    if direction != "neutral":
        lines.append(f"   - （本轮 direction={direction}，neutral 约束不适用。）")

    lines.extend([
        "",
        "若确需覆盖上述连续性规则，须在 `decision.reasoning` **首句**写明「连续性覆盖」及 K 线收盘证据。",
    ])
    return "\n".join(lines)


def continuity_violation_reason(
    ctx: dict[str, Any],
    decision: dict,
) -> str | None:
    """Return a reason string if decision violates continuity rules (for normalizer guard)."""
    if not isinstance(decision, dict):
        return None
    if not is_order_plan(decision):
        return None

    direction = str(ctx.get("direction") or "neutral")
    always_in = ctx.get("always_in_branch")
    curr_sign = order_direction_sign(str(decision.get("order_direction") or ""))

    if direction == "neutral":
        if always_in == "AIL" and curr_sign < 0:
            return "direction=neutral 且 §2.4=AIL：禁止做空方案"
        if always_in == "AIS" and curr_sign > 0:
            return "direction=neutral 且 §2.4=AIS：禁止做多方案"
        if always_in is None and curr_sign != 0:
            return "direction=neutral 且 §2.4 非 Always In：禁止无强信号的方向性方案"

    if not ctx.get("has_previous_plan"):
        return None

    prev = ctx.get("previous_decision") or {}
    prev_sign = order_direction_sign(str(prev.get("order_direction") or ""))
    prev_entry = ctx.get("previous_entry")
    curr_entry = _parse_price(decision.get("entry_price"))
    tick = float(ctx.get("tick") or 0.01)
    same_struct = entries_same_structure(prev_entry, curr_entry, tick=tick)
    bars_since = int(ctx.get("bars_since") or 1)
    cooldown = int(ctx.get("cooldown_bars") or DEFAULT_STRUCTURE_FLIP_COOLDOWN_BARS)
    invalidated = bool(ctx.get("invalidated"))

    if not invalidated and prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
        if same_struct and bars_since <= cooldown:
            return (
                f"上一轮方案未失效，{bars_since} 根 K 线内同结构位反手"
                f"（entry {prev_entry} → {curr_entry}）"
            )
    return None


def apply_continuity_guard(
    stage2: dict[str, Any],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Force wait/no-order when continuity rules are clearly violated."""
    if not ctx or not isinstance(stage2, dict):
        return stage2

    decision = stage2.get("decision")
    if not isinstance(decision, dict):
        return stage2

    reason = continuity_violation_reason(ctx, decision)
    if not reason:
        return stage2

    decision = dict(decision)
    decision["order_type"] = "不下单"
    for key in (
        "order_direction",
        "entry_price",
        "stop_loss_price",
        "take_profit_price",
        "take_profit_price_2",
        "entry_rule",
        "entry_basis_bar",
        "entry_basis_extreme",
        "estimated_win_rate",
    ):
        decision[key] = None

    existing = str(decision.get("reasoning") or "")
    prefix = f"【程序连续性守卫】{reason}；改为不下单。 "
    # Keep in sync with stage2_normalizer.DECISION_REASONING_MAX_LEN (schema maxLength).
    max_len = 280
    budget = max_len - len(prefix)
    if budget < 1:
        decision["reasoning"] = prefix[:max_len]
    else:
        body = existing
        if len(body) > budget:
            body = body[: budget - 1] + "…"
        decision["reasoning"] = prefix + body

    terminal = dict(stage2.get("terminal") or {})
    terminal["outcome"] = "wait"
    terminal["node_id"] = "continuity"
    terminal["label"] = "方案连续性守卫"

    stage2 = dict(stage2)
    stage2["decision"] = decision
    stage2["terminal"] = terminal
    return stage2
