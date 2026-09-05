"""PostgreSQL persistence for pa_core analysis records and trade-intent signals.

Replaces the JSON-file PendingWriter for the freqtrade watch system
(migration plan step 6). Two tables, on a SQLAlchemy metadata base that is
deliberately separate from freqtrade's own models:

- ``analysis_record`` — one row per two-stage run (full sanitized record as
  JSON). ``find_latest_successful`` restores the previous AnalysisRecord for
  incremental analysis after a restart.
- ``signal`` — the trade-intent ledger (docs/pa-watch-system-design.md §1.2):
  written only when Stage 2 produced an order plan. Joins with freqtrade's
  ``trades`` table (via ``executed_trade_id``, backfilled in trade mode) give
  fill rate / planned-vs-actual slippage / target-vs-realized analytics.

Error philosophy mirrors PendingWriter: persistence failures are logged and
never propagated into the strategy loop.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from pa_core.datetime_ts import ts_open_to_ms
from pa_core.records.schema import AnalysisRecord, FollowupTurn
from pa_core.util.mask_secret import mask_secret


class PaBase(DeclarativeBase):
    """Metadata base kept separate from freqtrade's SQLBase."""


class AnalysisRecordRow(PaBase):
    """One two-stage analysis run (full record JSON for replay/incremental)."""

    __tablename__ = "analysis_record"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    # K1 (newest closed bar) open time the decision was based on.
    candle_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    is_partial: Mapped[bool] = mapped_column(Boolean, default=False)
    partial_reason: Mapped[Optional[str]] = mapped_column(String(128))
    # Light queryable summary of stage 1.
    stage1_summary: Mapped[Optional[dict]] = mapped_column(JSON)
    record_json: Mapped[dict] = mapped_column(JSON)
    followups: Mapped[list] = mapped_column(JSON, default=list)


class SignalRow(PaBase):
    """Trade-intent ledger row — written when Stage 2 produced an order plan."""

    __tablename__ = "signal"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    timeframe: Mapped[str] = mapped_column(String(16), index=True)
    candle_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)
    order_type: Mapped[str] = mapped_column(String(32))          # 限价单/突破单/市价单
    order_direction: Mapped[str] = mapped_column(String(32))     # 做多/做空
    entry_price: Mapped[Optional[float]] = mapped_column(Float)
    stop_loss_price: Mapped[Optional[float]] = mapped_column(Float)
    take_profit_price: Mapped[Optional[float]] = mapped_column(Float)
    take_profit_price_2: Mapped[Optional[float]] = mapped_column(Float)
    risk_reward: Mapped[Optional[float]] = mapped_column(Float)  # reward:risk ratio
    trade_confidence: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_win_rate: Mapped[Optional[int]] = mapped_column(Integer)
    cycle_position: Mapped[Optional[str]] = mapped_column(String(64))
    direction: Mapped[Optional[str]] = mapped_column(String(32))  # stage-1 direction
    # Diagnosis digest + full decision dict for replay/diagnostics.
    snapshot: Mapped[Optional[dict]] = mapped_column(JSON)
    record_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("analysis_record.id"), index=True
    )
    # Backfilled by the reconciliation job in trade mode (freqtrade trades.id).
    executed_trade_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)


def _ms_to_utc_dt(ms: float) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _k1_candle_time(record: AnalysisRecord) -> datetime | None:
    """K1 (newest closed bar) open time from kline_data[0], newest-first."""
    if not record.kline_data:
        return None
    try:
        raw = record.kline_data[0].get("ts_open")
        if raw is None:
            return None
        return _ms_to_utc_dt(ts_open_to_ms(float(raw)))
    except (TypeError, ValueError):
        return None


def _sanitize(data: Any, api_key: str) -> Any:
    """Recursively mask any occurrence of *api_key* in string values."""
    if not api_key:
        return data
    masked = mask_secret(api_key)

    def _walk(node: Any) -> Any:
        if isinstance(node, str):
            return node.replace(api_key, masked)
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(data)


def _is_order_plan(decision: dict | None) -> bool:
    if not decision:
        return False
    ot = str(decision.get("order_type") or "").strip()
    return ot not in ("", "不下单", "none", "null")


def _risk_reward(decision: dict) -> float | None:
    from pa_core.util.trade_metrics import compute_risk_reward

    rr = compute_risk_reward(
        decision.get("entry_price"),
        decision.get("take_profit_price"),
        decision.get("stop_loss_price"),
        decision.get("order_direction"),
    )
    if rr is None:
        return None
    return float(rr["ratio"])


def _opt_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


class PgRecordStore:
    """PG-backed replacement for PendingWriter + analysis_history lookups.

    Thread model: one engine, short-lived sessions per call — safe to share
    between the strategy thread (writes) and the telegram plugin (reads).
    """

    def __init__(self, db_url: str, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._engine = create_engine(db_url, future=True)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        PaBase.metadata.create_all(self._engine)

    # ── Writes (PendingWriter-compatible semantics) ───────────────────────────

    def save_full(self, record: AnalysisRecord, api_key: str = "") -> int | None:
        """Persist a complete analysis run; returns the row id (None on failure)."""
        return self._save(record, is_partial=False, reason=None, api_key=api_key)

    def save_partial(self, record: AnalysisRecord, reason: str, api_key: str = "") -> int | None:
        """Persist a partial/failed run with a reason."""
        return self._save(record, is_partial=True, reason=reason, api_key=api_key)

    def _save(
        self,
        record: AnalysisRecord,
        *,
        is_partial: bool,
        reason: str | None,
        api_key: str,
    ) -> int | None:
        try:
            data = _sanitize(record.model_dump(), api_key)
            stage1 = record.stage1_diagnosis or {}
            success = (
                record.exception is None
                and bool(record.stage1_diagnosis)
                and bool(record.stage2_decision)
                and bool(record.kline_data)
            )
            with self._session_factory() as session, session.begin():
                row = AnalysisRecordRow(
                    symbol=record.meta.symbol,
                    timeframe=record.meta.timeframe,
                    candle_time=_k1_candle_time(record),
                    success=success,
                    is_partial=is_partial,
                    partial_reason=reason,
                    stage1_summary={
                        "cycle_position": stage1.get("cycle_position"),
                        "direction": stage1.get("direction"),
                        "diagnosis_confidence": stage1.get("diagnosis_confidence"),
                    }
                    if stage1
                    else None,
                    record_json=data,
                    followups=[],
                )
                session.add(row)
                session.flush()

                decision = (record.stage2_decision or {}).get("decision")
                if _is_order_plan(decision):
                    session.add(self._build_signal(row, record, decision))
                return row.id
        except Exception as exc:  # noqa: BLE001
            self._logger.error("PgRecordStore: save failed: %s", exc, exc_info=True)
            return None

    def _build_signal(
        self, row: AnalysisRecordRow, record: AnalysisRecord, decision: dict
    ) -> SignalRow:
        stage1 = record.stage1_diagnosis or {}
        s2 = record.stage2_decision or {}
        return SignalRow(
            symbol=record.meta.symbol,
            timeframe=record.meta.timeframe,
            candle_time=row.candle_time,
            order_type=str(decision.get("order_type") or ""),
            order_direction=str(decision.get("order_direction") or ""),
            entry_price=_opt_float(decision.get("entry_price")),
            stop_loss_price=_opt_float(decision.get("stop_loss_price")),
            take_profit_price=_opt_float(decision.get("take_profit_price")),
            take_profit_price_2=_opt_float(decision.get("take_profit_price_2")),
            risk_reward=_risk_reward(decision),
            trade_confidence=_opt_int(s2.get("trade_confidence")),
            estimated_win_rate=_opt_int(decision.get("estimated_win_rate")),
            cycle_position=stage1.get("cycle_position"),
            direction=stage1.get("direction"),
            snapshot={
                "stage1_digest": {
                    "cycle_position": stage1.get("cycle_position"),
                    "direction": stage1.get("direction"),
                    "detected_patterns": stage1.get("detected_patterns", [])[:10],
                    "support_levels": stage1.get("support_levels", [])[:10],
                    "resistance_levels": stage1.get("resistance_levels", [])[:10],
                },
                "decision": decision,
                "terminal": s2.get("terminal"),
            },
            record_id=row.id,
        )

    def append_followup(self, record_row_id: int, turn: FollowupTurn) -> None:
        """Append a followup turn to the stored record (best effort)."""
        try:
            with self._session_factory() as session, session.begin():
                row = session.get(AnalysisRecordRow, record_row_id)
                if row is None:
                    return
                followups = list(row.followups or [])
                followups.append(turn.model_dump())
                row.followups = followups
        except Exception as exc:  # noqa: BLE001
            self._logger.error("PgRecordStore: append_followup failed: %s", exc)

    # ── Reads (analysis_history-compatible) ───────────────────────────────────

    def find_latest_successful(
        self, *, symbol: str, timeframe: str
    ) -> AnalysisRecord | None:
        """Newest full successful record for incremental analysis restore."""
        try:
            with self._session_factory() as session:
                stmt = (
                    select(AnalysisRecordRow)
                    .where(
                        AnalysisRecordRow.symbol == symbol,
                        AnalysisRecordRow.timeframe == timeframe,
                        AnalysisRecordRow.success.is_(True),
                        AnalysisRecordRow.is_partial.is_(False),
                    )
                    .order_by(AnalysisRecordRow.created_at.desc())
                    .limit(1)
                )
                row = session.execute(stmt).scalar_one_or_none()
                if row is None:
                    return None
                return AnalysisRecord.model_validate(row.record_json)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("PgRecordStore: find_latest_successful failed: %s", exc)
            return None

    def query_signals(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Latest trade-intent rows, newest first (telegram plugin reads)."""
        try:
            with self._session_factory() as session:
                stmt = select(SignalRow).order_by(SignalRow.created_at.desc()).limit(limit)
                if symbol:
                    stmt = stmt.where(SignalRow.symbol == symbol)
                if timeframe:
                    stmt = stmt.where(SignalRow.timeframe == timeframe)
                rows = session.execute(stmt).scalars().all()
                return [
                    {
                        "id": r.id,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                        "symbol": r.symbol,
                        "timeframe": r.timeframe,
                        "candle_time": r.candle_time.isoformat() if r.candle_time else None,
                        "order_type": r.order_type,
                        "order_direction": r.order_direction,
                        "entry_price": r.entry_price,
                        "stop_loss_price": r.stop_loss_price,
                        "take_profit_price": r.take_profit_price,
                        "take_profit_price_2": r.take_profit_price_2,
                        "risk_reward": r.risk_reward,
                        "trade_confidence": r.trade_confidence,
                        "estimated_win_rate": r.estimated_win_rate,
                        "cycle_position": r.cycle_position,
                        "direction": r.direction,
                        "record_id": r.record_id,
                        "executed_trade_id": r.executed_trade_id,
                    }
                    for r in rows
                ]
        except Exception as exc:  # noqa: BLE001
            self._logger.error("PgRecordStore: query_signals failed: %s", exc)
            return []
