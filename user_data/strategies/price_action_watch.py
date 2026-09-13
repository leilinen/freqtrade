"""PriceActionWatch — freqtrade strategy wrapping the pa_core two-stage LLM pipeline.

Layer 7 of the PA migration (docs/pa-migration-plan.md). The strategy stays
thin on purpose: freqtrade owns candles/whitelist/telegram; pa_core owns the
price-action decision. Data flow per new closed candle (live/dry-run):

    freqtrade df ──df_to_kline_frame──▶ KlineFrame
                ──two_stage.submit()──▶ AnalysisRecord (persisted to PG by
                                         PgRecordStore injected as pending_writer)
                ──decision──▶ signal table row (order plans only) +
                              dp.send_msg text + enter_long/short when live

Config keys (user_data/watch_config.json):

    "pa_db_url": "postgresql+psycopg://..."   # PG store; default sqlite fallback
    "pa_llm": {model, base_url, api_key, thinking, reasoning_effort,
               analysis_bar_count, decision_stance, enable_next_bar_prediction,
               structure_flip_cooldown_bars}

Backtest note: each candle triggers a real LLM call (by design — LLM is the
strategy). Expect long runtimes; incremental mode keeps per-call tokens low.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from pandas import DataFrame

from freqtrade.strategy import IStrategy

from pa_core.config import EXPERIENCE_DIR, PROMPT_DIR
from pa_core.data_structures import KlineFrame
from pa_core.llm.deepseek_client import DeepSeekClient
from pa_core.llm.prompt_assembler import PromptAssembler
from pa_core.llm.router import route_strategy_files
from pa_core.llm.two_stage import TwoStageOrchestrator
from pa_core.records.pg_store import PgRecordStore
from pa_core.records.schema import AnalysisRecord
from pa_core.settings import AIProviderSettings, Settings
from pa_core.snapshot import INDICATOR_WARMUP_BARS
from pa_core.util.df_adapter import df_to_kline_frame
from pa_core.util.threading import CancelToken

logger = logging.getLogger(__name__)

_DEFAULT_ANALYSIS_BARS = 100


def _fallback_db_url(config: dict) -> str:
    """Local sqlite fallback, anchored at user_data_dir (cwd-independent)."""
    user_data = config.get("user_data_dir") or "user_data"
    return f"sqlite:///{user_data}/pa_records.sqlite"


def _build_settings(config: dict) -> Settings:
    """Map freqtrade config (pa_llm block) onto pa_core Settings."""
    pa = dict(config.get("pa_llm") or {})
    provider = AIProviderSettings(
        model=pa.get("model", "deepseek-v4-flash"),
        base_url=pa.get("base_url", "https://api.deepseek.com"),
        api_key=pa.get("api_key", ""),
        thinking=bool(pa.get("thinking", True)),
        reasoning_effort=pa.get("reasoning_effort", "high"),
    )
    settings = Settings(provider=provider)
    g = settings.general
    g.analysis_bar_count = int(pa.get("analysis_bar_count", _DEFAULT_ANALYSIS_BARS))
    g.decision_stance = pa.get("decision_stance", g.decision_stance)
    g.enable_next_bar_prediction = bool(
        pa.get("enable_next_bar_prediction", g.enable_next_bar_prediction)
    )
    g.structure_flip_cooldown_bars = int(
        pa.get("structure_flip_cooldown_bars", g.structure_flip_cooldown_bars)
    )
    settings.prompt.stage1_inject_pattern_briefs = bool(
        pa.get("stage1_inject_pattern_briefs", settings.prompt.stage1_inject_pattern_briefs)
    )
    return settings


def _is_order_plan(decision: dict | None) -> bool:
    if not decision:
        return False
    ot = str(decision.get("order_type") or "").strip()
    return ot not in ("", "不下单", "none", "null")


def _direction_sign(decision: dict) -> int:
    d = str(decision.get("order_direction") or "")
    if "多" in d:
        return 1
    if "空" in d:
        return -1
    return 0


def _signal_text(pair: str, timeframe: str, record: AnalysisRecord) -> str:
    """Telegram text for an order-plan decision (dp.send_msg, always_send=True)."""
    decision = (record.stage2_decision or {}).get("decision") or {}
    s1 = record.stage1_diagnosis or {}
    s2 = record.stage2_decision or {}
    lines = [
        f"📊 {pair} {timeframe} PA信号",
        f"周期: {s1.get('cycle_position', '—')} | 方向: {s1.get('direction', '—')}",
        f"决策: {decision.get('order_direction', '—')} "
        f"{decision.get('order_type', '—')} @ {decision.get('entry_price', '—')}",
        f"止损: {decision.get('stop_loss_price', '—')} | "
        f"TP1: {decision.get('take_profit_price', '—')} | "
        f"TP2: {decision.get('take_profit_price_2', '—')}",
        f"置信度: {s2.get('trade_confidence', '—')} | "
        f"预估胜率: {decision.get('estimated_win_rate', '—')}%",
    ]
    return "\n".join(lines)


class PriceActionWatch(IStrategy):
    """Watch strategy: two-stage LLM price-action analysis, zero orders by default."""

    INTERFACE_VERSION = 3
    timeframe = "1h"

    # Watch-mode guard (zero trades). Not a hyperopt parameter on purpose:
    # freqtrade's Parameter machinery requires a hyperopt space, and this is a
    # mode switch — config-driven instead ("pa_llm": {"watch_only": false}).
    watch_only: bool = True
    minimal_roi: dict = {}
    stoploss = -0.99
    use_exit_signal = False
    process_only_new_candles = True
    # analysis window + indicator warmup + safety margin
    startup_candle_count = _DEFAULT_ANALYSIS_BARS + INDICATOR_WARMUP_BARS + 10

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._settings: Settings | None = None
        self._orchestrator: TwoStageOrchestrator | None = None
        self._store: PgRecordStore | None = None
        # pair -> latest successful AnalysisRecord (incremental analysis state)
        self._prev_records: dict[str, AnalysisRecord] = {}
        self._analysis_bars = _DEFAULT_ANALYSIS_BARS

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def bot_start(self, **kwargs) -> None:
        self._settings = _build_settings(self.config)
        self.watch_only = bool(
            (self.config.get("pa_llm") or {}).get("watch_only", True)
        )
        self._analysis_bars = self._settings.general.analysis_bar_count
        self.startup_candle_count = self._analysis_bars + INDICATOR_WARMUP_BARS + 10

        fallback_url = _fallback_db_url(self.config)
        db_url = self.config.get("pa_db_url") or fallback_url
        try:
            self._store = PgRecordStore(db_url)
        except Exception as exc:  # noqa: BLE001
            # PG unavailable at startup must not kill the bot — degrade to
            # the local sqlite fallback (signals still land, just not in PG).
            logger.warning(
                "PgRecordStore(%s) unavailable (%s); falling back to %s",
                db_url,
                exc,
                fallback_url,
            )
            db_url = fallback_url
            self._store = PgRecordStore(db_url)

        client = DeepSeekClient(self._settings.provider)
        from pa_core.records.experience_reader import ExperienceReader

        exp_reader = ExperienceReader(experience_dir=EXPERIENCE_DIR)
        assembler = PromptAssembler(
            prompt_dir=PROMPT_DIR,
            experience_reader=exp_reader,
            prompt_settings=self._settings.prompt,
        )
        from pa_core.validation.json_validator import JsonValidator

        validator = JsonValidator(self._settings)
        self._orchestrator = TwoStageOrchestrator(
            client=client,
            assembler=assembler,
            router=route_strategy_files,
            validator=validator,
            pending_writer=self._store,  # PgRecordStore: PendingWriter-compatible
            exp_reader=exp_reader,
            settings=self._settings,
        )

        # Restore incremental state from PG for every whitelisted pair.
        for pair in self.dp.current_whitelist():
            record = self._store.find_latest_successful(
                symbol=pair, timeframe=self.timeframe
            )
            if record is not None:
                self._prev_records[pair] = record
                logger.info("Restored previous analysis for %s", pair)

        if not self._settings.provider.api_key.strip():
            logger.warning(
                "pa_llm.api_key is empty — every PA analysis will fail "
                "(crash partials are still recorded). Local OpenAI-compatible "
                "endpoints without keys may ignore this."
            )

        mode = "watch" if self.watch_only else "TRADE"
        logger.info(
            "PriceActionWatch started (%s mode, %s bars, %s pairs, db=%s)",
            mode,
            self._analysis_bars,
            len(self._prev_records),
            db_url.split("://")[0],
        )

    # ── freqtrade callbacks ────────────────────────────────────────────────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Passthrough — PA features are computed inside the prompt pipeline."""
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        runmode = self.dp.runmode

        # Ensure entry columns exist even when no signal ever fires
        # (a candle without an order plan writes nothing — backtesting
        # expects the columns to be present).
        if "enter_long" not in dataframe.columns:
            dataframe["enter_long"] = 0

        if runmode.value in ("live", "dry_run"):
            self._analyze_live(dataframe, pair)
        elif runmode.value == "backtest":
            self._analyze_backtest(dataframe, pair)
        # hyperopt/edge/utility modes: leave signals empty.

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # ── PA analysis ────────────────────────────────────────────────────────────

    def _analyze_live(self, dataframe: DataFrame, pair: str) -> None:
        """One incremental two-stage run on the newest closed candle."""
        frame = df_to_kline_frame(dataframe, pair, self.timeframe, self._analysis_bars)
        if frame is None:
            logger.debug("%s: not enough candles for PA analysis yet", pair)
            return
        record = self._run_pa(frame, pair, self._prev_records.get(pair))
        if record is not None:
            decision = (record.stage2_decision or {}).get("decision")
            if record.exception is None:
                self._prev_records[pair] = record
            self._apply_decision(dataframe, pair, record, decision)

    def _analyze_backtest(self, dataframe: DataFrame, pair: str) -> None:
        """Per-candle LLM analysis over history (incremental chain)."""
        prev: AnalysisRecord | None = self._prev_records.get(pair)
        start = self.startup_candle_count
        for i in range(start, len(dataframe)):
            window = dataframe.iloc[: i + 1]
            frame = df_to_kline_frame(window, pair, self.timeframe, self._analysis_bars)
            if frame is None:
                continue
            record = self._run_pa(frame, pair, prev)
            if record is None or record.exception is not None:
                continue
            prev = record
            decision = (record.stage2_decision or {}).get("decision")
            if _is_order_plan(decision):
                self._set_entry(dataframe, i, decision)

    def _run_pa(
        self, frame: KlineFrame, pair: str, prev: AnalysisRecord | None
    ) -> AnalysisRecord | None:
        """Submit to the orchestrator; never raises into populate_*."""
        if self._orchestrator is None:
            return None
        new_bars: int | None = None
        if prev is not None:
            from pa_core.records.analysis_history import count_new_bars_since_record

            new_bars = count_new_bars_since_record(frame, prev)
        try:
            record = self._orchestrator.submit(
                frame,
                CancelToken(),
                lambda event: None,  # no GUI event stream
                previous_record=prev if new_bars is not None else None,
                incremental_new_bar_count=new_bars,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: PA analysis crashed: %s", pair, exc, exc_info=True)
            self._save_crash_partial(frame, pair, exc)
            return None
        if record.exception is not None:
            logger.warning(
                "%s: PA analysis incomplete: %s", pair, record.exception.get("category")
            )
        return record

    def _save_crash_partial(self, frame: KlineFrame, pair: str, exc: Exception) -> None:
        """Persist an out-of-band crash as a partial record (audit trail).

        two_stage persists its own failure paths; this covers exceptions that
        escape submit() entirely (e.g. client construction without credentials).
        """
        if self._store is None:
            return
        from pa_core.records.schema import AnalysisRecord, RecordMeta

        record = AnalysisRecord(
            meta=RecordMeta(
                timestamp_local_iso=datetime.now(timezone.utc).isoformat(),
                timestamp_local_ms=frame.snapshot_ts_local_ms,
                symbol=pair,
                timeframe=frame.timeframe,
                bar_count=len(frame.bars),
                ai_provider={"model": self._settings.provider.model} if self._settings else {},
            ),
            kline_data=[
                {
                    "seq": b.seq, "ts_open": b.ts_open, "open": b.open,
                    "high": b.high, "low": b.low, "close": b.close, "volume": b.volume,
                }
                for b in frame.bars
            ],
            htf_text="",
            stage1_messages=[], stage1_response=None, stage1_diagnosis=None,
            stage2_messages=[], stage2_response=None, stage2_decision=None,
            strategy_files_used=[], experience_loaded=[],
            exception={"category": "strategy_exception", "error": str(exc)[:500]},
            usage_total={},
        )
        self._store.save_partial(record, "strategy_exception")

    def _apply_decision(
        self, dataframe: DataFrame, pair: str, record: AnalysisRecord, decision: dict | None
    ) -> None:
        """Record-level effects in live mode: TG push + entry signal."""
        if not _is_order_plan(decision):
            return
        # signal table row is written by PgRecordStore.save_full (order plans only)
        try:
            self.dp.send_msg(_signal_text(pair, self.timeframe, record), always_send=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: send_msg failed: %s", pair, exc)
        if not self.watch_only:
            self._set_entry(dataframe, len(dataframe) - 1, decision)

    def _set_entry(self, dataframe: DataFrame, row_index: int, decision: dict) -> None:
        sign = _direction_sign(decision)
        if sign > 0:
            dataframe.loc[dataframe.index[row_index], "enter_long"] = 1
        elif sign < 0:
            dataframe.loc[dataframe.index[row_index], "enter_short"] = 1
