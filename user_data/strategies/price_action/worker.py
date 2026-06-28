"""Background worker for non-blocking PA LLM analysis."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import queue
import threading
from typing import Any, Callable

from pandas import DataFrame

from .features import latest_closed_candle_time
from .orchestrator import PriceActionOrchestrator


logger = logging.getLogger(__name__)


@dataclass
class AnalysisJob:
    symbol: str
    dataframe: DataFrame
    timeframe: str
    market: str
    notifier: Any | None
    chart_generator: Callable | None


class PaAnalysisWorker:
    """Small queue with candle-key deduplication."""

    def __init__(
        self,
        orchestrator: PriceActionOrchestrator,
        *,
        max_workers: int = 1,
        max_seen: int = 2000,
    ) -> None:
        self.orchestrator = orchestrator
        self.max_workers = max(1, int(max_workers))
        self._queue: queue.Queue[AnalysisJob | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._seen: set[tuple[str, str, object]] = set()
        self._seen_order: deque[tuple[str, str, object]] = deque()
        self._max_seen = max_seen
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        for idx in range(self.max_workers):
            thread = threading.Thread(target=self._run, daemon=True, name=f"pa-analysis-{idx}")
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()

    def submit(
        self,
        *,
        symbol: str,
        dataframe: DataFrame,
        timeframe: str,
        market: str,
        notifier: Any | None = None,
        chart_generator: Callable | None = None,
    ) -> bool:
        """Enqueue a job if this closed candle has not already been submitted."""
        candle_time = latest_closed_candle_time(dataframe, timeframe, market=market)
        if candle_time is None:
            return False
        key = (symbol, timeframe, candle_time)
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            self._seen_order.append(key)
            while len(self._seen_order) > self._max_seen:
                old = self._seen_order.popleft()
                self._seen.discard(old)
        self._queue.put(
            AnalysisJob(
                symbol=symbol,
                dataframe=dataframe.copy(deep=True),
                timeframe=timeframe,
                market=market,
                notifier=notifier,
                chart_generator=chart_generator,
            )
        )
        return True

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                self.orchestrator.analyze(
                    symbol=job.symbol,
                    dataframe=job.dataframe,
                    timeframe=job.timeframe,
                    market=job.market,
                    notifier=job.notifier,
                    chart_generator=job.chart_generator,
                )
            except Exception:
                logger.warning("Unhandled PA analysis worker error for %s", job.symbol, exc_info=True)
            finally:
                self._queue.task_done()
