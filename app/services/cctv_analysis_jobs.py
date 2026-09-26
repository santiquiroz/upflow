"""Analisis CCTV sincronico o en segundo plano (spec §5.5 `analyze`: 200 o 202).

Todo analisis corre como tarea propia. La ruta espera `sync_seconds`: si termina,
responde 200 con el resultado; si no (clips largos: el indice decodifica el video
entero), responde 202 con `analysisJobId` y la UI consulta el estado.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from app.services.cctv_analysis import ANALYSIS_FAILED, CctvAnalysisError

SYNC_WAIT_SECONDS = 15.0
ANALYSIS_CONCURRENCY = 1
MAX_FINISHED_ANALYSES = 64

AnalysisStatus = Literal["running", "completed", "failed"]
AnalysisWork = Callable[[], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    id: str
    owner_id: str | None
    status: AnalysisStatus
    result: dict[str, Any] | None = None
    error: CctvAnalysisError | None = None


@dataclass(slots=True)
class _Entry:
    id: str
    owner_id: str | None
    task: asyncio.Task[None] | None = None
    result: dict[str, Any] | None = None
    error: CctvAnalysisError | None = None
    order: int = 0

    @property
    def done(self) -> bool:
        return self.result is not None or self.error is not None


def unexpected_failure(exc: Exception) -> CctvAnalysisError:
    return CctvAnalysisError(ANALYSIS_FAILED, f"The CCTV analysis failed: {exc}", 500)


def snapshot_of(entry: _Entry) -> AnalysisSnapshot:
    if entry.error is not None:
        return AnalysisSnapshot(entry.id, entry.owner_id, "failed", error=entry.error)
    if entry.result is not None:
        return AnalysisSnapshot(entry.id, entry.owner_id, "completed", result=entry.result)
    return AnalysisSnapshot(entry.id, entry.owner_id, "running")


@dataclass
class CctvAnalysisJobs:
    sync_seconds: float = SYNC_WAIT_SECONDS
    concurrency: int = ANALYSIS_CONCURRENCY
    max_finished: int = MAX_FINISHED_ANALYSES
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _slots: asyncio.Semaphore | None = None
    _counter: int = 0

    def _semaphore(self) -> asyncio.Semaphore:
        # Se crea en el loop que la usa: los analisis son CPU pesado y van de a `concurrency`.
        if self._slots is None:
            self._slots = asyncio.Semaphore(self.concurrency)
        return self._slots

    async def _run(self, entry: _Entry, work: AnalysisWork) -> None:
        async with self._semaphore():
            try:
                entry.result = await work()
            except CctvAnalysisError as exc:
                entry.error = exc
            except Exception as exc:
                entry.error = unexpected_failure(exc)

    def _new_entry(self, owner_id: str | None) -> _Entry:
        self._counter += 1
        entry = _Entry(id=uuid4().hex, owner_id=owner_id, order=self._counter)
        self._entries[entry.id] = entry
        return entry

    def _prune(self) -> None:
        finished = sorted((e for e in self._entries.values() if e.done), key=lambda e: e.order)
        for entry in finished[: max(0, len(finished) - self.max_finished)]:
            del self._entries[entry.id]

    async def submit(self, work: AnalysisWork, owner_id: str | None = None) -> AnalysisSnapshot:
        self._prune()
        entry = self._new_entry(owner_id)
        entry.task = asyncio.create_task(self._run(entry, work))
        await asyncio.wait({entry.task}, timeout=self.sync_seconds)
        return snapshot_of(entry)

    def get(self, analysis_id: str) -> AnalysisSnapshot | None:
        entry = self._entries.get(analysis_id)
        return None if entry is None else snapshot_of(entry)

    async def close(self) -> None:
        running = [entry.task for entry in self._entries.values() if entry.task is not None and not entry.task.done()]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
