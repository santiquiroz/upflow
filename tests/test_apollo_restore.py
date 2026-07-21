from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.services.engines.apollo_restore import ApolloRestorer
from app.services.gpu_session_coordinator import GpuSessionCoordinator

# ---------------------------------------------------------------------------
# ApolloRestorer -- GpuSessionCoordinator wiring (Fase 1 Task 2): same
# contract as AudioSrRestorer (see tests/test_audiosr_restorer.py), just
# keyed by a single cached session per device instead of a dict of graphs.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_release_device_clears_cached_session_for_that_device_only(tmp_path: Path) -> None:
    restorer = ApolloRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    restorer._session_cache["dml:0"] = "fake-session"
    restorer._session_cache["dml:1"] = "fake-session-1"

    restorer.release_device("dml:0")

    assert "dml:0" not in restorer._session_cache
    assert "dml:1" in restorer._session_cache


def test_release_device_on_empty_cache_is_a_noop(tmp_path: Path) -> None:
    restorer = ApolloRestorer(make_settings(tmp_path), GpuSessionCoordinator())

    restorer.release_device("dml:0")  # no debe lanzar


def test_get_session_calls_coordinator_acquire_before_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu_coordinator = GpuSessionCoordinator()
    restorer = ApolloRestorer(make_settings(tmp_path), gpu_coordinator)
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(gpu_coordinator, "acquire", lambda device, owner: calls.append((device, owner)))
    monkeypatch.setattr(restorer, "_create_session", lambda device: "fake-session")

    restorer._get_session("dml:0")

    assert calls == [("dml:0", restorer)]
