from __future__ import annotations

import sys

import pytest

from app.services.capability_probe import Lever, LeverStatus, probe_hags

HAGS_KEY_PATH = r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers"


class _FakeKey:
    def __enter__(self) -> "_FakeKey":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_probe_hags_not_applicable_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    lever = probe_hags()
    assert lever.status == LeverStatus.not_applicable
    assert lever.fixable is False


def test_probe_hags_ok_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    import app.services.capability_probe as mod

    monkeypatch.setattr(mod.winreg, "OpenKey", lambda *a, **k: _FakeKey())
    monkeypatch.setattr(mod.winreg, "QueryValueEx", lambda key, name: (2, 4))

    lever = probe_hags()

    assert lever.status == LeverStatus.ok
    assert lever.fixable is False


def test_probe_hags_unavailable_and_fixable_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    import app.services.capability_probe as mod

    monkeypatch.setattr(mod.winreg, "OpenKey", lambda *a, **k: _FakeKey())
    monkeypatch.setattr(mod.winreg, "QueryValueEx", lambda key, name: (1, 4))

    lever = probe_hags()

    assert lever.status == LeverStatus.unavailable
    assert lever.fixable is True


def test_probe_hags_unavailable_when_registry_value_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    import app.services.capability_probe as mod

    def _raise_open_key(*a: object, **k: object) -> None:
        raise OSError("not found")

    monkeypatch.setattr(mod.winreg, "OpenKey", _raise_open_key)

    lever = probe_hags()

    assert lever.status == LeverStatus.unavailable
    assert lever.fixable is False
