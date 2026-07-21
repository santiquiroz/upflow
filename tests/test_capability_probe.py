from __future__ import annotations

import asyncio
import json
import sys

import pytest

from app.services.capability_probe import (
    Lever,
    LeverStatus,
    parse_disk_write_cache_json,
    parse_pcie_json,
    probe_disk_write_cache,
    probe_hags,
    probe_pcie_link,
)

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


def test_parse_pcie_json_reports_downgrade() -> None:
    raw = json.dumps([{"name": "AMD Radeon RX 7800 XT", "linkSpeed": 3, "linkWidth": 8}])

    lever = parse_pcie_json(raw)

    assert lever.status.value == "unavailable"
    assert "Gen3" in lever.detail
    assert "x8" in lever.detail
    assert lever.fixable is False


def test_parse_pcie_json_ok_at_full_link() -> None:
    raw = json.dumps([{"name": "AMD Radeon RX 7800 XT", "linkSpeed": 4, "linkWidth": 16}])

    lever = parse_pcie_json(raw)

    assert lever.status.value == "ok"


def test_parse_pcie_json_unavailable_on_empty_list() -> None:
    lever = parse_pcie_json("[]")

    assert lever.status.value == "unavailable"
    assert "No GPU" in lever.detail


def test_parse_pcie_json_unavailable_on_garbage() -> None:
    lever = parse_pcie_json("not json")

    assert lever.status.value == "unavailable"


def test_parse_pcie_json_unavailable_on_valid_but_wrong_shape() -> None:
    # Test JSON that parses but isn't a list of dicts
    lever = parse_pcie_json("[1, 2, 3]")
    assert lever.status.value == "unavailable"
    assert "malformed" in lever.detail.lower()

    # Test JSON that parses to a scalar
    lever = parse_pcie_json("42")
    assert lever.status.value == "unavailable"


def test_probe_pcie_link_not_applicable_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    lever = asyncio.run(probe_pcie_link())

    assert lever.status.value == "not_applicable"


def test_probe_pcie_link_runs_powershell_and_parses_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    import app.services.capability_probe as mod

    invoked_command: list[str] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        invoked_command.extend(command)
        payload = json.dumps([{"name": "GPU", "linkSpeed": 4, "linkWidth": 16}]).encode()
        return payload, b"", 0

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    lever = asyncio.run(probe_pcie_link())

    assert lever.status.value == "ok"
    assert invoked_command[0] == "powershell.exe"


def test_probe_pcie_link_degrades_on_subprocess_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    import app.services.capability_probe as mod

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"boom", 1

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    lever = asyncio.run(probe_pcie_link())

    assert lever.status.value == "unavailable"


def test_parse_disk_write_cache_ok_when_enabled() -> None:
    raw = json.dumps({"ok": True, "diskName": "NVMe SSD", "writeCacheEnabled": True})

    lever = parse_disk_write_cache_json(raw)

    assert lever.status.value == "ok"
    assert lever.fixable is False


def test_parse_disk_write_cache_fixable_when_disabled() -> None:
    raw = json.dumps({"ok": True, "diskName": "NVMe SSD", "writeCacheEnabled": False})

    lever = parse_disk_write_cache_json(raw)

    assert lever.status.value == "unavailable"
    assert lever.fixable is True


def test_parse_disk_write_cache_degrades_on_script_error() -> None:
    raw = json.dumps({"ok": False, "error": "Access denied"})

    lever = parse_disk_write_cache_json(raw)

    assert lever.status.value == "unavailable"
    assert "Access denied" in lever.detail
    assert lever.fixable is False


def test_parse_disk_write_cache_unavailable_on_valid_but_wrong_shape() -> None:
    # Test JSON that parses to a scalar
    lever = parse_disk_write_cache_json("42")
    assert lever.status.value == "unavailable"

    # Test JSON that parses to a list
    lever = parse_disk_write_cache_json("[1, 2, 3]")
    assert lever.status.value == "unavailable"

    # Test JSON that is a dict but missing required keys
    lever = parse_disk_write_cache_json("{}")
    assert lever.status.value == "unavailable"


def test_probe_disk_write_cache_not_applicable_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    lever = asyncio.run(probe_disk_write_cache("C:/Upflow/runtime"))

    assert lever.status.value == "not_applicable"
