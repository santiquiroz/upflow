from __future__ import annotations

import asyncio
import json
import sys

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.services.capability_probe import (
    CapabilityProbe,
    Lever,
    LeverStatus,
    build_disk_write_cache_script,
    build_fix_script,
    parse_defender_exclusion_json,
    parse_disk_write_cache_json,
    parse_pcie_json,
    probe_defender_exclusion,
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


def test_build_disk_write_cache_script_escapes_single_quotes() -> None:
    # Test that single quotes in the path are escaped (doubled) for PowerShell
    runtime_path = "C:/it's/a/path"
    script = build_disk_write_cache_script(runtime_path)

    # Single quotes should be doubled in PowerShell literal strings
    assert "'C:/it''s/a/path'" in script
    assert runtime_path not in script  # Raw path should not appear unescaped


def test_build_disk_write_cache_script_matches_pnp_friendly_name_by_prefix() -> None:
    # Get-Disk's FriendlyName ("AMD-RAID Array 1") is a strict PREFIX of
    # Get-PnpDevice's FriendlyName ("AMD-RAID Array 1  SCSI Disk Device") on
    # real RAID/Storport-backed disks -- confirmed on real hardware. An
    # exact -eq match never fires for that disk class, so the script must
    # use -like with a trailing wildcard, and must guard the no-match case
    # instead of building a regPath from a null PnP device.
    script = build_disk_write_cache_script("C:/Upflow/runtime")

    assert "-like" in script
    assert "-eq $disk.FriendlyName" not in script
    assert "if (-not $pnp)" in script


def test_build_fix_script_disk_write_cache_matches_pnp_friendly_name_by_prefix() -> None:
    script = build_fix_script("disk_write_cache", "C:/Upflow/runtime")

    assert "-like" in script
    assert "-eq $disk.FriendlyName" not in script
    assert "if (-not $pnp)" in script


def test_probe_disk_write_cache_runs_powershell_and_parses_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    import app.services.capability_probe as mod

    invoked_command: list[str] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        invoked_command.extend(command)
        payload = json.dumps({"ok": True, "diskName": "NVMe SSD", "writeCacheEnabled": True}).encode()
        return payload, b"", 0

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    lever = asyncio.run(probe_disk_write_cache("C:/Upflow/runtime"))

    assert lever.status.value == "ok"
    assert invoked_command[0] == "powershell.exe"


def test_probe_disk_write_cache_degrades_on_subprocess_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    import app.services.capability_probe as mod

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"boom", 1

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    lever = asyncio.run(probe_disk_write_cache("C:/Upflow/runtime"))

    assert lever.status.value == "unavailable"


def test_probe_disk_write_cache_catches_subprocess_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    import app.services.capability_probe as mod

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        raise RuntimeError("subprocess explosion")

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    # Should not raise; should degrade gracefully
    lever = asyncio.run(probe_disk_write_cache("C:/Upflow/runtime"))

    assert lever.status.value == "unavailable"
    assert "Could not run" in lever.detail


def test_parse_defender_exclusion_ok_when_path_excluded() -> None:
    raw = json.dumps({"ok": True, "exclusions": ["C:\\Upflow\\runtime", "C:\\Other"]})

    lever = parse_defender_exclusion_json(raw, "C:\\Upflow\\runtime")

    assert lever.status.value == "ok"
    assert lever.fixable is False


def test_parse_defender_exclusion_fixable_when_not_excluded() -> None:
    raw = json.dumps({"ok": True, "exclusions": ["C:\\Other"]})

    lever = parse_defender_exclusion_json(raw, "C:\\Upflow\\runtime")

    assert lever.status.value == "unavailable"
    assert lever.fixable is True


def test_parse_defender_exclusion_needs_admin_when_read_fails() -> None:
    raw = json.dumps({"ok": False, "error": "Access denied"})

    lever = parse_defender_exclusion_json(raw, "C:\\Upflow\\runtime")

    assert lever.status.value == "needs_admin"


def test_parse_defender_exclusion_unavailable_on_valid_but_wrong_shape() -> None:
    # Test JSON that parses to a scalar
    lever = parse_defender_exclusion_json("42", "C:\\Upflow\\runtime")
    assert lever.status.value == "unavailable"

    # Test JSON that parses to a list
    lever = parse_defender_exclusion_json("[1, 2, 3]", "C:\\Upflow\\runtime")
    assert lever.status.value == "unavailable"

    # Test JSON that is a dict but exclusions is not a list
    lever = parse_defender_exclusion_json('{"ok": true, "exclusions": 42}', "C:\\Upflow\\runtime")
    assert lever.status.value == "unavailable"

    # Test JSON that is a dict but exclusions contains non-strings
    lever = parse_defender_exclusion_json('{"ok": true, "exclusions": [1, 2, 3]}', "C:\\Upflow\\runtime")
    assert lever.status.value == "unavailable"


def test_probe_defender_exclusion_not_applicable_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    lever = asyncio.run(probe_defender_exclusion("C:/Upflow/runtime"))

    assert lever.status.value == "not_applicable"


def test_probe_defender_exclusion_runs_powershell_and_parses_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    import app.services.capability_probe as mod

    invoked_command: list[str] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        invoked_command.extend(command)
        payload = json.dumps({"ok": True, "exclusions": ["C:\\Upflow\\runtime"]}).encode()
        return payload, b"", 0

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    lever = asyncio.run(probe_defender_exclusion("C:/Upflow/runtime"))

    assert lever.status.value == "ok"
    assert invoked_command[0] == "powershell.exe"


def test_probe_defender_exclusion_degrades_on_subprocess_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    import app.services.capability_probe as mod

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"boom", 1

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    lever = asyncio.run(probe_defender_exclusion("C:/Upflow/runtime"))

    assert lever.status.value == "needs_admin"


def test_probe_defender_exclusion_catches_subprocess_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    import app.services.capability_probe as mod

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        raise RuntimeError("subprocess explosion")

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    # Should not raise; should degrade gracefully
    lever = asyncio.run(probe_defender_exclusion("C:/Upflow/runtime"))

    assert lever.status.value == "needs_admin"
    assert "Could not run" in lever.detail


def test_build_fix_script_hags_sets_registry_value() -> None:
    script = build_fix_script("hags", "C:/Upflow/runtime")
    assert "HwSchMode" in script
    assert "-Value 2" in script


def test_build_fix_script_defender_adds_exclusion() -> None:
    script = build_fix_script("defender_exclusion", "C:/Upflow/runtime")
    assert "Add-MpPreference" in script
    assert "C:/Upflow/runtime" in script


def test_build_fix_script_disk_write_cache_never_restarts_device() -> None:
    script = build_fix_script("disk_write_cache", "C:/Upflow/runtime")
    assert "Set-ItemProperty" in script
    assert "Restart-Device" not in script
    assert "Disable-PnpDevice" not in script


def test_build_fix_script_rejects_non_fixable_lever() -> None:
    with pytest.raises(ValueError):
        build_fix_script("pcie_link", "C:/Upflow/runtime")


def test_build_fix_script_rejects_unknown_lever() -> None:
    with pytest.raises(ValueError):
        build_fix_script("not_a_real_lever", "C:/Upflow/runtime")


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_capability_probe_list_levers_caches_until_rescan(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = CapabilityProbe(make_settings())

    import app.services.capability_probe as mod

    monkeypatch.setattr(mod, "probe_hags", lambda: Lever("hags", "HAGS", LeverStatus.ok, "d", False))
    monkeypatch.setattr(mod, "probe_pcie_link", _async_stub_lever("pcie_link"))
    monkeypatch.setattr(mod, "probe_disk_write_cache", _async_stub_lever_with_arg("disk_write_cache"))
    monkeypatch.setattr(mod, "probe_defender_exclusion", _async_stub_lever_with_arg("defender_exclusion"))

    first = asyncio.run(probe.list_levers())
    second = asyncio.run(probe.list_levers())

    assert [lever.id for lever in first] == ["hags", "pcie_link", "disk_write_cache", "defender_exclusion"]
    assert first == second


def _async_stub_lever(lever_id: str):
    async def _stub(*args: object, **kwargs: object) -> Lever:
        return Lever(lever_id, lever_id, LeverStatus.ok, "d", False)

    return _stub


def _async_stub_lever_with_arg(lever_id: str):
    async def _stub(runtime_path: str, *args: object, **kwargs: object) -> Lever:
        return Lever(lever_id, lever_id, LeverStatus.ok, "d", False)

    return _stub


def test_capability_probe_rescan_reprobes_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.capability_probe as mod

    probe = CapabilityProbe(make_settings())
    monkeypatch.setattr(mod, "probe_hags", lambda: Lever("hags", "HAGS", LeverStatus.ok, "d", False))
    monkeypatch.setattr(mod, "probe_pcie_link", _async_stub_lever("pcie_link"))
    monkeypatch.setattr(mod, "probe_disk_write_cache", _async_stub_lever_with_arg("disk_write_cache"))
    monkeypatch.setattr(mod, "probe_defender_exclusion", _async_stub_lever_with_arg("defender_exclusion"))
    asyncio.run(probe.list_levers())

    monkeypatch.setattr(mod, "probe_hags", lambda: Lever("hags", "HAGS", LeverStatus.unavailable, "changed", True))
    levers = asyncio.run(probe.rescan())

    assert next(l for l in levers if l.id == "hags").detail == "changed"


def test_capability_probe_apply_fix_reprobes_after_elevation(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.capability_probe as mod

    probe = CapabilityProbe(make_settings())

    async def fake_run_guarded_process(command: list[str], timeout: float):
        return b"", b"", 0

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)
    monkeypatch.setattr(mod, "probe_hags", lambda: Lever("hags", "HAGS", LeverStatus.ok, "now enabled", False))

    lever = asyncio.run(probe.apply_fix("hags"))

    assert lever.status == LeverStatus.ok
    assert lever.detail == "now enabled"


def test_capability_probe_apply_fix_reports_elevation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.capability_probe as mod

    probe = CapabilityProbe(make_settings())

    async def fake_run_guarded_process(command: list[str], timeout: float):
        return b"", b"", 1

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)
    monkeypatch.setattr(
        mod, "probe_hags", lambda: Lever("hags", "HAGS", LeverStatus.unavailable, "still disabled", True)
    )

    lever = asyncio.run(probe.apply_fix("hags"))

    assert lever.status == LeverStatus.unavailable
    assert "cancelled" in lever.detail.lower() or "failed" in lever.detail.lower()


def test_run_elevated_tracks_and_kills_process_instead_of_bare_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.capability_probe as mod

    captured_command: list[str] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        captured_command.extend(command)
        return b"", b"", 0

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    ok, message = asyncio.run(mod._run_elevated("Set-ItemProperty -Path x -Name y -Value 1", 30.0))

    outer_script = captured_command[-1]
    assert "WaitForExit" in outer_script
    assert "Stop-Process" in outer_script
    assert "-Wait" not in outer_script
    assert ok is True
    assert message == ""


def test_elevation_wait_milliseconds_applies_margin_below_outer_timeout() -> None:
    import app.services.capability_probe as mod

    assert mod._elevation_wait_milliseconds(30.0) == 25_000


def test_elevation_wait_milliseconds_has_floor_for_small_timeout() -> None:
    import app.services.capability_probe as mod

    assert mod._elevation_wait_milliseconds(3.0) == 1_000


def test_capability_fix_timeout_rejects_values_below_minimum() -> None:
    with pytest.raises(ValidationError):
        make_settings(CAPABILITY_FIX_TIMEOUT_SECONDS=0.5)
