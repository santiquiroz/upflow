# Optimization Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect OS/driver-level optimization levers (HAGS, PCIe link, disk write-cache, Windows Defender exclusion) and ONNX CPU-EP-fallback ops, expose them in a new "Optimization Center" panel in Settings with one-click admin fixes where safe, and retrofit IOBinding into `ApolloRestorer`.

**Architecture:** Two new backend services following the codebase's existing "pure detection function + safe fallback" pattern (`backend_registry.py`, `video_encoders.py`) and "never-raise, cache, degrade" pattern (`update_service.py`): `CapabilityProbe` (OS/driver levers) and `OnnxCpuFallbackProbe` (ONNX Runtime profiling-based diagnostic). Both are read-only by default; `CapabilityProbe` gains an elevation path for the 3 levers that are safely automatable via `Start-Process -Verb RunAs`. A new frontend panel under `modules/settings/` renders both, following the existing `SettingsPage`/`DeviceDefault` query pattern and the `useModels.ts` mutation pattern for the Fix action.

**Tech Stack:** Python stdlib `winreg` (HAGS read), PowerShell via the existing `app/services/process_runner.run_guarded_process` (PCIe link, disk write-cache, Defender exclusion, elevation), ONNX Runtime `session_options.enable_profiling` (CPU-EP-fallback diagnostic), FastAPI + Pydantic (API), React + TanStack Query (UI).

## Global Constraints

- Every OS-level probe is Windows-only; on any other `sys.platform`, it MUST return `LeverStatus.not_applicable` and never raise — same guard as `app/services/devices_service.py:71` (`if sys.platform != "win32": return []`).
- No probe or fix may ever raise out of its public entry point. A failure (permissions, missing registry key, PowerShell error, timeout) degrades to a `Lever` with an honest `status` and `detail` string — same principle as `app/services/update_service.py`'s never-raise `check()`.
- The disk write-cache fix writes the registry value ONLY. It must NEVER call `Disable-PnpDevice`/`Restart-Device`/anything that live-cycles a disk device — the target disk can be the boot volume, and a live device reset on it is a real crash/hang risk. The change takes effect on next reboot, exactly like HAGS — this is documented in the lever's `detail` text, not silently assumed.
- Elevated actions run via `Start-Process powershell.exe -Verb RunAs -Wait` from a NON-elevated backend process. The inner script is passed via `-EncodedCommand` (base64 UTF-16LE) to avoid nested-quoting bugs — never string-concatenate user-controlled or path-controlled text directly into an outer `-Command` string.
- All new response schemas use the existing `Field(serialization_alias="camelCase")` convention (see `app/schemas.py`), snake_case in Python.
- No new Python dependencies (no `wmi`, no `pywin32`) — PowerShell subprocess + stdlib `winreg`, matching `devices_service.py`'s "ctypes/no extra dependency" precedent.
- Commit messages in this repo follow the layered Dominio:/Aplicación:/Infraestructura: convention is NOT used here (that's bipolar-code's convention) — Upflow commits are plain `type: description` (see `git log` in this repo). Do not use the Spanish layered format from a different repo's CLAUDE.md.

---

## File Structure

**Backend — new:**
- `app/services/capability_probe.py` — `Lever`, `LeverStatus`, `CapabilityProbe` (4 OS-level probes + elevation).
- `app/services/onnx_cpu_fallback_probe.py` — `CpuFallbackReport`, synthetic-input builder, profiling-JSON parser, `OnnxCpuFallbackProbe`.
- `app/api/capability_routes.py` — new router, kept separate from `routes.py` (already near its 800-line follow-up limit).
- `tests/test_capability_probe.py`, `tests/test_onnx_cpu_fallback_probe.py`, `tests/test_capability_routes.py`.

**Backend — modified:**
- `app/config.py` — one new field: `capability_fix_timeout_seconds`.
- `app/main.py` — construct `CapabilityProbe` and `OnnxCpuFallbackProbe` in `lifespan`, mount `capability_routes.router`.
- `app/schemas.py` — `LeverResponse`, `CapabilitiesResponse`, `FixLeverResponse`, `OnnxDiagnosticCatalogEntryResponse`, `OnnxDiagnosticsResponse`, `CpuFallbackReportResponse`.
- `app/services/engines/apollo_restore.py` — IOBinding retrofit (Fase 0.2).
- `tests/test_apollo_restore.py` — IOBinding tests.

**Frontend — new:**
- `frontend/src/hooks/useCapabilities.ts` (+ `.test.tsx`).
- `frontend/src/modules/settings/OptimizationCenter.tsx` (+ `.test.tsx`) — levers + diagnostics table + BIOS checklist.

**Frontend — modified:**
- `frontend/src/lib/apiTypes.ts`, `frontend/src/lib/api.ts` — new types + calls.
- `frontend/src/modules/settings/SettingsPage.tsx` — renders `<OptimizationCenter />`.

**Docs:**
- `README.md`, `CLAUDE.md`, `.env.example` — brief mentions (Task 13).

**Deferred (documented, not built here):** IOBinding retrofit for `AudioSrRestorer` (multi-graph DDIM loop, dynamic shapes per step) and `GmfssEngine` (4 graphs, `ORT_DISABLE_ALL` + subprocess isolation already load-bearing — adding IOBinding there needs its own dedicated benchmark pass, not a drive-by change) and `OnnxUpscaler`'s HF-arbitrary-model path (lower priority than the builtin path already covered by `OnnxVideoUpscaler`). Fase 3 (dual-pipeline iGPU+dGPU) is out of scope per the spec.

---

### Task 1: `CapabilityProbe` core types + HAGS probe

**Files:**
- Create: `app/services/capability_probe.py`
- Test: `tests/test_capability_probe.py`

**Interfaces:**
- Produces: `class LeverStatus(str, Enum)` with members `ok`, `unavailable`, `not_applicable`, `needs_admin`. `@dataclass(frozen=True, slots=True) class Lever` with fields `id: str`, `label: str`, `status: LeverStatus`, `detail: str`, `fixable: bool`. `def probe_hags() -> Lever`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_capability_probe.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_capability_probe.py -v` (repo root is the pytest rootdir; run from repo root: `pytest tests/test_capability_probe.py -v`)
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.capability_probe'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/capability_probe.py
from __future__ import annotations

import sys
import winreg
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Optimization Center: detects OS/driver-level levers the app cannot
# influence at the ONNX/ncnn layer -- HAGS, PCIe link, disk write-cache
# policy, Windows Defender exclusion. Same "detect, activate if supported,
# fall back to informational-only otherwise, never break, never degrade
# silently" principle already used by backend_registry.py (ncnn/onnx) and
# video_encoders.py (software/hw encoder).
# ---------------------------------------------------------------------------

HAGS_KEY_PATH = r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
HAGS_VALUE_NAME = "HwSchMode"
HAGS_ENABLED_VALUE = 2


class LeverStatus(str, Enum):
    ok = "ok"
    unavailable = "unavailable"
    not_applicable = "not_applicable"
    needs_admin = "needs_admin"


@dataclass(frozen=True, slots=True)
class Lever:
    id: str
    label: str
    status: LeverStatus
    detail: str
    fixable: bool


def probe_hags() -> Lever:
    lever_id, label = "hags", "Hardware-accelerated GPU scheduling"
    if sys.platform != "win32":
        return Lever(lever_id, label, LeverStatus.not_applicable, "Windows only", False)
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, HAGS_KEY_PATH) as key:
            value, _ = winreg.QueryValueEx(key, HAGS_VALUE_NAME)
    except OSError:
        return Lever(lever_id, label, LeverStatus.unavailable, "HwSchMode registry value not found", False)
    if value == HAGS_ENABLED_VALUE:
        return Lever(lever_id, label, LeverStatus.ok, "Enabled (HwSchMode=2)", False)
    return Lever(
        lever_id, label, LeverStatus.unavailable,
        "Disabled (HwSchMode=1). Measured impact on this workload is ~0 -- informational only.",
        True,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_capability_probe.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/capability_probe.py tests/test_capability_probe.py
git commit -m "feat: add CapabilityProbe HAGS lever (Optimization Center Fase 1)"
```

---

### Task 2: PCIe link speed/width probe (read-only)

**Files:**
- Modify: `app/services/capability_probe.py`
- Modify: `tests/test_capability_probe.py`

**Interfaces:**
- Consumes: `app.services.process_runner.run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]`.
- Produces: `def parse_pcie_json(raw_stdout: str) -> Lever` (pure), `async def probe_pcie_link(timeout: float = 10.0) -> Lever`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_capability_probe.py
import asyncio
import json

from app.services.capability_probe import parse_pcie_json, probe_pcie_link


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


def test_probe_pcie_link_not_applicable_off_windows(monkeypatch) -> None:
    import sys

    monkeypatch.setattr(sys, "platform", "linux")

    lever = asyncio.run(probe_pcie_link())

    assert lever.status.value == "not_applicable"


def test_probe_pcie_link_runs_powershell_and_parses_result(monkeypatch) -> None:
    import sys

    import app.services.capability_probe as mod

    monkeypatch.setattr(sys, "platform", "win32")

    async def fake_run_guarded_process(command, timeout):
        payload = json.dumps([{"name": "GPU", "linkSpeed": 4, "linkWidth": 16}]).encode()
        return payload, b"", 0

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    lever = asyncio.run(probe_pcie_link())

    assert lever.status.value == "ok"


def test_probe_pcie_link_degrades_on_subprocess_failure(monkeypatch) -> None:
    import sys

    import app.services.capability_probe as mod

    monkeypatch.setattr(sys, "platform", "win32")

    async def fake_run_guarded_process(command, timeout):
        return b"", b"boom", 1

    monkeypatch.setattr(mod, "run_guarded_process", fake_run_guarded_process)

    lever = asyncio.run(probe_pcie_link())

    assert lever.status.value == "unavailable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_capability_probe.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_pcie_json'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to app/services/capability_probe.py, after the imports:
import json
import logging

from app.services.process_runner import run_guarded_process

logger = logging.getLogger(__name__)

# DEVPKEY_PciDevice_CurrentLinkSpeed is a PCI_EXPRESS_LINK_SPEED enum, not a
# raw GT/s value -- Microsoft docs: 1=Gen1, 2=Gen2, 3=Gen3, 4=Gen4, 5=Gen5.
_PCIE_GEN_LABELS = {1: "Gen1", 2: "Gen2", 3: "Gen3", 4: "Gen4", 5: "Gen5"}
_MAX_KNOWN_LINK_WIDTH = 16  # x16 is the widest slot this app's target GPUs use

_PCIE_LINK_SCRIPT = """
$gpus = Get-CimInstance Win32_VideoController | Where-Object { $_.PNPDeviceID -and $_.Name -notlike '*Basic Render*' }
$results = foreach ($gpu in $gpus) {
    $props = Get-PnpDeviceProperty -InstanceId $gpu.PNPDeviceID -KeyName "DEVPKEY_PciDevice_CurrentLinkSpeed","DEVPKEY_PciDevice_CurrentLinkWidth" -ErrorAction SilentlyContinue
    $speed = ($props | Where-Object KeyName -eq "DEVPKEY_PciDevice_CurrentLinkSpeed").Data
    $width = ($props | Where-Object KeyName -eq "DEVPKEY_PciDevice_CurrentLinkWidth").Data
    [PSCustomObject]@{ name = $gpu.Name; linkSpeed = $speed; linkWidth = $width }
}
$results | ConvertTo-Json -Compress
""".strip()


def parse_pcie_json(raw_stdout: str) -> Lever:
    lever_id, label = "pcie_link", "PCIe link speed/width"
    try:
        entries = json.loads(raw_stdout)
    except (json.JSONDecodeError, TypeError):
        return Lever(lever_id, label, LeverStatus.unavailable, "Could not read PCIe link info", False)
    if isinstance(entries, dict):
        entries = [entries]
    if not entries:
        return Lever(lever_id, label, LeverStatus.unavailable, "No GPU adapter found to check", False)
    descriptions = []
    downgraded = False
    for entry in entries:
        gen = _PCIE_GEN_LABELS.get(entry.get("linkSpeed"), "unknown")
        width = entry.get("linkWidth")
        descriptions.append(f"{entry.get('name', 'GPU')}: {gen} x{width}")
        if width is not None and width < _MAX_KNOWN_LINK_WIDTH:
            downgraded = True
    detail = "; ".join(descriptions)
    status = LeverStatus.unavailable if downgraded else LeverStatus.ok
    return Lever(lever_id, label, status, detail, False)


async def probe_pcie_link(timeout: float = 10.0) -> Lever:
    lever_id, label = "pcie_link", "PCIe link speed/width"
    if sys.platform != "win32":
        return Lever(lever_id, label, LeverStatus.not_applicable, "Windows only", False)
    try:
        stdout, stderr, returncode = await run_guarded_process(
            ["powershell.exe", "-NoProfile", "-Command", _PCIE_LINK_SCRIPT], timeout
        )
    except Exception:  # noqa: BLE001 -- a probe must never raise
        logger.warning("PCIe link probe failed to run", exc_info=True)
        return Lever(lever_id, label, LeverStatus.unavailable, "Could not run the PCIe link probe", False)
    if returncode != 0:
        return Lever(lever_id, label, LeverStatus.unavailable, f"PCIe link probe failed: {stderr.decode(errors='replace')[:200]}", False)
    return parse_pcie_json(stdout.decode(errors="replace"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_capability_probe.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/capability_probe.py tests/test_capability_probe.py
git commit -m "feat: add PCIe link speed/width probe to CapabilityProbe"
```

---

### Task 3: Disk write-cache policy probe (read-only, fix deferred to Task 5)

**Files:**
- Modify: `app/services/capability_probe.py`
- Modify: `tests/test_capability_probe.py`

**Interfaces:**
- Produces: `def parse_disk_write_cache_json(raw_stdout: str) -> Lever` (pure), `async def probe_disk_write_cache(runtime_path: str, timeout: float = 10.0) -> Lever`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_capability_probe.py
from app.services.capability_probe import parse_disk_write_cache_json, probe_disk_write_cache


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


def test_probe_disk_write_cache_not_applicable_off_windows(monkeypatch) -> None:
    import sys

    monkeypatch.setattr(sys, "platform", "linux")

    lever = asyncio.run(probe_disk_write_cache("C:/Upflow/runtime"))

    assert lever.status.value == "not_applicable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_capability_probe.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_disk_write_cache_json'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to app/services/capability_probe.py

_DISK_WRITE_CACHE_SCRIPT_TEMPLATE = """
try {{
    $target = {path_literal}
    $drive = (Get-Item -LiteralPath $target -ErrorAction Stop).PSDrive.Name
    $partition = Get-Partition -DriveLetter $drive -ErrorAction Stop
    $disk = Get-Disk -Number $partition.DiskNumber -ErrorAction Stop
    $pnp = Get-PnpDevice -Class DiskDrive -ErrorAction Stop | Where-Object {{ $_.FriendlyName -eq $disk.FriendlyName }} | Select-Object -First 1
    $regPath = "HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\$($pnp.InstanceId)\\Device Parameters\\Disk"
    $cache = (Get-ItemProperty -Path $regPath -Name "UserWriteCacheSetting" -ErrorAction Stop).UserWriteCacheSetting
    [PSCustomObject]@{{ ok = $true; diskName = $disk.FriendlyName; writeCacheEnabled = [bool]$cache }} | ConvertTo-Json -Compress
}} catch {{
    [PSCustomObject]@{{ ok = $false; error = $_.Exception.Message }} | ConvertTo-Json -Compress
}}
""".strip()


def _ps_single_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_disk_write_cache_script(runtime_path: str) -> str:
    return _DISK_WRITE_CACHE_SCRIPT_TEMPLATE.format(path_literal=_ps_single_quote(runtime_path))


def parse_disk_write_cache_json(raw_stdout: str) -> Lever:
    lever_id, label = "disk_write_cache", "Disk write-cache policy"
    try:
        payload = json.loads(raw_stdout)
    except (json.JSONDecodeError, TypeError):
        return Lever(lever_id, label, LeverStatus.unavailable, "Could not read disk write-cache policy", False)
    if not payload.get("ok"):
        return Lever(lever_id, label, LeverStatus.unavailable, str(payload.get("error", "unknown error")), False)
    enabled = bool(payload.get("writeCacheEnabled"))
    disk_name = payload.get("diskName", "disk")
    if enabled:
        return Lever(lever_id, label, LeverStatus.ok, f"Write caching enabled on {disk_name}", False)
    return Lever(
        lever_id, label, LeverStatus.unavailable,
        f"Write caching disabled on {disk_name} ('Quick removal' policy) -- affects save-bound video jobs. Fix requires a reboot to take effect.",
        True,
    )


async def probe_disk_write_cache(runtime_path: str, timeout: float = 10.0) -> Lever:
    lever_id, label = "disk_write_cache", "Disk write-cache policy"
    if sys.platform != "win32":
        return Lever(lever_id, label, LeverStatus.not_applicable, "Windows only", False)
    script = build_disk_write_cache_script(runtime_path)
    try:
        stdout, stderr, returncode = await run_guarded_process(
            ["powershell.exe", "-NoProfile", "-Command", script], timeout
        )
    except Exception:  # noqa: BLE001 -- a probe must never raise
        logger.warning("disk write-cache probe failed to run", exc_info=True)
        return Lever(lever_id, label, LeverStatus.unavailable, "Could not run the disk write-cache probe", False)
    if returncode != 0:
        return Lever(lever_id, label, LeverStatus.unavailable, f"Disk write-cache probe failed: {stderr.decode(errors='replace')[:200]}", False)
    return parse_disk_write_cache_json(stdout.decode(errors="replace"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_capability_probe.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/capability_probe.py tests/test_capability_probe.py
git commit -m "feat: add disk write-cache policy probe to CapabilityProbe"
```

**Manual verification (cannot be covered by unit tests):** once merged, run on the real dev machine and confirm `build_disk_write_cache_script` against the actual runtime disk resolves a real disk and a real `UserWriteCacheSetting` value — this task's tests only prove the JSON contract, not that the PnP/registry correlation matches this specific machine's disk topology.

---

### Task 4: Windows Defender exclusion probe (read-only, fix in Task 5)

**Files:**
- Modify: `app/services/capability_probe.py`
- Modify: `tests/test_capability_probe.py`

**Interfaces:**
- Produces: `def parse_defender_exclusion_json(raw_stdout: str, runtime_path: str) -> Lever` (pure), `async def probe_defender_exclusion(runtime_path: str, timeout: float = 10.0) -> Lever`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_capability_probe.py
from app.services.capability_probe import parse_defender_exclusion_json, probe_defender_exclusion


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


def test_probe_defender_exclusion_not_applicable_off_windows(monkeypatch) -> None:
    import sys

    monkeypatch.setattr(sys, "platform", "linux")

    lever = asyncio.run(probe_defender_exclusion("C:/Upflow/runtime"))

    assert lever.status.value == "not_applicable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_capability_probe.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_defender_exclusion_json'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to app/services/capability_probe.py

_DEFENDER_EXCLUSIONS_SCRIPT = """
try {
    $prefs = Get-MpPreference -ErrorAction Stop
    [PSCustomObject]@{ ok = $true; exclusions = @($prefs.ExclusionPath) } | ConvertTo-Json -Compress
} catch {
    [PSCustomObject]@{ ok = $false; error = $_.Exception.Message } | ConvertTo-Json -Compress
}
""".strip()


def _normalize_path_for_compare(path: str) -> str:
    return path.strip().rstrip("\\/").lower()


def parse_defender_exclusion_json(raw_stdout: str, runtime_path: str) -> Lever:
    lever_id, label = "defender_exclusion", "Windows Defender exclusion on runtime/"
    try:
        payload = json.loads(raw_stdout)
    except (json.JSONDecodeError, TypeError):
        return Lever(lever_id, label, LeverStatus.unavailable, "Could not read Defender exclusions", False)
    if not payload.get("ok"):
        return Lever(lever_id, label, LeverStatus.needs_admin, str(payload.get("error", "unknown error")), True)
    exclusions = {_normalize_path_for_compare(p) for p in payload.get("exclusions", []) if p}
    if _normalize_path_for_compare(runtime_path) in exclusions:
        return Lever(lever_id, label, LeverStatus.ok, "runtime/ is excluded from real-time scanning", False)
    return Lever(
        lever_id, label, LeverStatus.unavailable,
        "runtime/ is not excluded -- Defender real-time scanning adds overhead to every frame write",
        True,
    )


async def probe_defender_exclusion(runtime_path: str, timeout: float = 10.0) -> Lever:
    lever_id, label = "defender_exclusion", "Windows Defender exclusion on runtime/"
    if sys.platform != "win32":
        return Lever(lever_id, label, LeverStatus.not_applicable, "Windows only", False)
    try:
        stdout, stderr, returncode = await run_guarded_process(
            ["powershell.exe", "-NoProfile", "-Command", _DEFENDER_EXCLUSIONS_SCRIPT], timeout
        )
    except Exception:  # noqa: BLE001 -- a probe must never raise
        logger.warning("Defender exclusion probe failed to run", exc_info=True)
        return Lever(lever_id, label, LeverStatus.needs_admin, "Could not run the Defender exclusion probe", True)
    if returncode != 0:
        return Lever(lever_id, label, LeverStatus.needs_admin, f"Defender probe failed: {stderr.decode(errors='replace')[:200]}", True)
    return parse_defender_exclusion_json(stdout.decode(errors="replace"), runtime_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_capability_probe.py -v`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/capability_probe.py tests/test_capability_probe.py
git commit -m "feat: add Windows Defender exclusion probe to CapabilityProbe"
```

---

### Task 5: Elevation runner + `build_fix_script` + `CapabilityProbe` orchestrator class

**Files:**
- Modify: `app/services/capability_probe.py`
- Modify: `tests/test_capability_probe.py`
- Modify: `app/config.py`

**Interfaces:**
- Consumes: `probe_hags`, `probe_pcie_link`, `probe_disk_write_cache`, `probe_defender_exclusion` (Tasks 1-4), `run_guarded_process`.
- Produces: `def build_fix_script(lever_id: str, runtime_path: str) -> str` (raises `ValueError` for unknown/non-fixable ids), `class CapabilityProbe` with `__init__(self, settings: Settings)`, `async def list_levers(self) -> list[Lever]`, `async def rescan(self) -> list[Lever]`, `async def apply_fix(self, lever_id: str) -> Lever`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_capability_probe.py
from app.services.capability_probe import CapabilityProbe, build_fix_script
from app.config import Settings


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_capability_probe.py -v`
Expected: FAIL with `ImportError: cannot import name 'CapabilityProbe'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to app/config.py, near the other timeout-style fields (e.g. next to
# update_api_timeout_seconds):
capability_fix_timeout_seconds: float = Field(default=120.0, alias="CAPABILITY_FIX_TIMEOUT_SECONDS")
```

```python
# add to app/services/capability_probe.py
import base64

from app.config import Settings

_HAGS_FIX_SCRIPT = (
    "Set-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers' "
    "-Name HwSchMode -Value 2"
)

_DISK_WRITE_CACHE_FIX_TEMPLATE = """
$target = {path_literal}
$drive = (Get-Item -LiteralPath $target -ErrorAction Stop).PSDrive.Name
$partition = Get-Partition -DriveLetter $drive -ErrorAction Stop
$disk = Get-Disk -Number $partition.DiskNumber -ErrorAction Stop
$pnp = Get-PnpDevice -Class DiskDrive -ErrorAction Stop | Where-Object {{ $_.FriendlyName -eq $disk.FriendlyName }} | Select-Object -First 1
$regPath = "HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\$($pnp.InstanceId)\\Device Parameters\\Disk"
Set-ItemProperty -Path $regPath -Name "UserWriteCacheSetting" -Value 1
""".strip()


def build_fix_script(lever_id: str, runtime_path: str) -> str:
    if lever_id == "hags":
        return _HAGS_FIX_SCRIPT
    if lever_id == "defender_exclusion":
        return f"Add-MpPreference -ExclusionPath {_ps_single_quote(runtime_path)}"
    if lever_id == "disk_write_cache":
        return _DISK_WRITE_CACHE_FIX_TEMPLATE.format(path_literal=_ps_single_quote(runtime_path))
    raise ValueError(f"Lever {lever_id!r} has no fix script (not fixable or unknown)")


async def _run_elevated(inner_script: str, timeout: float) -> tuple[bool, str]:
    encoded = base64.b64encode(inner_script.encode("utf-16-le")).decode("ascii")
    outer = (
        "$p = Start-Process powershell.exe "
        f"-ArgumentList '-NoProfile','-EncodedCommand','{encoded}' "
        "-Verb RunAs -Wait -PassThru; exit $p.ExitCode"
    )
    try:
        _, _, returncode = await run_guarded_process(["powershell.exe", "-NoProfile", "-Command", outer], timeout)
    except Exception:  # noqa: BLE001 -- elevation failures must degrade, never raise
        return False, "Elevation failed to run"
    if returncode != 0:
        return False, "Elevation was cancelled or failed"
    return True, ""


class CapabilityProbe:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: list[Lever] | None = None

    async def list_levers(self) -> list[Lever]:
        if self._cache is None:
            self._cache = await self._probe_all()
        return self._cache

    async def rescan(self) -> list[Lever]:
        self._cache = await self._probe_all()
        return self._cache

    async def apply_fix(self, lever_id: str) -> Lever:
        runtime_path = str(self.settings.runtime_path)
        script = build_fix_script(lever_id, runtime_path)
        ok, message = await _run_elevated(script, self.settings.capability_fix_timeout_seconds)
        levers = await self.rescan()
        lever = next(l for l in levers if l.id == lever_id)
        if not ok and lever.status != LeverStatus.ok:
            return Lever(lever.id, lever.label, lever.status, f"{lever.detail} ({message})", lever.fixable)
        return lever

    async def _probe_all(self) -> list[Lever]:
        runtime_path = str(self.settings.runtime_path)
        return [
            probe_hags(),
            await probe_pcie_link(),
            await probe_disk_write_cache(runtime_path),
            await probe_defender_exclusion(runtime_path),
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_capability_probe.py -v`
Expected: 28 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/capability_probe.py app/config.py tests/test_capability_probe.py
git commit -m "feat: add elevation runner and CapabilityProbe orchestrator"
```

---

### Task 6: Capability API routes + `main.py` wiring

**Files:**
- Create: `app/api/capability_routes.py`
- Test: `tests/test_capability_routes.py`
- Modify: `app/schemas.py`, `app/main.py`

**Interfaces:**
- Consumes: `CapabilityProbe` (Task 5).
- Produces: `router = APIRouter(prefix="/api/v1/capabilities", tags=["capabilities"])` with `GET /`, `POST /rescan`, `POST /{lever_id}/fix`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_capability_routes.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.capability_routes import get_capability_probe, router
from app.services.capability_probe import CapabilityProbe, Lever, LeverStatus
from app.config import Settings


class FakeCapabilityProbe:
    def __init__(self) -> None:
        self.rescan_called = False
        self.fix_called_with: str | None = None

    async def list_levers(self) -> list[Lever]:
        return [Lever("hags", "HAGS", LeverStatus.ok, "enabled", False)]

    async def rescan(self) -> list[Lever]:
        self.rescan_called = True
        return await self.list_levers()

    async def apply_fix(self, lever_id: str) -> Lever:
        self.fix_called_with = lever_id
        return Lever(lever_id, lever_id, LeverStatus.ok, "fixed", False)


def make_client(fake: FakeCapabilityProbe) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_capability_probe] = lambda: fake
    return TestClient(app)


def test_get_capabilities_returns_levers() -> None:
    fake = FakeCapabilityProbe()
    client = make_client(fake)

    response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["levers"][0]["id"] == "hags"
    assert body["levers"][0]["fixable"] is False


def test_post_rescan_calls_rescan() -> None:
    fake = FakeCapabilityProbe()
    client = make_client(fake)

    response = client.post("/api/v1/capabilities/rescan")

    assert response.status_code == 200
    assert fake.rescan_called is True


def test_post_fix_calls_apply_fix_with_lever_id() -> None:
    fake = FakeCapabilityProbe()
    client = make_client(fake)

    response = client.post("/api/v1/capabilities/hags/fix")

    assert response.status_code == 200
    assert fake.fix_called_with == "hags"
    assert response.json()["lever"]["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_capability_routes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.capability_routes'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to app/schemas.py
class LeverResponse(BaseModel):
    id: str
    label: str
    status: LeverStatus
    detail: str
    fixable: bool


class CapabilitiesResponse(BaseModel):
    levers: list[LeverResponse]


class FixLeverResponse(BaseModel):
    lever: LeverResponse
```

Add the import at the top of `app/schemas.py`: `from app.services.capability_probe import LeverStatus`.

```python
# app/api/capability_routes.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import CapabilitiesResponse, FixLeverResponse, LeverResponse
from app.services.capability_probe import CapabilityProbe, Lever

router = APIRouter(prefix="/api/v1/capabilities", tags=["capabilities"])


def get_capability_probe(request: Request) -> CapabilityProbe:
    return request.app.state.capability_probe


def _to_response(lever: Lever) -> LeverResponse:
    return LeverResponse(id=lever.id, label=lever.label, status=lever.status, detail=lever.detail, fixable=lever.fixable)


@router.get("", response_model=CapabilitiesResponse)
async def get_capabilities(probe: CapabilityProbe = Depends(get_capability_probe)) -> CapabilitiesResponse:
    levers = await probe.list_levers()
    return CapabilitiesResponse(levers=[_to_response(lever) for lever in levers])


@router.post("/rescan", response_model=CapabilitiesResponse)
async def rescan_capabilities(probe: CapabilityProbe = Depends(get_capability_probe)) -> CapabilitiesResponse:
    levers = await probe.rescan()
    return CapabilitiesResponse(levers=[_to_response(lever) for lever in levers])


@router.post("/{lever_id}/fix", response_model=FixLeverResponse)
async def fix_lever(lever_id: str, probe: CapabilityProbe = Depends(get_capability_probe)) -> FixLeverResponse:
    try:
        lever = await probe.apply_fix(lever_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FixLeverResponse(lever=_to_response(lever))
```

```python
# app/main.py -- add import:
from app.api.capability_routes import router as capability_router
from app.services.capability_probe import CapabilityProbe

# inside lifespan(), alongside the other service constructions:
capability_probe = CapabilityProbe(settings)
# ...
app.state.capability_probe = capability_probe

# at module level, alongside app.include_router(api_router):
app.include_router(capability_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_capability_routes.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/api/capability_routes.py app/schemas.py app/main.py tests/test_capability_routes.py
git commit -m "feat: add /api/v1/capabilities routes"
```

---

### Task 7: ONNX CPU-EP-fallback probe -- synthetic input builder + profiling parser (pure)

**Files:**
- Create: `app/services/onnx_cpu_fallback_probe.py`
- Test: `tests/test_onnx_cpu_fallback_probe.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True, slots=True) class CpuFallbackReport` with fields `model_id: str`, `device_id: str`, `hot_ops: tuple[str, ...]`, `clean: bool`. `def build_synthetic_inputs(input_nodes: list[Any]) -> dict[str, "np.ndarray"]`. `def hot_cpu_ops(profile_events: list[dict], device_provider: str) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onnx_cpu_fallback_probe.py
from __future__ import annotations

import numpy as np

from app.services.onnx_cpu_fallback_probe import build_synthetic_inputs, hot_cpu_ops


class _FakeInputNode:
    def __init__(self, name: str, shape: list, type_: str) -> None:
        self.name = name
        self.shape = shape
        self.type = type_


def test_build_synthetic_inputs_uses_fixed_dims_as_is() -> None:
    nodes = [_FakeInputNode("audio", [1, 1, 100], "tensor(float)")]

    feeds = build_synthetic_inputs(nodes)

    assert feeds["audio"].shape == (1, 1, 100)
    assert feeds["audio"].dtype == np.float32


def test_build_synthetic_inputs_replaces_dynamic_dims_with_default() -> None:
    nodes = [_FakeInputNode("frame", [1, "height", "width", 3], "tensor(uint8)")]

    feeds = build_synthetic_inputs(nodes)

    assert feeds["frame"].shape == (1, 64, 64, 3)
    assert feeds["frame"].dtype == np.uint8


def test_build_synthetic_inputs_handles_multiple_inputs() -> None:
    nodes = [
        _FakeInputNode("a", [1, 8], "tensor(float16)"),
        _FakeInputNode("b", [1, 2], "tensor(int64)"),
    ]

    feeds = build_synthetic_inputs(nodes)

    assert set(feeds) == {"a", "b"}
    assert feeds["a"].dtype == np.float16
    assert feeds["b"].dtype == np.int64


def test_hot_cpu_ops_filters_nodes_on_other_provider() -> None:
    events = [
        {"cat": "Node", "args": {"op_name": "Conv", "provider": "CPUExecutionProvider"}},
        {"cat": "Node", "args": {"op_name": "Relu", "provider": "DmlExecutionProvider"}},
        {"cat": "Session", "args": {}},
    ]

    hot = hot_cpu_ops(events, device_provider="DmlExecutionProvider")

    assert hot == ["Conv"]


def test_hot_cpu_ops_returns_empty_when_all_on_device() -> None:
    events = [{"cat": "Node", "args": {"op_name": "Relu", "provider": "DmlExecutionProvider"}}]

    hot = hot_cpu_ops(events, device_provider="DmlExecutionProvider")

    assert hot == []


def test_hot_cpu_ops_ignores_events_without_provider_arg() -> None:
    events = [{"cat": "Node", "args": {"op_name": "Reshape"}}]

    hot = hot_cpu_ops(events, device_provider="DmlExecutionProvider")

    assert hot == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_onnx_cpu_fallback_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.onnx_cpu_fallback_probe'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/onnx_cpu_fallback_probe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Fase 0.1: detects ONNX Runtime ops that silently fall back to the CPU EP on
# a DirectML (or other GPU) session. `disable_cpu_ep_fallback` would make
# session CREATION raise on the first such op instead of enumerating them, so
# this uses ONNX Runtime's own profiling API instead: run once with a
# synthetic input, read the profiling JSON `end_profiling()` returns, and
# read the `provider` field ORT stamps on every executed node -- an official,
# structured mechanism (not string-scraping the general log stream).
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, Any] = {
    "tensor(uint8)": np.uint8,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}
_DEFAULT_DYNAMIC_DIM = 64


@dataclass(frozen=True, slots=True)
class CpuFallbackReport:
    model_id: str
    device_id: str
    hot_ops: tuple[str, ...]
    clean: bool


def _resolve_dim(dim: object) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    return _DEFAULT_DYNAMIC_DIM


def build_synthetic_inputs(input_nodes: list[Any]) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for node in input_nodes:
        shape = [_resolve_dim(dim) for dim in node.shape]
        dtype = _DTYPE_MAP.get(node.type, np.float32)
        feeds[node.name] = np.zeros(shape, dtype=dtype)
    return feeds


def hot_cpu_ops(profile_events: list[dict], device_provider: str) -> list[str]:
    hot: list[str] = []
    for event in profile_events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args") or {}
        provider = args.get("provider")
        if provider is None or provider == device_provider:
            continue
        hot.append(args.get("op_name", event.get("name", "unknown")))
    return hot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_onnx_cpu_fallback_probe.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/onnx_cpu_fallback_probe.py tests/test_onnx_cpu_fallback_probe.py
git commit -m "feat: add pure synthetic-input builder and profiling-JSON parser for CPU-EP-fallback diagnostic"
```

---

### Task 8: ONNX CPU-EP-fallback probe -- `probe_cpu_fallback` integration + catalog + `OnnxCpuFallbackProbe`

**Files:**
- Modify: `app/services/onnx_cpu_fallback_probe.py`
- Modify: `tests/test_onnx_cpu_fallback_probe.py`

**Interfaces:**
- Consumes: `build_synthetic_inputs`, `hot_cpu_ops` (Task 7). `app.services.backend_registry.BUILTIN_ONNX_MODELS` (existing), `app.services.devices_service.DevicesService` (existing).
- Produces: `def probe_cpu_fallback(model_path: str, device_ep: str, providers: list[str]) -> CpuFallbackReport`. `class OnnxCpuFallbackProbe` with `__init__(self, settings, devices)`, `def catalog(self) -> list[tuple[str, str]]` (model_id, device_id pairs), `async def scan(self, model_id: str, device_id: str) -> CpuFallbackReport`, `def cached(self, model_id: str, device_id: str) -> CpuFallbackReport | None`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_onnx_cpu_fallback_probe.py
import asyncio
from pathlib import Path

import onnx
import pytest
from onnx import helper, TensorProto

from app.config import Settings
from app.services.devices_service import CPU_DEVICE, DevicesService
from app.services.onnx_cpu_fallback_probe import OnnxCpuFallbackProbe, probe_cpu_fallback


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _write_trivial_relu_model(path: Path) -> None:
    # A single-node graph (Relu) is enough to exercise real ORT profiling
    # end-to-end on the CPU EP without any GPU or vendored model file.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "trivial", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def test_probe_cpu_fallback_reports_clean_when_all_on_target_ep(tmp_path: Path) -> None:
    model_path = tmp_path / "trivial.onnx"
    _write_trivial_relu_model(model_path)

    # device_ep is the raw ORT provider string profiling stamps on each node
    # ("CPUExecutionProvider"/"DmlExecutionProvider") -- NOT the user-facing
    # device_id ("cpu"/"dml:0"), which is a separate argument (see the
    # OnnxCpuFallbackProbe._resolve fix below for where these come from).
    report = probe_cpu_fallback(str(model_path), "cpu", "CPUExecutionProvider", providers=["CPUExecutionProvider"])

    assert report.clean is True
    assert report.hot_ops == ()
    assert report.device_id == "cpu"


def test_onnx_cpu_fallback_probe_catalog_includes_builtin_models(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    devices = DevicesService(settings)
    probe = OnnxCpuFallbackProbe(settings, devices)

    catalog = probe.catalog()

    # DevicesService always includes CPU_DEVICE (id "cpu") even with no GPU
    # present, so every builtin model is guaranteed to appear at least once.
    assert ("realesrgan-x4plus", CPU_DEVICE["id"]) in catalog


def test_onnx_cpu_fallback_probe_scan_caches_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path)
    devices = DevicesService(settings)
    probe = OnnxCpuFallbackProbe(settings, devices)
    calls = {"n": 0}

    def fake_probe_cpu_fallback(
        model_path: str, device_id: str, device_ep: str, providers: list[str]
    ) -> "CpuFallbackReport":
        calls["n"] += 1
        from app.services.onnx_cpu_fallback_probe import CpuFallbackReport

        return CpuFallbackReport("m", "cpu", (), True)

    import app.services.onnx_cpu_fallback_probe as mod

    monkeypatch.setattr(mod, "probe_cpu_fallback", fake_probe_cpu_fallback)

    first = asyncio.run(probe.scan("m", "cpu"))
    assert probe.cached("m", "cpu") == first
    assert calls["n"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_onnx_cpu_fallback_probe.py -v`
Expected: FAIL with `ImportError: cannot import name 'OnnxCpuFallbackProbe'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to app/services/onnx_cpu_fallback_probe.py
import asyncio
import json
from pathlib import Path

from app.config import Settings
from app.services.backend_registry import BUILTIN_ONNX_MODELS
from app.services.devices_service import CPU_DEVICE_ID, DevicesService


def probe_cpu_fallback(model_path: str, device_id: str, device_ep: str, providers: list[Any]) -> CpuFallbackReport:
    """device_id is the user-facing id ("cpu"/"dml:0", stored on the report);
    device_ep is the raw ORT provider string ("CPUExecutionProvider"/
    "DmlExecutionProvider") profiling stamps on each node -- they are NOT the
    same string for GPU devices, see OnnxCpuFallbackProbe._resolve."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_profiling = True
    session = ort.InferenceSession(model_path, so, providers=providers)
    feeds = build_synthetic_inputs(session.get_inputs())
    output_names = [output.name for output in session.get_outputs()]
    session.run(output_names, feeds)
    profile_path = session.end_profiling()
    try:
        events = json.loads(Path(profile_path).read_text())
    finally:
        Path(profile_path).unlink(missing_ok=True)
    hot_ops = tuple(hot_cpu_ops(events, device_ep))
    return CpuFallbackReport(model_id=Path(model_path).stem, device_id=device_id, hot_ops=hot_ops, clean=not hot_ops)


class OnnxCpuFallbackProbe:
    """Diagnostic-only: probe_cpu_fallback runs a real ORT session, so this
    is never called from a job's hot path -- only manually from the
    Optimization Center diagnostics panel, one (model, device) pair at a
    time."""

    def __init__(self, settings: Settings, devices: DevicesService) -> None:
        self.settings = settings
        self.devices = devices
        self._cache: dict[tuple[str, str], CpuFallbackReport] = {}
        self._lock = asyncio.Lock()

    def catalog(self) -> list[tuple[str, str]]:
        # Builtin Real-ESRGAN ONNX exports + Apollo -- both have a single,
        # fixed-role graph with a known input contract. AudioSR (multi-graph
        # DDIM loop) and GMFSS (4 graphs, ORT_DISABLE_ALL-gated) need their
        # own model-specific harness to probe meaningfully -- deferred, see
        # the plan's "Deferred" section.
        device_ids = [device["id"] for device in self.devices.list_devices()]
        pairs: list[tuple[str, str]] = []
        for model_id in BUILTIN_ONNX_MODELS:
            for device_id in device_ids:
                pairs.append((model_id, device_id))
        if self.settings.apollo_restore_model_path.exists():
            for device_id in device_ids:
                pairs.append(("apollo", device_id))
        return pairs

    def cached(self, model_id: str, device_id: str) -> CpuFallbackReport | None:
        return self._cache.get((model_id, device_id))

    async def scan(self, model_id: str, device_id: str) -> CpuFallbackReport:
        model_path, providers, device_ep = self._resolve(model_id, device_id)
        async with self._lock:
            report = await asyncio.to_thread(probe_cpu_fallback, model_path, device_id, device_ep, providers)
            self._cache[(model_id, device_id)] = report
            return report

    def _resolve(self, model_id: str, device_id: str) -> tuple[str, list[Any], str]:
        from app.services.engines.onnx_upscaler import _build_providers

        providers = _build_providers(device_id)
        # _build_providers returns a plain string for "cpu" but a
        # (name, options) tuple for "dml:N" -- device_ep must always be the
        # bare provider name string to compare against profiling's `provider`
        # field, so unwrap the tuple case.
        first = providers[0]
        device_ep = first[0] if isinstance(first, tuple) else first
        if model_id == "apollo":
            return str(self.settings.apollo_restore_model_path), providers, device_ep
        model = BUILTIN_ONNX_MODELS[model_id]
        return str(self.settings.builtin_onnx_path / model.filename), providers, device_ep
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_onnx_cpu_fallback_probe.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/onnx_cpu_fallback_probe.py tests/test_onnx_cpu_fallback_probe.py
git commit -m "feat: add probe_cpu_fallback and OnnxCpuFallbackProbe with builtin-model catalog"
```

---

### Task 9: ONNX diagnostics API routes + `main.py` wiring

**Files:**
- Modify: `app/api/capability_routes.py`, `app/schemas.py`, `app/main.py`
- Modify: `tests/test_capability_routes.py`

**Interfaces:**
- Consumes: `OnnxCpuFallbackProbe` (Task 8).
- Produces: `GET /api/v1/capabilities/onnx-diagnostics`, `POST /api/v1/capabilities/onnx-diagnostics/{model_id}/{device_id}/scan` on the same router.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_capability_routes.py
from app.api.capability_routes import get_onnx_cpu_fallback_probe
from app.services.onnx_cpu_fallback_probe import CpuFallbackReport


class FakeOnnxCpuFallbackProbe:
    def catalog(self) -> list[tuple[str, str]]:
        return [("realesrgan-x4plus", "cpu")]

    def cached(self, model_id: str, device_id: str) -> CpuFallbackReport | None:
        return None

    async def scan(self, model_id: str, device_id: str) -> CpuFallbackReport:
        return CpuFallbackReport(model_id, device_id, ("Conv",), False)


def make_diagnostics_client(fake_probe, fake_onnx_probe) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_capability_probe] = lambda: fake_probe
    app.dependency_overrides[get_onnx_cpu_fallback_probe] = lambda: fake_onnx_probe
    return TestClient(app)


def test_get_onnx_diagnostics_lists_catalog_with_cached_results() -> None:
    client = make_diagnostics_client(FakeCapabilityProbe(), FakeOnnxCpuFallbackProbe())

    response = client.get("/api/v1/capabilities/onnx-diagnostics")

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert entries[0]["modelId"] == "realesrgan-x4plus"
    assert entries[0]["report"] is None


def test_post_onnx_diagnostics_scan_runs_and_returns_report() -> None:
    client = make_diagnostics_client(FakeCapabilityProbe(), FakeOnnxCpuFallbackProbe())

    response = client.post("/api/v1/capabilities/onnx-diagnostics/realesrgan-x4plus/cpu/scan")

    assert response.status_code == 200
    report = response.json()["report"]
    assert report["hotOps"] == ["Conv"]
    assert report["clean"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_capability_routes.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_onnx_cpu_fallback_probe'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to app/schemas.py
class CpuFallbackReportResponse(BaseModel):
    model_id: str = Field(serialization_alias="modelId")
    device_id: str = Field(serialization_alias="deviceId")
    hot_ops: list[str] = Field(serialization_alias="hotOps")
    clean: bool


class OnnxDiagnosticEntryResponse(BaseModel):
    model_id: str = Field(serialization_alias="modelId")
    device_id: str = Field(serialization_alias="deviceId")
    report: CpuFallbackReportResponse | None = None


class OnnxDiagnosticsResponse(BaseModel):
    entries: list[OnnxDiagnosticEntryResponse]


class ScanOnnxDiagnosticResponse(BaseModel):
    report: CpuFallbackReportResponse
```

```python
# add to app/api/capability_routes.py
from app.schemas import (
    CpuFallbackReportResponse,
    OnnxDiagnosticEntryResponse,
    OnnxDiagnosticsResponse,
    ScanOnnxDiagnosticResponse,
)
from app.services.onnx_cpu_fallback_probe import CpuFallbackReport, OnnxCpuFallbackProbe


def get_onnx_cpu_fallback_probe(request: Request) -> OnnxCpuFallbackProbe:
    return request.app.state.onnx_cpu_fallback_probe


def _report_to_response(report: CpuFallbackReport) -> CpuFallbackReportResponse:
    return CpuFallbackReportResponse(
        model_id=report.model_id, device_id=report.device_id, hot_ops=list(report.hot_ops), clean=report.clean
    )


@router.get("/onnx-diagnostics", response_model=OnnxDiagnosticsResponse)
async def get_onnx_diagnostics(
    probe: OnnxCpuFallbackProbe = Depends(get_onnx_cpu_fallback_probe),
) -> OnnxDiagnosticsResponse:
    entries = [
        OnnxDiagnosticEntryResponse(
            model_id=model_id,
            device_id=device_id,
            report=_report_to_response(cached) if (cached := probe.cached(model_id, device_id)) else None,
        )
        for model_id, device_id in probe.catalog()
    ]
    return OnnxDiagnosticsResponse(entries=entries)


@router.post("/onnx-diagnostics/{model_id}/{device_id}/scan", response_model=ScanOnnxDiagnosticResponse)
async def scan_onnx_diagnostic(
    model_id: str, device_id: str, probe: OnnxCpuFallbackProbe = Depends(get_onnx_cpu_fallback_probe)
) -> ScanOnnxDiagnosticResponse:
    try:
        report = await probe.scan(model_id, device_id)
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ScanOnnxDiagnosticResponse(report=_report_to_response(report))
```

```python
# app/main.py -- add import and wiring:
from app.services.onnx_cpu_fallback_probe import OnnxCpuFallbackProbe

# inside lifespan(), after devices_service is constructed:
onnx_cpu_fallback_probe = OnnxCpuFallbackProbe(settings, devices_service)
# ...
app.state.onnx_cpu_fallback_probe = onnx_cpu_fallback_probe
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_capability_routes.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add app/api/capability_routes.py app/schemas.py app/main.py tests/test_capability_routes.py
git commit -m "feat: add ONNX CPU-EP-fallback diagnostics routes"
```

---

### Task 10: `ApolloRestorer` IOBinding retrofit (Fase 0.2)

**Files:**
- Modify: `app/services/engines/apollo_restore.py`
- Modify: `tests/test_apollo_restore.py`

**Interfaces:**
- Consumes: same IOBinding technique as `OnnxVideoUpscaler._infer_iobinding` (`app/services/engines/onnx_video_upscaler.py:614-640`): `session.io_binding()`, `ort.OrtValue.ortvalue_from_numpy`, `bind_ortvalue_input`, `bind_output`, `run_with_iobinding`, `copy_outputs_to_cpu`.
- Produces: `ApolloRestorer._infer_chunk` gains a DirectML IOBinding fast path with the same best-effort/fallback/one-time-warning contract.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_apollo_restore.py
class FakeIoBinding:
    def __init__(self) -> None:
        self.bound_input: tuple[str, Any] | None = None
        self.bound_output: str | None = None

    def bind_ortvalue_input(self, name: str, value: Any) -> None:
        self.bound_input = (name, value)

    def bind_output(self, name: str, device: str) -> None:
        self.bound_output = name

    def copy_outputs_to_cpu(self) -> list[np.ndarray]:
        return [self.bound_input[1].numpy() * 2.0]  # arbitrary marker so the test can assert this path ran


class FakeDmlSession:
    def __init__(self) -> None:
        self.io_binding_calls = 0
        self.run_with_iobinding_calls = 0

    def get_inputs(self) -> list[FakeIoInfo]:
        return [FakeIoInfo("audio")]

    def get_outputs(self) -> list[FakeIoInfo]:
        return [FakeIoInfo("restored")]

    def io_binding(self) -> FakeIoBinding:
        self.io_binding_calls += 1
        return FakeIoBinding()

    def run_with_iobinding(self, binding: FakeIoBinding) -> None:
        self.run_with_iobinding_calls += 1


def test_infer_chunk_uses_iobinding_on_dml_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    restorer = ApolloRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    session = FakeDmlSession()
    segment = np.ones(8, dtype=np.float32)

    class _FakeOrtValue:
        def __init__(self, array: np.ndarray) -> None:
            self._array = array

        def numpy(self) -> np.ndarray:
            return self._array

    class _FakeOrt:
        class OrtValue:
            @staticmethod
            def ortvalue_from_numpy(array: np.ndarray, device: str, device_id: int) -> "_FakeOrtValue":
                return _FakeOrtValue(array)

    monkeypatch.setattr("app.services.engines.apollo_restore._import_onnxruntime", lambda: _FakeOrt)

    result = restorer._infer_chunk(session, segment, device="dml:0")

    assert session.io_binding_calls == 1
    assert session.run_with_iobinding_calls == 1
    assert result.shape == (8,)
    assert np.array_equal(result, segment.astype(np.float64) * 2.0)


def test_infer_chunk_falls_back_to_plain_run_off_dml(tmp_path: Path) -> None:
    restorer = ApolloRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    session = fake_session("cpu")
    segment = np.ones(8, dtype=np.float32)

    result = restorer._infer_chunk(session, segment, device="cpu")

    assert result.shape == (8,)


def test_infer_chunk_falls_back_when_iobinding_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    restorer = ApolloRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    session = fake_session("dml:0")  # plain FakeApolloSession has no io_binding()
    segment = np.ones(8, dtype=np.float32)

    result = restorer._infer_chunk(session, segment, device="dml:0")

    assert result.shape == (8,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_apollo_restore.py -v`
Expected: FAIL with `TypeError: _infer_chunk() got an unexpected keyword argument 'device'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/engines/apollo_restore.py -- modify _restore_chunked to pass
# device through, and replace _infer_chunk with an IOBinding-aware version.
# Change the _restore_chunked signature and its one call site to thread
# `device` through, then:

DML_DEVICE_PREFIX = "dml:"


def _import_onnxruntime() -> Any:
    import onnxruntime as ort

    return ort


class ApolloRestorer:
    # ... (__init__, available, release_device, run, _run_and_save unchanged) ...

    def _restore_chunked(
        self, session: Any, audio: np.ndarray, device: str, throttle: float = 0.0,
        chunk_seconds: float = 1.0, overlap_seconds: float = OVERLAP_SECONDS,
    ) -> np.ndarray:
        total = audio.shape[-1]
        window_length = max(1, int(chunk_seconds * APOLLO_SAMPLE_RATE))
        overlap = min(int(overlap_seconds * APOLLO_SAMPLE_RATE), window_length // 2)
        hop = max(1, window_length - overlap)
        window = np.hanning(window_length).astype(np.float64)

        accumulator = np.zeros(total, dtype=np.float64)
        weight_sum = np.zeros(total, dtype=np.float64)
        start = 0
        while start < total:
            end = min(start + window_length, total)
            segment = audio[start:end]
            weights = window[: end - start]
            restored = self._infer_chunk(session, segment, device)
            accumulator[start:end] += restored * weights
            weight_sum[start:end] += weights
            if end >= total:
                break
            start += hop
            if throttle > 0:
                time.sleep(throttle)
        return (accumulator / np.maximum(weight_sum, 1e-8)).astype(np.float32)

    def _infer_chunk(self, session: Any, segment: np.ndarray, device: str) -> np.ndarray:
        batch = segment.reshape(1, 1, -1).astype(np.float32)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        if device.startswith(DML_DEVICE_PREFIX):
            bound = self._infer_iobinding(session, batch, input_name, output_name, device)
            if bound is not None:
                return np.asarray(bound, dtype=np.float64).reshape(-1)
        try:
            result = session.run([output_name], {input_name: batch})[0]
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise _wrap_onnx_error("Apollo inference failed", exc) from exc
        return np.asarray(result, dtype=np.float64).reshape(-1)

    def _infer_iobinding(
        self, session: Any, batch: np.ndarray, input_name: str, output_name: str, device: str
    ) -> np.ndarray | None:
        # Best-effort, same contract as OnnxVideoUpscaler._infer_iobinding:
        # any failure falls back to a plain run rather than failing the job,
        # and a persistent failure is logged once (not once per chunk) so it
        # doesn't silently downgrade every chunk to the slower path forever.
        try:
            ort = _import_onnxruntime()
            device_id = int(device.split(":", 1)[1]) if ":" in device else 0
            io_binding = session.io_binding()
            input_value = ort.OrtValue.ortvalue_from_numpy(batch, "dml", device_id)
            io_binding.bind_ortvalue_input(input_name, input_value)
            io_binding.bind_output(output_name, "dml")
            session.run_with_iobinding(io_binding)
            return io_binding.copy_outputs_to_cpu()[0]
        except Exception:  # noqa: BLE001
            if not self._iobinding_warned:
                self._iobinding_warned = True
                logger.warning(
                    "Apollo ONNX IO binding failed on %s; falling back to the slower plain-run path", device,
                    exc_info=True,
                )
            return None
```

Also add `self._iobinding_warned = False` to `ApolloRestorer.__init__`, `import logging` + `logger = logging.getLogger(__name__)` at module level (mirroring `onnx_video_upscaler.py`), and update the one call site in `_run_and_save` (`restore_mono`) to pass `device` into `self._restore_chunked(session, mono, device, throttle, chunk_seconds, overlap_seconds)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_apollo_restore.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/engines/apollo_restore.py tests/test_apollo_restore.py
git commit -m "perf: add DirectML IOBinding fast path to ApolloRestorer (Fase 0.2)"
```

**Manual verification:** benchmark Apollo restore before/after on the RX 7800 XT (same clip, `device=dml:0`), record the real ms/chunk delta in the PR description -- per the spec's "no gain claimed without measurement" rule. `AudioSrRestorer` and `GmfssEngine` IOBinding retrofits are explicitly deferred (see the plan's File Structure section) -- do not attempt them as part of this task.

---

### Task 11: Frontend API client + `useCapabilities` hook

**Files:**
- Modify: `frontend/src/lib/apiTypes.ts`, `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/useCapabilities.ts`
- Test: `frontend/src/hooks/useCapabilities.test.tsx`

**Interfaces:**
- Produces: `LeverResponse`, `CapabilitiesResponse`, `CpuFallbackReportResponse`, `OnnxDiagnosticEntryResponse`, `OnnxDiagnosticsResponse` types; `getCapabilities`, `rescanCapabilities`, `fixLever`, `getOnnxDiagnostics`, `scanOnnxDiagnostic` functions; `useCapabilities()` hook returning `{ levers, isLoading, isError, rescan, isRescanning, fix, fixingLeverId }`.

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/hooks/useCapabilities.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useCapabilities } from "./useCapabilities";
import * as api from "../lib/api";

function withQueryClient() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe("useCapabilities", () => {
  it("loads levers on mount", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({
      levers: [{ id: "hags", label: "HAGS", status: "ok", detail: "enabled", fixable: false }],
    });

    const { result } = renderHook(() => useCapabilities(), { wrapper: withQueryClient() });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.levers[0].id).toBe("hags");
  });

  it("calls fixLever and tracks which lever is fixing", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({ levers: [] });
    vi.spyOn(api, "fixLever").mockResolvedValue({
      lever: { id: "hags", label: "HAGS", status: "ok", detail: "fixed", fixable: false },
    });

    const { result } = renderHook(() => useCapabilities(), { wrapper: withQueryClient() });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => {
      result.current.fix("hags");
    });

    await waitFor(() => expect(api.fixLever).toHaveBeenCalledWith("hags"));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run useCapabilities`
Expected: FAIL with `Cannot find module './useCapabilities'`

- [ ] **Step 3: Write minimal implementation**

```typescript
// add to frontend/src/lib/apiTypes.ts
export type LeverStatus = "ok" | "unavailable" | "not_applicable" | "needs_admin";

export interface LeverResponse {
  id: string;
  label: string;
  status: LeverStatus;
  detail: string;
  fixable: boolean;
}

export interface CapabilitiesResponse {
  levers: LeverResponse[];
}

export interface FixLeverResponse {
  lever: LeverResponse;
}

export interface CpuFallbackReportResponse {
  modelId: string;
  deviceId: string;
  hotOps: string[];
  clean: boolean;
}

export interface OnnxDiagnosticEntryResponse {
  modelId: string;
  deviceId: string;
  report: CpuFallbackReportResponse | null;
}

export interface OnnxDiagnosticsResponse {
  entries: OnnxDiagnosticEntryResponse[];
}

export interface ScanOnnxDiagnosticResponse {
  report: CpuFallbackReportResponse;
}
```

```typescript
// add to frontend/src/lib/api.ts (add the new types to the existing import block)
import type {
  CapabilitiesResponse,
  FixLeverResponse,
  OnnxDiagnosticsResponse,
  ScanOnnxDiagnosticResponse,
  // ...(keep the existing imports too)
} from "./apiTypes";

export function getCapabilities(): Promise<CapabilitiesResponse> {
  return apiGet<CapabilitiesResponse>("/capabilities");
}

export function rescanCapabilities(): Promise<CapabilitiesResponse> {
  return apiPost<CapabilitiesResponse>("/capabilities/rescan");
}

export function fixLever(leverId: string): Promise<FixLeverResponse> {
  return apiPost<FixLeverResponse>(`/capabilities/${leverId}/fix`);
}

export function getOnnxDiagnostics(): Promise<OnnxDiagnosticsResponse> {
  return apiGet<OnnxDiagnosticsResponse>("/capabilities/onnx-diagnostics");
}

export function scanOnnxDiagnostic(modelId: string, deviceId: string): Promise<ScanOnnxDiagnosticResponse> {
  return apiPost<ScanOnnxDiagnosticResponse>(`/capabilities/onnx-diagnostics/${modelId}/${deviceId}/scan`);
}
```

```typescript
// frontend/src/hooks/useCapabilities.ts
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { fixLever, getCapabilities, rescanCapabilities } from "../lib/api";
import type { LeverResponse } from "../lib/apiTypes";

const CAPABILITIES_QUERY_KEY = ["capabilities"] as const;

export function useCapabilities() {
  const queryClient = useQueryClient();
  const [fixingLeverId, setFixingLeverId] = useState<string | null>(null);

  const capabilitiesQuery = useQuery({ queryKey: CAPABILITIES_QUERY_KEY, queryFn: getCapabilities });

  const rescanMutation = useMutation({
    mutationFn: rescanCapabilities,
    onSuccess: (data) => queryClient.setQueryData(CAPABILITIES_QUERY_KEY, data),
  });

  const fixMutation = useMutation({
    mutationFn: fixLever,
    onMutate: (leverId: string) => setFixingLeverId(leverId),
    onSettled: () => setFixingLeverId(null),
    onSuccess: (data, leverId) => {
      queryClient.setQueryData<{ levers: LeverResponse[] } | undefined>(CAPABILITIES_QUERY_KEY, (prev) => {
        if (!prev) return prev;
        return { levers: prev.levers.map((lever) => (lever.id === leverId ? data.lever : lever)) };
      });
    },
  });

  return {
    levers: capabilitiesQuery.data?.levers ?? [],
    isLoading: capabilitiesQuery.isLoading,
    isError: capabilitiesQuery.isError,
    rescan: rescanMutation.mutate,
    isRescanning: rescanMutation.isPending,
    fix: fixMutation.mutate,
    fixingLeverId,
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- --run useCapabilities`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/apiTypes.ts frontend/src/lib/api.ts frontend/src/hooks/useCapabilities.ts frontend/src/hooks/useCapabilities.test.tsx
git commit -m "feat: add capabilities API client and useCapabilities hook"
```

---

### Task 12: `OptimizationCenter` component (levers + diagnostics + BIOS checklist) wired into Settings

**Files:**
- Create: `frontend/src/modules/settings/OptimizationCenter.tsx`
- Test: `frontend/src/modules/settings/OptimizationCenter.test.tsx`
- Modify: `frontend/src/modules/settings/SettingsPage.tsx`

**Interfaces:**
- Consumes: `useCapabilities()` (Task 11), `getOnnxDiagnostics`/`scanOnnxDiagnostic` (Task 11).
- Produces: `export function OptimizationCenter()`.

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/modules/settings/OptimizationCenter.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { OptimizationCenter } from "./OptimizationCenter";
import * as api from "../../lib/api";

function renderWithClient() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <OptimizationCenter />
    </QueryClientProvider>,
  );
}

describe("OptimizationCenter", () => {
  it("renders a row per lever with its status", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({
      levers: [
        { id: "hags", label: "Hardware-accelerated GPU scheduling", status: "ok", detail: "enabled", fixable: false },
        { id: "defender_exclusion", label: "Windows Defender exclusion", status: "unavailable", detail: "not excluded", fixable: true },
      ],
    });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });

    renderWithClient();

    expect(await screen.findByText("Hardware-accelerated GPU scheduling")).toBeInTheDocument();
    expect(screen.getByText("Windows Defender exclusion")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /fix/i })).toBeInTheDocument();
  });

  it("calls fixLever when the Fix button is clicked", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({
      levers: [{ id: "hags", label: "HAGS", status: "unavailable", detail: "disabled", fixable: true }],
    });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });
    const fixSpy = vi.spyOn(api, "fixLever").mockResolvedValue({
      lever: { id: "hags", label: "HAGS", status: "ok", detail: "fixed", fixable: false },
    });

    renderWithClient();
    const button = await screen.findByRole("button", { name: /fix/i });
    await userEvent.click(button);

    await waitFor(() => expect(fixSpy).toHaveBeenCalledWith("hags"));
  });

  it("renders the Resizable BAR checklist section", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({ levers: [] });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });

    renderWithClient();

    expect(await screen.findByText(/Resizable BAR/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run OptimizationCenter`
Expected: FAIL with `Cannot find module './OptimizationCenter'`

- [ ] **Step 3: Write minimal implementation**

```tsx
// frontend/src/modules/settings/OptimizationCenter.tsx
import { useQuery } from "@tanstack/react-query";
import { useCapabilities } from "../../hooks/useCapabilities";
import { getOnnxDiagnostics, scanOnnxDiagnostic } from "../../lib/api";
import type { LeverResponse, LeverStatus, OnnxDiagnosticEntryResponse } from "../../lib/apiTypes";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

const STATUS_BADGE: Record<LeverStatus, string> = {
  ok: "✅",
  unavailable: "❌",
  not_applicable: "⚠️",
  needs_admin: "🔒",
};

function LeverRow({ lever, onFix, isFixing }: { lever: LeverResponse; onFix: (id: string) => void; isFixing: boolean }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border py-2 text-sm">
      <div className="flex flex-col">
        <span className="text-text">{lever.label}</span>
        <span className="text-xs text-text-faint">{lever.detail}</span>
      </div>
      <div className="flex items-center gap-3">
        <span aria-hidden="true">{STATUS_BADGE[lever.status]}</span>
        {lever.fixable && (
          <button
            type="button"
            className="rounded border border-border px-2 py-1 text-xs text-text hover:bg-surface-2"
            onClick={() => onFix(lever.id)}
            disabled={isFixing}
          >
            {isFixing ? "Fixing…" : "Fix"}
          </button>
        )}
      </div>
    </div>
  );
}

function LeversSection() {
  const { levers, isLoading, isError, rescan, isRescanning, fix, fixingLeverId } = useCapabilities();

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <div className="flex items-center justify-between">
        <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Optimization Center</h2>
        <button
          type="button"
          className="rounded border border-border px-2 py-1 text-xs text-text hover:bg-surface-2"
          onClick={() => rescan()}
          disabled={isRescanning}
        >
          {isRescanning ? "Scanning…" : "Re-scan"}
        </button>
      </div>
      {isLoading && <p className="text-sm text-text-dim">Loading capability levers…</p>}
      {isError && <p className="text-sm text-danger">Could not load capability levers.</p>}
      {levers.map((lever) => (
        <LeverRow key={lever.id} lever={lever} onFix={fix} isFixing={fixingLeverId === lever.id} />
      ))}
    </div>
  );
}

function DiagnosticEntryRow({ entry }: { entry: OnnxDiagnosticEntryResponse }) {
  const queryClient = useQueryClient();
  const scanMutation = useMutation({
    mutationFn: () => scanOnnxDiagnostic(entry.modelId, entry.deviceId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["onnx-diagnostics"] }),
  });

  return (
    <div className="flex items-center justify-between gap-4 border-b border-border py-2 text-sm">
      <span className="text-text">
        {entry.modelId} @ {entry.deviceId}
      </span>
      <div className="flex items-center gap-3">
        {entry.report && (
          <span className="text-xs text-text-faint">
            {entry.report.clean ? "No CPU fallback" : `${entry.report.hotOps.length} op(s) on CPU`}
          </span>
        )}
        <button
          type="button"
          className="rounded border border-border px-2 py-1 text-xs text-text hover:bg-surface-2"
          onClick={() => scanMutation.mutate()}
          disabled={scanMutation.isPending}
        >
          {scanMutation.isPending ? "Scanning…" : "Scan"}
        </button>
      </div>
    </div>
  );
}

function DiagnosticsSection() {
  const diagnosticsQuery = useQuery({ queryKey: ["onnx-diagnostics"], queryFn: getOnnxDiagnostics });
  const entries = diagnosticsQuery.data?.entries ?? [];

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Diagnostics</h2>
      <p className="text-xs text-text-faint">
        Checks whether an ONNX model silently falls back to the CPU execution provider on your GPU. Run manually per
        model/device -- not part of a real job.
      </p>
      {entries.map((entry) => (
        <DiagnosticEntryRow key={`${entry.modelId}:${entry.deviceId}`} entry={entry} />
      ))}
    </div>
  );
}

function ResizableBarChecklist() {
  const STORAGE_KEY = "upflow.resizableBarConfirmed";
  const [confirmed, setConfirmed] = useState<boolean>(() => localStorage.getItem(STORAGE_KEY) === "true");

  function toggle(): void {
    const next = !confirmed;
    setConfirmed(next);
    localStorage.setItem(STORAGE_KEY, String(next));
  }

  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-surface p-4 text-sm">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        Resizable BAR / Above 4G Decoding
      </h2>
      <p className="text-xs text-text-faint">
        Not detectable from software (lives in BIOS/UEFI firmware). Enable it in your motherboard's BIOS setup if
        supported, then confirm here so this panel remembers your setup.
      </p>
      <label className="flex items-center gap-2 text-xs text-text">
        <input type="checkbox" checked={confirmed} onChange={toggle} />
        I've confirmed Resizable BAR / Above 4G Decoding is enabled in BIOS
      </label>
    </div>
  );
}

export function OptimizationCenter() {
  return (
    <div className="flex flex-col gap-4">
      <LeversSection />
      <DiagnosticsSection />
      <ResizableBarChecklist />
    </div>
  );
}
```

```tsx
// frontend/src/modules/settings/SettingsPage.tsx -- add the import and render it:
import { OptimizationCenter } from "./OptimizationCenter";

// inside export function SettingsPage(), after the existing grid div:
<OptimizationCenter />
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- --run OptimizationCenter`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add frontend/src/modules/settings/OptimizationCenter.tsx frontend/src/modules/settings/OptimizationCenter.test.tsx frontend/src/modules/settings/SettingsPage.tsx
git commit -m "feat: add Optimization Center panel to Settings"
```

---

### Task 13: Docs

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `.env.example`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Add `.env.example` entry**

Add near the other timeout-style vars in `.env.example`:

```
# Timeout (seconds) for a one-click Optimization Center admin fix to wait for
# the user to respond to the Windows UAC elevation prompt.
CAPABILITY_FIX_TIMEOUT_SECONDS=120
```

- [ ] **Step 2: Add a README section**

Add a new `## Optimization Center` section to `README.md` describing: what it detects (HAGS, PCIe link, disk write-cache, Defender exclusion, ONNX CPU-EP-fallback), that fixes run via a UAC-elevated PowerShell one-liner, that the disk write-cache and HAGS fixes need a reboot to take effect, and that AudioSR/GMFSS IOBinding + the BIOS Resizable BAR check are diagnostic/manual only in this iteration.

- [ ] **Step 3: Update `CLAUDE.md`**

Add a row to the services table in `CLAUDE.md` for `capability_probe.py` and `onnx_cpu_fallback_probe.py`, same style as the existing `providers_service.py`/`proxy_service.py`-style rows (adjust to this repo's actual table content, e.g. under `backend/app/services/` or wherever this repo's `CLAUDE.md` documents `app/services/`).

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md .env.example
git commit -m "docs: document Optimization Center levers and diagnostics"
```

---

## Self-Review

**Spec coverage:**
- Fase 0.1 (CPU-EP-fallback diagnostic) → Tasks 7-9. ✅
- Fase 0.2 (IOBinding audit) → audit done via `grep` before writing this plan (confirmed via reading `app/services/engines/*.py`: only `OnnxVideoUpscaler` had it); retrofit scoped to `ApolloRestorer` in Task 10, `AudioSrRestorer`/`GmfssEngine`/`OnnxUpscaler` explicitly deferred with rationale in File Structure. ✅
- Fase 1 (`CapabilityProbe` + panel, 4 levers, UAC elevation, API) → Tasks 1-6, 11-12. ✅
- Fase 2 (BIOS checklist) → folded into Task 12 (`ResizableBarChecklist`), no backend needed per spec. ✅
- Fase 3 → explicitly out of scope, not a task. ✅
- "No gain claimed without measurement" → called out as a manual step in Tasks 3 and 10 (the two probes/fixes with a real, unverified-by-CI hardware dependency).

**Placeholder scan:** no TBD/TODO; every step has complete code. The one deliberately open item ("Manual verification" callouts in Tasks 3 and 10) is explicit and actionable, not a placeholder for missing work.

**Type/name consistency:** `Lever`/`LeverStatus` (Task 1) used identically through Tasks 2-6 and the schema layer (Task 6/9). `CpuFallbackReport` (Task 7) used identically through Tasks 8-9. `useCapabilities()`'s returned shape (Task 11) matches exactly what `OptimizationCenter` (Task 12) destructures.
