# Admisión de jobs por capacidad real de dispositivo (subproyecto B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `DeviceSemaphores` so job admission also checks real free resources (VRAM for GPU, RAM for CPU) of the device a job wants, not just a job-count ceiling — catching both external GPU/RAM pressure (other apps) and heterogeneous per-job memory footprints, while staying pluggable for hardware/platforms that don't exist yet (NPU, non-Windows).

**Architecture:** New `ResourceProbe` protocol (`app/services/resource_probes.py`) with `DxgiVramProbe` (real, Windows/DirectML via `IDXGIAdapter3::QueryVideoMemoryInfo`), `SystemRamProbe` (real, `psutil`), and `NullProbe` (always "unknown" — fail-open stub for `npu` and any future kind). `DeviceSemaphores` (`app/services/device_semaphores.py`) takes an optional `resource_probes: dict[str, ResourceProbe]` and its `_reserve_if_free` gains a second condition alongside the existing job-count check: `probe.free_capacity_mb(device_id) >= MIN_FREE_MB[kind]`. The blocking `acquire()` wait loop adds a periodic re-check (`RESOURCE_POLL_INTERVAL_SECONDS`) so resource pressure that clears for reasons outside this app (not from our own `release()`) is still detected — `notify_all()` alone can't see that.

**Tech Stack:** Python stdlib `ctypes` (DXGI COM bindings, same pattern already used in `app/services/devices_service.py`, zero new dependency there), `psutil` (new dependency, RAM probe), `asyncio.Condition`/`asyncio.wait_for` (Python 3.11+ — this project's floor per `pyproject.toml`; 3.11 is also where `asyncio.Condition.wait()` guarantees lock re-acquisition on cancellation, which this design's timeout-based poll relies on).

## Global Constraints

- Every probe is **fail-open**: any failure (missing dependency, unsupported COM interface, DXGI error, non-Windows platform) returns `None` from `free_capacity_mb`, which `DeviceSemaphores` treats as "unknown, don't gate on this" — never raises, never blocks a job that would otherwise run today. This mirrors the existing contract of `devices_service._enumerate_gpu_adapter_names()` ("Never raises: returns [] on any failure").
- Zero behavior change when `resource_probes` isn't injected (`DeviceSemaphores(settings)` with no second arg): every existing test in `tests/test_device_semaphores.py` must keep passing unmodified.
- No estimation of per-job VRAM/RAM need. This plan implements a generic free-resource **threshold** check only (`MIN_FREE_VRAM_MB` / `MIN_FREE_RAM_MB`), not a per-model-kind size table — see the design spec's "Fases futuras" for why.
- Backend tests: `.venv\Scripts\python.exe -m pytest tests/ -q` (baseline before this plan: **1208 tests collected**, all passing). Run the single-file command shown in each task first, then the full suite at the end of every task that touches shared files (`config.py`, `device_semaphores.py`, `devices_service.py`, `main.py`).
- Commit messages in Spanish where the file being touched already uses Spanish comments (matches the rest of the codebase's convention — see e.g. `device_semaphores.py`'s own docstring, which is in English, vs. `config.py`'s inline comments, which are in Spanish; follow whichever the specific file already uses).
- Branch: `feature/gpu-capacity-admission` (repo's pre-commit hook enforces the `feature/` prefix).
- SDD ledger: add your own section to `.superpowers/sdd/progress.md` under a new `--- feature/gpu-capacity-admission ---` heading — do not touch other plans' entries there.
- This plan does **not** implement a real NPU probe (no hardware available to validate against) — `NullProbe` is the deliberate, final MVP state for `npu`, not a placeholder to fill in later within this plan.

---

## File Structure

**New files:**

| File | Responsibility |
|---|---|
| `app/services/resource_probes.py` | `ResourceProbe` protocol + `NullProbe`, `SystemRamProbe`, `DxgiVramProbe`. |
| `tests/test_resource_probes.py` | Unit tests for all three probes (fakes/monkeypatched seams only — no real hardware needed to pass CI). |

**Modified files:** `app/services/devices_service.py` (+DXGI `IDXGIAdapter3::QueryVideoMemoryInfo` bindings, new module-level seam `_query_adapter_free_vram_mb`), `app/services/device_semaphores.py` (+`resource_probes` param, +threshold check, +periodic poll wait loop), `app/config.py` (+`MIN_FREE_VRAM_MB`/`MIN_FREE_RAM_MB`/`RESOURCE_POLL_INTERVAL_SECONDS`), `app/main.py` (wiring), `pyproject.toml` (+`psutil`), `tests/test_device_semaphores.py` (+resource-gating tests, +new settings validator tests), `tests/test_devices.py` (+seam fail-open test), `.env.example` + `README.md` (docs for the 3 new env vars).

---

## Task 1: `resource_probes.py` — `ResourceProbe` protocol + `NullProbe`

**Files:**
- Create: `app/services/resource_probes.py`
- Test: `tests/test_resource_probes.py`

**Interfaces:**
- Produces: `ResourceProbe` (Protocol: `free_capacity_mb(self, device_id: str) -> int | None`), `NullProbe` — consumed by Tasks 2, 4, 6, 7.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resource_probes.py
from __future__ import annotations

from app.services.resource_probes import NullProbe


def test_null_probe_always_returns_none_for_any_device_id() -> None:
    probe = NullProbe()

    assert probe.free_capacity_mb("npu:0") is None
    assert probe.free_capacity_mb("dml:0") is None
    assert probe.free_capacity_mb("cpu") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_resource_probes.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.resource_probes'`

- [ ] **Step 3: Write the minimal implementation**

```python
# app/services/resource_probes.py
from __future__ import annotations

from typing import Protocol


class ResourceProbe(Protocol):
    """A read-only probe of free capacity (in MB) for one device kind.

    `None` means "can't measure this right now" -- callers (DeviceSemaphores)
    treat that as fail-open: the resource check is skipped entirely for that
    call, never blocking a job that today's job-count-only gate would admit.
    """

    def free_capacity_mb(self, device_id: str) -> int | None: ...


class NullProbe:
    """Extension point for device kinds with no real probe yet (npu today,
    and any future kind on a platform this project doesn't target yet).
    Always fail-open -- DeviceSemaphores behaves exactly as if no probe were
    registered for that kind at all."""

    def free_capacity_mb(self, device_id: str) -> int | None:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_resource_probes.py -q`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add app/services/resource_probes.py tests/test_resource_probes.py
git commit -m "feat: add ResourceProbe protocol and NullProbe stub"
```

---

## Task 2: `SystemRamProbe` (real, `psutil`)

**Files:**
- Modify: `app/services/resource_probes.py`
- Modify: `pyproject.toml`
- Test: `tests/test_resource_probes.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SystemRamProbe`, module-level seam `_probe_psutil_available_mb() -> int | None` — consumed by Task 6 (wiring), and by this task's own tests (monkeypatch seam).

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add this line to the `dependencies` array, right after the `diffusers` line (last entry before the closing `]`):

```toml
  # Subproyecto B (admision por capacidad, ver docs/superpowers/specs/
  # 2026-07-24-gpu-capacity-admission-design.md) - RAM libre real para
  # SystemRamProbe (app/services/resource_probes.py), jobs en cpu.
  "psutil>=6.0.0,<7.0.0",
```

- [ ] **Step 2: Install it**

Run: `.venv\Scripts\python.exe -m pip install "psutil>=6.0.0,<7.0.0"`
Expected: successful install, `psutil.__version__` importable.

- [ ] **Step 3: Write the failing tests**

Append to `tests/test_resource_probes.py`:

```python
import sys

import pytest

import app.services.resource_probes as resource_probes


def test_system_ram_probe_returns_available_mb_from_psutil_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resource_probes, "_probe_psutil_available_mb", lambda: 4096)
    probe = resource_probes.SystemRamProbe()

    assert probe.free_capacity_mb("cpu") == 4096


def test_system_ram_probe_returns_none_when_seam_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resource_probes, "_probe_psutil_available_mb", lambda: None)
    probe = resource_probes.SystemRamProbe()

    assert probe.free_capacity_mb("cpu") is None


def test_probe_psutil_available_mb_survives_missing_psutil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same technique test_devices.py already uses for onnxruntime: a None
    # entry in sys.modules makes `import psutil` raise ImportError even
    # though the real package is installed in this venv.
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert resource_probes._probe_psutil_available_mb() is None
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_resource_probes.py -q`
Expected: FAIL — `AttributeError: module 'app.services.resource_probes' has no attribute '_probe_psutil_available_mb'` (and `SystemRamProbe` undefined)

- [ ] **Step 5: Write the implementation**

Append to `app/services/resource_probes.py`:

```python
def _probe_psutil_available_mb() -> int | None:
    # Lazy + tolerant import, same pattern as devices_service._probe_onnxruntime:
    # keeps the app/tests working even if psutil is somehow absent (fresh
    # checkout not yet pip install-ed, trimmed embedded-Python build).
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return int(psutil.virtual_memory().available // (1024 * 1024))


class SystemRamProbe:
    """Real system RAM probe for cpu-kind devices."""

    def free_capacity_mb(self, device_id: str) -> int | None:
        return _probe_psutil_available_mb()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_resource_probes.py -q`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add app/services/resource_probes.py pyproject.toml tests/test_resource_probes.py
git commit -m "feat: add SystemRamProbe (psutil) for cpu-kind resource admission"
```

---

## Task 3: `devices_service.py` — live VRAM query via `IDXGIAdapter3::QueryVideoMemoryInfo`

**Files:**
- Modify: `app/services/devices_service.py`
- Test: `tests/test_devices.py`

**Interfaces:**
- Consumes: existing DXGI plumbing in this file (`_create_dxgi_factory1`, `_com_vtable`, `_release_com`, `_next_adapter`, `_hardware_adapter_name`, `_EnumAdapters1Proto`, `_ENUM_ADAPTERS1_VTABLE_INDEX`, `_IDXGIFACTORY1_VTABLE_SIZE`, `_IDXGIADAPTER1_VTABLE_SIZE`, `_Guid`).
- Produces: module-level seam `_query_adapter_free_vram_mb(adapter_index: int) -> int | None` — consumed by Task 4 (`DxgiVramProbe`).

**Important:** `DedicatedVideoMemory` (already read by `_hardware_adapter_name`'s sibling code via `_DxgiAdapterDesc1`) is TOTAL adapter capacity, static per-GPU-model. `QueryVideoMemoryInfo` is a **different COM interface** (`IDXGIAdapter3`, not `IDXGIAdapter1`) that returns LIVE `Budget`/`CurrentUsage` — this is the one that reflects VRAM used by processes outside this app (games, other GPU-heavy software). Getting the vtable slot index wrong here is a real risk (calling the wrong function pointer with a mismatched signature is undefined behavior at the ABI level) — Step 7 below is a **mandatory manual hardware verification**, not optional polish; do not skip it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_devices.py`:

```python
def test_query_adapter_free_vram_mb_fails_open_when_dxgi_query_raises_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_oserror(adapter_index: int) -> int | None:
        raise OSError("simulated DXGI failure")

    monkeypatch.setattr(devices_service, "_query_adapter_free_vram_mb_dxgi", raise_oserror)

    assert devices_service._query_adapter_free_vram_mb(0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_devices.py -k free_vram_mb -q`
Expected: FAIL with `AttributeError: module 'app.services.devices_service' has no attribute '_query_adapter_free_vram_mb_dxgi'`

- [ ] **Step 3: Write the implementation**

Insert into `app/services/devices_service.py`, right after the existing `_enumerate_dxgi_adapter_names()` function (which ends the current DXGI section) and before `_build_gpu_device`:

```python
DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0
_IID_IDXGI_ADAPTER3 = _Guid(
    0x645967A4, 0x1392, 0x4310, (ctypes.c_ubyte * 8)(0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD)
)

# IDXGIAdapter3 vtable = IUnknown(3) + IDXGIObject(4: SetPrivateData,
# SetPrivateDataInterface, GetPrivateData, GetParent) + IDXGIAdapter(3:
# EnumOutputs, GetDesc, CheckInterfaceSupport) + IDXGIAdapter1(1: GetDesc1)
# + IDXGIAdapter2(1: GetDesc2) + IDXGIAdapter3-new(6: RegisterHardware...,
# UnregisterHardware..., QueryVideoMemoryInfo, SetVideoMemoryReservation,
# RegisterVideoMemoryBudgetChangeNotificationEvent,
# UnregisterVideoMemoryBudgetChangeNotification) = 18 slots, 0-indexed.
_IDXGIADAPTER3_VTABLE_SIZE = 18
_QUERY_INTERFACE_VTABLE_INDEX = 0
_QUERY_VIDEO_MEMORY_INFO_VTABLE_INDEX = 14


class _DxgiQueryVideoMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
    ]


_QueryInterfaceProto = ctypes.WINFUNCTYPE(
    ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p)
)
_QueryVideoMemoryInfoProto = ctypes.WINFUNCTYPE(
    ctypes.c_long, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(_DxgiQueryVideoMemoryInfo),
)


def _hardware_adapter_ptr_by_index(
    factory_ptr: ctypes.c_void_p, enum_adapters1, target_index: int
) -> ctypes.c_void_p | None:
    """Same hardware-only, hybrid-not-deduped enumeration order as
    _iter_hardware_adapter_names / _enumerate_gpu_devices (dml:N == the Nth
    hardware adapter found, never renumbered). Caller owns (must Release)
    the returned pointer; every skipped adapter is Released here."""
    hardware_index = 0
    index = 0
    while True:
        adapter_ptr = _next_adapter(factory_ptr, enum_adapters1, index)
        if adapter_ptr is None:
            return None
        index += 1
        name = _hardware_adapter_name(adapter_ptr)
        if name is None:
            _release_com(adapter_ptr, _IDXGIADAPTER1_VTABLE_SIZE)
            continue
        if hardware_index == target_index:
            return adapter_ptr
        hardware_index += 1
        _release_com(adapter_ptr, _IDXGIADAPTER1_VTABLE_SIZE)


def _query_video_memory_info_mb(adapter1_ptr: ctypes.c_void_p) -> int | None:
    adapter1_vtable = _com_vtable(adapter1_ptr.value, _IDXGIADAPTER1_VTABLE_SIZE)
    query_interface = _QueryInterfaceProto(adapter1_vtable[_QUERY_INTERFACE_VTABLE_INDEX])
    adapter3_ptr = ctypes.c_void_p()
    result = query_interface(adapter1_ptr, byref(_IID_IDXGI_ADAPTER3), byref(adapter3_ptr))
    if result != 0 or not adapter3_ptr.value:
        return None
    try:
        adapter3_vtable = _com_vtable(adapter3_ptr.value, _IDXGIADAPTER3_VTABLE_SIZE)
        query_video_memory_info = _QueryVideoMemoryInfoProto(
            adapter3_vtable[_QUERY_VIDEO_MEMORY_INFO_VTABLE_INDEX]
        )
        info = _DxgiQueryVideoMemoryInfo()
        result = query_video_memory_info(adapter3_ptr, 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, byref(info))
        if result != 0:
            return None
        # Budget can transiently dip below CurrentUsage under real pressure
        # (documented DXGI behavior) -- clamp instead of returning negative.
        free_bytes = max(0, info.Budget - info.CurrentUsage)
        return int(free_bytes // (1024 * 1024))
    finally:
        _release_com(adapter3_ptr, _IDXGIADAPTER3_VTABLE_SIZE)


def _query_adapter_free_vram_mb_dxgi(adapter_index: int) -> int | None:
    factory_ptr = _create_dxgi_factory1()
    try:
        factory_vtable = _com_vtable(factory_ptr.value, _IDXGIFACTORY1_VTABLE_SIZE)
        enum_adapters1 = _EnumAdapters1Proto(factory_vtable[_ENUM_ADAPTERS1_VTABLE_INDEX])
        adapter1_ptr = _hardware_adapter_ptr_by_index(factory_ptr, enum_adapters1, adapter_index)
        if adapter1_ptr is None:
            return None
        try:
            return _query_video_memory_info_mb(adapter1_ptr)
        finally:
            _release_com(adapter1_ptr, _IDXGIADAPTER1_VTABLE_SIZE)
    finally:
        _release_com(factory_ptr, _IDXGIFACTORY1_VTABLE_SIZE)


def _query_adapter_free_vram_mb(adapter_index: int) -> int | None:
    """Live free VRAM (Budget - CurrentUsage) in MB for the hardware adapter
    at `adapter_index` (0-based, same order dml:N ids use). Never raises:
    non-Windows, missing DXGI, unsupported IDXGIAdapter3, or any COM error
    all fall back to None -- same never-raises contract as
    _enumerate_gpu_adapter_names above."""
    if sys.platform != "win32":
        return None
    try:
        return _query_adapter_free_vram_mb_dxgi(adapter_index)
    except OSError:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_devices.py -k free_vram_mb -q`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full existing test_devices.py file to confirm no regression**

Run: `.venv\Scripts\python.exe -m pytest tests/test_devices.py -q`
Expected: PASS, all previously-passing tests plus the 1 new one.

- [ ] **Step 6: Run the full backend suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, 1208 + (this task's new tests) — no existing test calls the new functions, so nothing else should be affected.

- [ ] **Step 7: MANDATORY manual smoke on real hardware (not automated, not CI)**

On a machine with a real DirectML GPU (matches the project's existing SP1-T8/GMFSS smoke-gate convention for exactly this class of "raw COM ABI, can't be safely unit-tested" risk):

```python
# run via: .venv\Scripts\python.exe -c "..."
from app.services.devices_service import _query_adapter_free_vram_mb
print(_query_adapter_free_vram_mb(0))
```

Cross-check the printed MB value against Task Manager → Performance → GPU → "Dedicated GPU memory usage" (or `nvidia-smi`/`GPU-Z` if available) for the same adapter — they should be in the same ballpark (Budget can differ slightly from Task Manager's usage graph, but not by an order of magnitude). Then launch a GPU-heavy app (a game, or anything that visibly moves Task Manager's GPU memory graph) and re-run the script — the printed free-MB number must drop, proving this reads LIVE external usage, not just this app's own allocations. If the call returns `None` on real hardware where DXGI adapter enumeration already works (i.e. `list_devices()` shows a `dml:N` entry), STOP — the vtable index or GUID is wrong, do not proceed to Task 4 until fixed.

- [ ] **Step 8: Commit**

```bash
git add app/services/devices_service.py tests/test_devices.py
git commit -m "feat: add live VRAM query via IDXGIAdapter3::QueryVideoMemoryInfo"
```

---

## Task 4: `DxgiVramProbe` — wraps the devices_service seam

**Files:**
- Modify: `app/services/resource_probes.py`
- Test: `tests/test_resource_probes.py`

**Interfaces:**
- Consumes: `devices_service._query_adapter_free_vram_mb` (Task 3).
- Produces: `DxgiVramProbe` — consumed by Task 7 (wiring).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_resource_probes.py`:

```python
import app.services.devices_service as devices_service


def test_dxgi_vram_probe_delegates_to_devices_service_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        devices_service, "_query_adapter_free_vram_mb", lambda index: 2048 if index == 1 else None
    )
    probe = resource_probes.DxgiVramProbe()

    assert probe.free_capacity_mb("dml:1") == 2048
    assert probe.free_capacity_mb("dml:0") is None


def test_dxgi_vram_probe_returns_none_for_non_dml_device_id() -> None:
    probe = resource_probes.DxgiVramProbe()

    assert probe.free_capacity_mb("cpu") is None
    assert probe.free_capacity_mb("npu:0") is None


def test_dxgi_vram_probe_returns_none_for_malformed_dml_device_id() -> None:
    probe = resource_probes.DxgiVramProbe()

    assert probe.free_capacity_mb("dml:not-a-number") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_resource_probes.py -k DxgiVramProbe -q`
Expected: FAIL with `AttributeError: module 'app.services.resource_probes' has no attribute 'DxgiVramProbe'`

- [ ] **Step 3: Write the implementation**

Append to `app/services/resource_probes.py`:

```python
from app.services import devices_service

_DML_DEVICE_PREFIX = "dml:"


class DxgiVramProbe:
    """Real VRAM probe for gpu-kind devices (dml:N), Windows/DirectML only.
    Thin adapter: parses the device_id and delegates the actual DXGI query
    to devices_service, which owns the ctypes/COM bindings."""

    def free_capacity_mb(self, device_id: str) -> int | None:
        if not device_id.startswith(_DML_DEVICE_PREFIX):
            return None
        try:
            adapter_index = int(device_id[len(_DML_DEVICE_PREFIX):])
        except ValueError:
            return None
        return devices_service._query_adapter_free_vram_mb(adapter_index)
```

Add the `from app.services import devices_service` import at the top of the file with the other imports (not inline) — move it up next to `from typing import Protocol`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_resource_probes.py -q`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add app/services/resource_probes.py tests/test_resource_probes.py
git commit -m "feat: add DxgiVramProbe wrapping the devices_service VRAM seam"
```

---

## Task 5: `config.py` — `MIN_FREE_VRAM_MB` / `MIN_FREE_RAM_MB` / `RESOURCE_POLL_INTERVAL_SECONDS`

**Files:**
- Modify: `app/config.py`
- Test: `tests/test_device_semaphores.py`

**Interfaces:**
- Produces: `Settings.min_free_vram_mb: int`, `Settings.min_free_ram_mb: int`, `Settings.resource_poll_interval_seconds: float` — consumed by Task 6.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_device_semaphores.py`:

```python
@pytest.mark.parametrize("field_alias", ["MIN_FREE_VRAM_MB", "MIN_FREE_RAM_MB"])
def test_min_free_mb_settings_reject_negative_values(tmp_path: Path, field_alias: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(tmp_path, **{field_alias: -1})


def test_min_free_mb_settings_accept_zero_as_no_floor(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, MIN_FREE_VRAM_MB=0, MIN_FREE_RAM_MB=0)

    assert settings.min_free_vram_mb == 0
    assert settings.min_free_ram_mb == 0


def test_min_free_mb_settings_have_documented_defaults(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    assert settings.min_free_vram_mb == 768
    assert settings.min_free_ram_mb == 1024


def test_resource_poll_interval_rejects_non_positive_values(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        make_settings(tmp_path, RESOURCE_POLL_INTERVAL_SECONDS=0)


def test_resource_poll_interval_default(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    assert settings.resource_poll_interval_seconds == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_device_semaphores.py -k "min_free or resource_poll" -q`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'min_free_vram_mb'` (and the negative/zero tests fail because no `ValidationError` is raised for an unknown-but-accepted kwarg — pydantic-settings rejects unknown fields by default in this project's `Settings`, so these actually fail with a different pydantic error; either way, RED)

- [ ] **Step 3: Write the implementation**

In `app/config.py`, add these three fields right after `capability_fix_timeout_seconds` (the last field before the `@field_validator` methods, currently line 425):

```python
    # Subproyecto B (admision por capacidad, ver docs/superpowers/specs/
    # 2026-07-24-gpu-capacity-admission-design.md): piso minimo de recurso
    # libre para admitir un job nuevo en el device seleccionado. 0 = sin piso
    # (equivale al comportamiento de antes de este subproyecto: solo conteo).
    min_free_vram_mb: int = Field(default=768, alias="MIN_FREE_VRAM_MB")
    min_free_ram_mb: int = Field(default=1024, alias="MIN_FREE_RAM_MB")
    # Cada cuanto se re-evalua el predicado de recursos mientras un job
    # espera gateado por VRAM/RAM, ademas del notify_all() que ya dispara un
    # release() propio -- necesario porque liberar memoria por un proceso
    # EXTERNO a Upflow no dispara ningun notify interno.
    resource_poll_interval_seconds: float = Field(default=5.0, alias="RESOURCE_POLL_INTERVAL_SECONDS")
```

Add these two validators right after `_validate_concurrency_at_least_one`:

```python
    @field_validator("min_free_vram_mb", "min_free_ram_mb")
    @classmethod
    def _validate_min_free_mb_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("MIN_FREE_VRAM_MB and MIN_FREE_RAM_MB must be >= 0")
        return value

    @field_validator("resource_poll_interval_seconds")
    @classmethod
    def _validate_resource_poll_interval_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("RESOURCE_POLL_INTERVAL_SECONDS must be greater than 0")
        return value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_device_semaphores.py -k "min_free or resource_poll" -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full backend suite to confirm no regression**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS — no existing test references these new fields, so nothing else should be affected.

- [ ] **Step 6: Commit**

```bash
git add app/config.py tests/test_device_semaphores.py
git commit -m "feat: add MIN_FREE_VRAM_MB/MIN_FREE_RAM_MB/RESOURCE_POLL_INTERVAL_SECONDS settings"
```

---

## Task 6: `device_semaphores.py` — resource-aware admission + periodic poll

**Files:**
- Modify: `app/services/device_semaphores.py`
- Test: `tests/test_device_semaphores.py`

**Interfaces:**
- Consumes: `ResourceProbe` (Task 1), `Settings.min_free_vram_mb`/`min_free_ram_mb`/`resource_poll_interval_seconds` (Task 5).
- Produces: `DeviceSemaphores.__init__(self, settings: Settings, resource_probes: dict[str, ResourceProbe] | None = None)` — consumed by Task 7 (`main.py` wiring). Behavior of `acquire`/`in_flight`/`capacity_for`/`free_capacity`/`reserve`/`release` is unchanged in signature; only the internal admission predicate changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_device_semaphores.py`:

```python
class ConstantProbe:
    def __init__(self, value: int | None) -> None:
        self._value = value

    def free_capacity_mb(self, device_id: str) -> int | None:
        return self._value


class SequenceProbe:
    """Returns each value in `values` in order, repeating the last one once
    exhausted -- simulates external pressure resolving over time without any
    job of ours ever calling release()."""

    def __init__(self, values: list[int | None]) -> None:
        self._values = values
        self._calls = 0

    def free_capacity_mb(self, device_id: str) -> int | None:
        value = self._values[min(self._calls, len(self._values) - 1)]
        self._calls += 1
        return value


async def test_acquire_waits_when_capacity_ok_but_vram_below_threshold(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, PER_DEVICE_GPU_CONCURRENCY=2, MIN_FREE_VRAM_MB=1024,
        RESOURCE_POLL_INTERVAL_SECONDS=0.05,
    )
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": ConstantProbe(512)})

    async def try_acquire() -> None:
        async with semaphores.acquire("dml:0"):
            pass

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(try_acquire(), timeout=0.3)


async def test_acquire_admits_once_external_vram_pressure_clears(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, PER_DEVICE_GPU_CONCURRENCY=2, MIN_FREE_VRAM_MB=1024,
        RESOURCE_POLL_INTERVAL_SECONDS=0.05,
    )
    probe = SequenceProbe([256, 256, 256, 2048])
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": probe})

    async def acquire_and_check() -> None:
        async with semaphores.acquire("dml:0"):
            assert semaphores.in_flight("dml:0") == 1

    await asyncio.wait_for(acquire_and_check(), timeout=1.0)


async def test_acquire_ignores_resource_probes_for_devices_outside_their_kind(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MIN_FREE_VRAM_MB=999_999)
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": ConstantProbe(0)})

    async def acquire_and_check() -> None:
        async with semaphores.acquire("cpu"):
            assert semaphores.in_flight("cpu") == 1

    await asyncio.wait_for(acquire_and_check(), timeout=0.5)


async def test_acquire_behavior_unchanged_when_no_resource_probes_injected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)

    async def acquire_and_check() -> None:
        async with semaphores.acquire("dml:0"):
            assert semaphores.in_flight("dml:0") == 1

    await asyncio.wait_for(acquire_and_check(), timeout=0.5)


async def test_acquire_treats_probe_none_as_fail_open(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MIN_FREE_VRAM_MB=999_999)
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": ConstantProbe(None)})

    async def acquire_and_check() -> None:
        async with semaphores.acquire("dml:0"):
            assert semaphores.in_flight("dml:0") == 1

    await asyncio.wait_for(acquire_and_check(), timeout=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_device_semaphores.py -k "vram or resource_probes or fail_open" -q`
Expected: FAIL — `TypeError: DeviceSemaphores.__init__() got an unexpected keyword argument 'resource_probes'`

- [ ] **Step 3: Write the implementation**

Replace the full contents of `app/services/device_semaphores.py` with:

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import Settings
from app.services.devices_service import CPU_DEVICE_ID
from app.services.resource_probes import ResourceProbe

_GPU_DEVICE_PREFIX = "dml:"
_NPU_DEVICE_PREFIX = "npu:"


def _device_kind(device_id: str | None) -> str | None:
    if device_id == CPU_DEVICE_ID:
        return "cpu"
    if device_id is not None and device_id.startswith(_GPU_DEVICE_PREFIX):
        return "gpu"
    if device_id is not None and device_id.startswith(_NPU_DEVICE_PREFIX):
        return "npu"
    return None


class DeviceSemaphores:
    """Per-device concurrency gate backed by a single shared condition.

    Each device_id (e.g. "cpu", "dml:0", "dml:1") has a configured capacity
    (`capacity_for`) and a live `in_flight` count. A job for a device only
    runs while `in_flight < capacity` for that device; jobs on different
    devices never gate each other. All accounting lives under ONE shared
    `asyncio.Condition`, and every release does `notify_all()`, so a waiter
    blocked on capacity for device X wakes the instant ANY device frees a
    permit -- crucially including a job releasing a device the waiter is not
    even interested in. That single release signal is what lets the
    auto-router (app/services/device_router.py) re-select the first device
    that frees instead of staying parked on a stale pick (no idle-device
    head-of-line blocking, no busy-wait).

    Implemented with manual counting + a condition rather than one
    asyncio.Semaphore per device precisely so that non-blocking "reserve iff
    free" (needed by the router's atomic pick) and the shared release signal
    share the exact same in_flight ledger and capacity limit as the blocking
    `acquire()` used by pinned-device jobs.

    Subproyecto B (ver docs/superpowers/specs/2026-07-24-gpu-capacity-
    admission-design.md): admission ALSO consults an optional
    `resource_probes` registry (device kind -> ResourceProbe), checking real
    free VRAM/RAM before granting a permit, on top of the job-count check
    above. A probe returning None (or no probe registered for a kind) is
    fail-open -- identical behavior to not having this feature at all. The
    wait loop polls periodically (not just on notify_all()) because a
    resource freed by a process OUTSIDE this app never triggers our own
    release()'s notify.
    """

    def __init__(
        self, settings: Settings, resource_probes: dict[str, ResourceProbe] | None = None
    ) -> None:
        self._settings = settings
        self._in_flight: dict[str | None, int] = {}
        self._condition = asyncio.Condition()
        self._resource_probes = resource_probes or {}
        self._min_free_mb: dict[str, int] = {
            "gpu": settings.min_free_vram_mb,
            "cpu": settings.min_free_ram_mb,
        }

    def capacity_for(self, device_id: str | None) -> int:
        if device_id == CPU_DEVICE_ID:
            return self._settings.cpu_concurrency
        return self._settings.per_device_gpu_concurrency

    def in_flight(self, device_id: str | None) -> int:
        return self._in_flight.get(device_id, 0)

    def free_capacity(self, device_id: str | None) -> int:
        return self.capacity_for(device_id) - self.in_flight(device_id)

    @property
    def release_condition(self) -> asyncio.Condition:
        """The shared condition the auto-router waits on for a freed permit.

        A caller reserving through it MUST hold it while checking
        `free_capacity` and calling `reserve`, so the check-and-take is
        atomic against every other acquirer under asyncio's cooperative
        scheduling.
        """
        return self._condition

    def reserve(self, device_id: str | None) -> None:
        """Take one permit unconditionally. Caller MUST hold release_condition
        AND have already confirmed `free_capacity(device_id) > 0` under it."""
        self._in_flight[device_id] = self.in_flight(device_id) + 1

    def _has_enough_resources(self, device_id: str | None) -> bool:
        kind = _device_kind(device_id)
        if kind is None:
            return True
        probe = self._resource_probes.get(kind)
        if probe is None:
            return True
        free_mb = probe.free_capacity_mb(device_id)
        if free_mb is None:
            return True
        threshold = self._min_free_mb.get(kind)
        if threshold is None:
            return True
        return free_mb >= threshold

    def _reserve_if_free(self, device_id: str | None) -> bool:
        if self.free_capacity(device_id) <= 0:
            return False
        if not self._has_enough_resources(device_id):
            return False
        self.reserve(device_id)
        return True

    async def release(self, device_id: str | None) -> None:
        async with self._condition:
            self._in_flight[device_id] = self.in_flight(device_id) - 1
            # Wake every waiter (pinned acquirers AND auto-router pickers): the
            # freed device may be exactly what an otherwise-idle picker was
            # waiting for, even if it's not the device it originally eyed.
            self._condition.notify_all()

    @asynccontextmanager
    async def acquire(self, device_id: str | None) -> AsyncIterator[None]:
        # Blocking reserve for a pinned device: wait until this specific
        # device has a free permit AND (if a resource probe applies) enough
        # free VRAM/RAM, take it, then run the job WITHOUT holding the
        # condition (it's a leaf lock, held only during reserve/release).
        #
        # The wait uses a bounded asyncio.wait_for(...) around each
        # condition.wait() instead of one plain wait_for(predicate) so the
        # predicate gets re-evaluated periodically even when nothing calls
        # notify_all() -- required for resource pressure that clears for
        # reasons outside this app. This relies on asyncio.Condition.wait()
        # re-acquiring its lock even when cancelled by the outer timeout,
        # guaranteed since Python 3.11 (this project's floor).
        async with self._condition:
            while not self._reserve_if_free(device_id):
                try:
                    await asyncio.wait_for(
                        self._condition.wait(), timeout=self._settings.resource_poll_interval_seconds
                    )
                except TimeoutError:
                    pass
        try:
            yield
        finally:
            await self.release(device_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_device_semaphores.py -q`
Expected: PASS, all previously-passing tests in this file plus the 5 new ones (no signature/behavior change for callers that never pass `resource_probes`).

- [ ] **Step 5: Run the full backend suite to confirm no regression**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS — `DeviceSemaphores(settings)` (no second arg) is still valid everywhere it's constructed today (`app/main.py`, other tests), and `resource_probes` defaults to `{}`.

- [ ] **Step 6: Commit**

```bash
git add app/services/device_semaphores.py tests/test_device_semaphores.py
git commit -m "feat: gate DeviceSemaphores admission on real free VRAM/RAM, not just job count"
```

---

## Task 7: `main.py` — wire the probe registry

**Files:**
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `DxgiVramProbe`, `SystemRamProbe` (Tasks 2, 4), `DeviceSemaphores.__init__`'s new `resource_probes` param (Task 6).
- Produces: nothing new for other tasks — this is the final wiring point.

- [ ] **Step 1: Add the import**

In `app/main.py`, add this import line alphabetically among the existing `app.services.*` imports — right after `from app.services.onnx_cpu_fallback_probe import OnnxCpuFallbackProbe` and before `from app.services.restorer_registry import build_restorers`:

```python
from app.services.resource_probes import DxgiVramProbe, SystemRamProbe
```

- [ ] **Step 2: Wire the registry into `DeviceSemaphores`**

In the `lifespan` function, replace:

```python
    device_semaphores = DeviceSemaphores(settings)
```

with:

```python
    # Subproyecto B: real VRAM/RAM admission on top of the existing
    # job-count gate. "npu" has no real probe yet (no NPU enumeration story
    # in devices_service.py) -- omitting it from this dict is equivalent to
    # registering a NullProbe, both fail-open.
    resource_probes = {"gpu": DxgiVramProbe(), "cpu": SystemRamProbe()}
    device_semaphores = DeviceSemaphores(settings, resource_probes=resource_probes)
```

- [ ] **Step 3: Run the full backend suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS — no test constructs `app.main`'s `lifespan` `device_semaphores` directly with assertions on its probes; this only needs to not break app startup, which `tests/test_devices.py`'s `TestClient(app)` usage already exercises.

- [ ] **Step 4: Start the server and confirm it boots**

Run: `.venv\Scripts\python.exe -m uvicorn app.main:app --port 8091` (a throwaway port, so it doesn't collide with a real running instance), then `curl http://127.0.0.1:8091/api/v1/devices` in another shell, confirm it returns 200 with the device list, then stop the server (Ctrl+C).
Expected: server boots without exceptions, `/api/v1/devices` responds normally (this wiring change doesn't touch that endpoint's shape, just what `device_semaphores` gates on internally).

- [ ] **Step 5: Commit**

```bash
git add app/main.py
git commit -m "feat: wire DxgiVramProbe/SystemRamProbe into DeviceSemaphores"
```

---

## Task 8: Docs — `.env.example` + `README.md`

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: `.env.example`**

Add right after the existing `PER_DEVICE_GPU_CONCURRENCY`/`CPU_CONCURRENCY`/`MAX_CONCURRENT_JOBS` block:

```
# --- Admision por capacidad real de dispositivo (subproyecto B, ver
# docs/superpowers/specs/2026-07-24-gpu-capacity-admission-design.md) ---
MIN_FREE_VRAM_MB=768                 # VRAM libre minima (MB) para admitir un job GPU nuevo; 0 = sin piso (solo conteo, como antes)
MIN_FREE_RAM_MB=1024                 # RAM libre minima (MB) para admitir un job CPU nuevo; 0 = sin piso
RESOURCE_POLL_INTERVAL_SECONDS=5     # Cada cuanto se re-chequea VRAM/RAM mientras un job espera por presion externa (fuera de Upflow)
```

- [ ] **Step 2: `README.md` — prose section**

In the "Multi-GPU (colas por dispositivo + auto-router opcional)" section, right after the existing `MAX_CONCURRENT_JOBS` bullet (before the "Auto-router opcional" paragraph), add:

```markdown
Además del conteo de jobs, cada permiso también puede exigir recursos libres reales del device antes de otorgarse (`app/services/resource_probes.py`):

- **`MIN_FREE_VRAM_MB`** (default `768`) — VRAM libre mínima para admitir un job GPU nuevo, medida en vivo vía `IDXGIAdapter3::QueryVideoMemoryInfo` (detecta presión de **otras apps**, no solo jobs propios). `0` = sin piso.
- **`MIN_FREE_RAM_MB`** (default `1024`) — RAM libre mínima para admitir un job `cpu` nuevo, vía `psutil`. `0` = sin piso.
- **`RESOURCE_POLL_INTERVAL_SECONDS`** (default `5`) — cada cuánto se re-chequea mientras un job espera gateado por recursos, para detectar presión externa que se libera sola.

Un job nunca falla por esto — si no hay recursos suficientes, espera en cola (igual que por conteo de jobs) hasta que se liberen, propios o ajenos.
```

- [ ] **Step 3: `README.md` — settings table**

In the "Configuración" section's table, add these 3 rows right after the existing `CPU_CONCURRENCY` row:

```markdown
| `MIN_FREE_VRAM_MB` | `768` | VRAM libre mínima (MB) para admitir un job GPU nuevo; `0` = sin piso. Ver [Multi-GPU](#multi-gpu-colas-por-dispositivo--auto-router-opcional) |
| `MIN_FREE_RAM_MB` | `1024` | RAM libre mínima (MB) para admitir un job `cpu` nuevo; `0` = sin piso |
| `RESOURCE_POLL_INTERVAL_SECONDS` | `5` | Cada cuánto se re-chequea VRAM/RAM mientras un job espera por presión externa |
```

- [ ] **Step 4: Commit**

```bash
git add .env.example README.md
git commit -m "docs: documenta MIN_FREE_VRAM_MB/MIN_FREE_RAM_MB/RESOURCE_POLL_INTERVAL_SECONDS"
```

---

## Task 9: Full non-regression pass + end-to-end smoke

**Files:** none (verification-only task; the SDD ledger entry required by this plan's Global Constraints gets written by whichever process executes this plan — subagent-driven-development or executing-plans — as each task actually completes, with real commit hashes and review notes, not scaffolded here in advance).

- [ ] **Step 1: Run the full backend suite one more time**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, 1208 + all tests added in Tasks 1-6 (test_resource_probes.py: 7, test_devices.py: +1, test_device_semaphores.py: +10), zero failures, zero new warnings.

- [ ] **Step 2: Re-run the mandatory Task 3 hardware smoke if it hasn't been re-verified after Task 6/7's changes**

Same script as Task 3 Step 7, but this time going through the full stack: start the real server (`.venv\Scripts\python.exe -m uvicorn app.main:app --port 8090`), set `MIN_FREE_VRAM_MB` artificially high (e.g. `999999`) in `.env`, restart, submit any GPU job via the UI/API, confirm it sits queued (not running) instead of erroring, then lower `MIN_FREE_VRAM_MB` back to a sane value (or `0`), restart, confirm the same job now runs. This is the end-to-end proof that admission actually gates real job execution, not just the unit-level predicate.

Expected: job stays `queued` (never transitions to `running`) while the threshold is unreachably high, then transitions to `running` normally once the threshold is lowered back and the server restarted.
