from __future__ import annotations

import ctypes
import sys
from ctypes import byref
from typing import NamedTuple, TypedDict

from app.config import Settings

# ---------------------------------------------------------------------------
# Device contract (stable ids other tasks build on): "cpu", "dml:0", "dml:1",
# "npu:0". NPU detection has no enumeration story yet (no onnxruntime NPU
# execution provider is probed here) -- "npu"/"winml" are reserved in the
# DeviceInfo shape for a future task, not produced by list_devices() today.
# ---------------------------------------------------------------------------

CPU_DEVICE_ID = "cpu"
DML_EXECUTION_PROVIDER = "DmlExecutionProvider"
DXGI_ADAPTER_FLAG_SOFTWARE = 2

_IDXGIFACTORY1_VTABLE_SIZE = 20
_IDXGIADAPTER1_VTABLE_SIZE = 12
_ENUM_ADAPTERS1_VTABLE_INDEX = 12
_GET_DESC1_VTABLE_INDEX = 10
_RELEASE_VTABLE_INDEX = 2


class DeviceInfo(TypedDict):
    id: str
    kind: str  # "cpu" | "gpu" | "npu"
    name: str
    backend: str  # "cpu" | "directml" | "winml"


CPU_DEVICE: DeviceInfo = {"id": CPU_DEVICE_ID, "kind": "cpu", "name": "CPU", "backend": "cpu"}


class OnnxRuntimeProbe(NamedTuple):
    available_providers: list[str]


def _probe_onnxruntime() -> OnnxRuntimeProbe | None:
    # Lazy + tolerant import: onnxruntime-directml is a real dependency going
    # forward, but the app/tests must keep working (cpu-only) if it's absent
    # (fresh checkout not yet `pip install`-ed, trimmed binary build, etc.).
    try:
        import onnxruntime  # type: ignore[import-not-found]
    except (ImportError, OSError):
        # OSError covers native-extension load failures (missing VC++
        # redistributable / DirectML DLLs), not just an uninstalled package.
        return None
    return OnnxRuntimeProbe(available_providers=list(onnxruntime.get_available_providers()))


def _enumerate_gpu_adapter_names() -> list[str]:
    """Real GPU adapter names via DXGI (Windows-only, no extra dependency).

    `get_available_providers()` only tells us the installed onnxruntime
    build has DirectML compiled in -- not how many DirectML-capable GPUs are
    physically present, or their names. That has to come from the OS, so
    this enumerates DXGI adapters directly via ctypes and filters out
    software adapters (e.g. "Microsoft Basic Render Driver").

    Never raises: returns [] on any failure (non-Windows, no DXGI, COM
    error) so callers always have a safe "no GPU" fallback.
    """
    if sys.platform != "win32":
        return []
    try:
        return _enumerate_dxgi_adapter_names()
    except OSError:
        return []


class _DxgiAdapterDesc1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", ctypes.c_int64),
        ("Flags", ctypes.c_uint32),
    ]


class _Guid(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


_IID_IDXGI_FACTORY1 = _Guid(
    0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87)
)

_EnumAdapters1Proto = ctypes.WINFUNCTYPE(
    ctypes.c_long, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)
)
_GetDesc1Proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_DxgiAdapterDesc1))
_ReleaseProto = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)


def _com_vtable(instance_ptr: int, slot_count: int) -> ctypes.Array:
    # A COM object pointer points to a vtable POINTER, not the vtable array
    # itself -- skipping this extra indirection calls garbage addresses
    # (confirmed: raises an access-violation OSError under ctypes.WINFUNCTYPE
    # rather than hard-crashing the process).
    vtable_address = ctypes.cast(instance_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value
    return ctypes.cast(vtable_address, ctypes.POINTER(ctypes.c_void_p * slot_count)).contents


def _create_dxgi_factory1() -> ctypes.c_void_p:
    factory_ptr = ctypes.c_void_p()
    result = ctypes.windll.dxgi.CreateDXGIFactory1(byref(_IID_IDXGI_FACTORY1), byref(factory_ptr))
    if result != 0 or not factory_ptr.value:
        raise OSError(f"CreateDXGIFactory1 failed: 0x{result & 0xFFFFFFFF:08x}")
    return factory_ptr


def _release_com(instance_ptr: ctypes.c_void_p, vtable_size: int) -> None:
    vtable = _com_vtable(instance_ptr.value, vtable_size)
    _ReleaseProto(vtable[_RELEASE_VTABLE_INDEX])(instance_ptr)


def _hardware_adapter_name(adapter_ptr: ctypes.c_void_p) -> str | None:
    vtable = _com_vtable(adapter_ptr.value, _IDXGIADAPTER1_VTABLE_SIZE)
    get_desc1 = _GetDesc1Proto(vtable[_GET_DESC1_VTABLE_INDEX])
    desc = _DxgiAdapterDesc1()
    if get_desc1(adapter_ptr, byref(desc)) != 0:
        return None
    if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE:
        return None
    return desc.Description


def _next_adapter(factory_ptr: ctypes.c_void_p, enum_adapters1, index: int) -> ctypes.c_void_p | None:
    adapter_ptr = ctypes.c_void_p()
    result = enum_adapters1(factory_ptr, index, byref(adapter_ptr))
    if result != 0 or not adapter_ptr.value:
        return None
    return adapter_ptr


def _iter_hardware_adapter_names(factory_ptr: ctypes.c_void_p, enum_adapters1) -> list[str]:
    names: list[str] = []
    index = 0
    while True:
        adapter_ptr = _next_adapter(factory_ptr, enum_adapters1, index)
        if adapter_ptr is None:
            return names
        try:
            name = _hardware_adapter_name(adapter_ptr)
        finally:
            _release_com(adapter_ptr, _IDXGIADAPTER1_VTABLE_SIZE)
        if name is not None:
            names.append(name)
        index += 1


def _enumerate_dxgi_adapter_names() -> list[str]:
    factory_ptr = _create_dxgi_factory1()
    try:
        factory_vtable = _com_vtable(factory_ptr.value, _IDXGIFACTORY1_VTABLE_SIZE)
        enum_adapters1 = _EnumAdapters1Proto(factory_vtable[_ENUM_ADAPTERS1_VTABLE_INDEX])
        return _iter_hardware_adapter_names(factory_ptr, enum_adapters1)
    finally:
        _release_com(factory_ptr, _IDXGIFACTORY1_VTABLE_SIZE)


def _build_gpu_device(index: int, name: str) -> DeviceInfo:
    return {"id": f"dml:{index}", "kind": "gpu", "name": name or f"GPU {index}", "backend": "directml"}


class DevicesService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def list_devices(self) -> list[DeviceInfo]:
        return [CPU_DEVICE, *self._enumerate_gpu_devices()]

    def _enumerate_gpu_devices(self) -> list[DeviceInfo]:
        if not self._directml_supported():
            return []
        return [_build_gpu_device(index, name) for index, name in enumerate(_enumerate_gpu_adapter_names())]

    @staticmethod
    def _directml_supported() -> bool:
        probe = _probe_onnxruntime()
        return probe is not None and DML_EXECUTION_PROVIDER in probe.available_providers

    @staticmethod
    def _find_device(devices: list[DeviceInfo], device_id: str) -> DeviceInfo | None:
        return next((device for device in devices if device["id"] == device_id), None)

    def validate(self, device_id: str) -> DeviceInfo:
        device = self._find_device(self.list_devices(), device_id)
        if device is None:
            raise ValueError(f"Unknown device id: {device_id!r}")
        return device

    def resolve_default(self, devices: list[DeviceInfo] | None = None) -> DeviceInfo:
        # Accepts a precomputed snapshot so one request never enumerates the
        # hardware twice (and the default stays consistent with that list).
        snapshot = devices if devices is not None else self.list_devices()
        return self._find_device(snapshot, self.settings.default_device) or CPU_DEVICE
