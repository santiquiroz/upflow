from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum

if sys.platform == "win32":
    import winreg

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
        f"Disabled (HwSchMode={value}). Measured impact on this workload is ~0 -- informational only.",
        True,
    )
