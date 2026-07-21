from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum

if sys.platform == "win32":
    import winreg

from app.services.process_runner import run_guarded_process

logger = logging.getLogger(__name__)

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
    if not isinstance(entries, list):
        return Lever(lever_id, label, LeverStatus.unavailable, "PCIe link data is not a list", False)
    if not entries:
        return Lever(lever_id, label, LeverStatus.unavailable, "No GPU adapter found to check", False)
    descriptions = []
    downgraded = False
    for entry in entries:
        if not isinstance(entry, dict):
            return Lever(lever_id, label, LeverStatus.unavailable, "PCIe link data entry is malformed", False)
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
        if returncode != 0:
            return Lever(lever_id, label, LeverStatus.unavailable, f"PCIe link probe failed: {stderr.decode(errors='replace')[:200]}", False)
        return parse_pcie_json(stdout.decode(errors="replace"))
    except Exception:  # noqa: BLE001 -- a probe must never raise
        logger.warning("PCIe link probe failed to run", exc_info=True)
        return Lever(lever_id, label, LeverStatus.unavailable, "Could not run the PCIe link probe", False)
