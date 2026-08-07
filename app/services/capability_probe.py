from __future__ import annotations

import base64
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

if sys.platform == "win32":
    import winreg

from app.config import Settings
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


_KMTQAITYPE_WDDM_2_7_CAPS = 70
_HWSCH_SUPPORTED_BIT = 1
_HWSCH_ENABLED_BIT = 2


def _query_hags_kernel_caps() -> tuple[bool, bool] | None:
    """(soportado, activado) según el kernel gráfico, o None si no se pudo leer.

    El registro NO alcanza: con HAGS activado por defecto Windows no escribe
    HwSchMode — el valor solo aparece si alguien tocó el toggle. Visto real:
    "registry value not found" en rojo con el panel de Windows diciendo
    Activado. La fuente de verdad es D3DKMT (WDDM 2.7 caps), la misma que usa
    ese panel.
    """
    import ctypes
    from ctypes import wintypes

    class Luid(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class AdapterInfo(ctypes.Structure):
        _fields_ = [
            ("hAdapter", wintypes.UINT),
            ("AdapterLuid", Luid),
            ("NumOfSources", wintypes.ULONG),
            ("bPresentMoveRegionsPreferred", wintypes.BOOL),
        ]

    class EnumAdapters2(ctypes.Structure):
        _fields_ = [("NumAdapters", wintypes.ULONG), ("pAdapters", ctypes.POINTER(AdapterInfo))]

    class Wddm27Caps(ctypes.Structure):
        _fields_ = [("Value", wintypes.UINT)]

    class QueryAdapterInfo(ctypes.Structure):
        _fields_ = [
            ("hAdapter", wintypes.UINT),
            ("Type", ctypes.c_int),
            ("pPrivateDriverData", ctypes.c_void_p),
            ("PrivateDriverDataSize", wintypes.UINT),
        ]

    class CloseAdapter(ctypes.Structure):
        _fields_ = [("hAdapter", wintypes.UINT)]

    try:
        gdi32 = ctypes.WinDLL("gdi32")
        enum = EnumAdapters2()
        if gdi32.D3DKMTEnumAdapters2(ctypes.byref(enum)) != 0 or enum.NumAdapters == 0:
            return None
        adapters = (AdapterInfo * enum.NumAdapters)()
        enum.pAdapters = ctypes.cast(adapters, ctypes.POINTER(AdapterInfo))
        if gdi32.D3DKMTEnumAdapters2(ctypes.byref(enum)) != 0:
            return None
        supported = False
        enabled = False
        for i in range(enum.NumAdapters):
            info = adapters[i]
            caps = Wddm27Caps()
            query = QueryAdapterInfo(
                hAdapter=info.hAdapter,
                Type=_KMTQAITYPE_WDDM_2_7_CAPS,
                pPrivateDriverData=ctypes.cast(ctypes.byref(caps), ctypes.c_void_p),
                PrivateDriverDataSize=ctypes.sizeof(caps),
            )
            status = gdi32.D3DKMTQueryAdapterInfo(ctypes.byref(query))
            gdi32.D3DKMTCloseAdapter(ctypes.byref(CloseAdapter(hAdapter=info.hAdapter)))
            if status != 0:
                continue
            if caps.Value & _HWSCH_SUPPORTED_BIT:
                supported = True
                if caps.Value & _HWSCH_ENABLED_BIT:
                    enabled = True
        return supported, enabled
    except Exception:  # noqa: BLE001 -- un probe jamás puede tumbar nada
        logger.warning("HAGS kernel caps query failed", exc_info=True)
        return None


def probe_hags() -> Lever:
    lever_id, label = "hags", "Hardware-accelerated GPU scheduling"
    if sys.platform != "win32":
        return Lever(lever_id, label, LeverStatus.not_applicable, "Windows only", False)
    caps = _query_hags_kernel_caps()
    if caps is not None:
        supported, enabled = caps
        if enabled:
            return Lever(lever_id, label, LeverStatus.ok, "Enabled (reported by the graphics kernel)", False)
        if not supported:
            return Lever(
                lever_id, label, LeverStatus.unavailable,
                "GPU/driver does not support hardware scheduling -- informational only", False,
            )
        return Lever(
            lever_id, label, LeverStatus.unavailable,
            "Disabled (graphics kernel reports it off). Measured impact on this workload is ~0 -- informational only.",
            True,
        )
    return _probe_hags_registry(lever_id, label)


def _probe_hags_registry(lever_id: str, label: str) -> Lever:
    # Fallback si D3DKMT no respondió. Un valor AUSENTE no es un estado: es
    # "Windows decide el default", así que no se reporta como apagado.
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, HAGS_KEY_PATH) as key:
            value, _ = winreg.QueryValueEx(key, HAGS_VALUE_NAME)
    except OSError:
        return Lever(
            lever_id, label, LeverStatus.unavailable,
            "No se pudo determinar el estado (sin respuesta del kernel gráfico y sin HwSchMode en el "
            "registro: Windows aplica su default). El estado real está en Configuración > Sistema > "
            "Pantalla > Gráficos.",
            False,
        )
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


_DISK_WRITE_CACHE_SCRIPT_TEMPLATE = """
try {{
    $target = {path_literal}
    $drive = (Get-Item -LiteralPath $target -ErrorAction Stop).PSDrive.Name
    $partition = Get-Partition -DriveLetter $drive -ErrorAction Stop
    $disk = Get-Disk -Number $partition.DiskNumber -ErrorAction Stop
    # Get-Disk's FriendlyName ("AMD-RAID Array 1") is a PREFIX of
    # Get-PnpDevice's FriendlyName ("AMD-RAID Array 1  SCSI Disk Device") on
    # real RAID/Storport-backed disks -- an exact -eq match never fires on
    # that class of hardware (confirmed on a real RX 7800 XT + AMD-RAID
    # machine), silently degrading every such disk to "unavailable" even
    # when the real state IS determinable. -like with a trailing wildcard
    # matches both the plain and suffixed forms.
    $candidates = @(Get-PnpDevice -Class DiskDrive -ErrorAction Stop | Where-Object {{ $_.Status -eq 'OK' -and $_.FriendlyName -like "$($disk.FriendlyName)*" }})
    if (-not $candidates) {{ throw "No matching PnP disk device found for '$($disk.FriendlyName)'" }}
    $pnp = $null
    $regPath = $null
    foreach ($candidate in $candidates) {{
        $candidateRegPath = "HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\$($candidate.InstanceId)\\Device Parameters\\Disk"
        if (Test-Path $candidateRegPath) {{
            $pnp = $candidate
            $regPath = $candidateRegPath
            break
        }}
    }}
    if (-not $pnp) {{
        $pnp = $candidates | Select-Object -First 1
        $regPath = "HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\$($pnp.InstanceId)\\Device Parameters\\Disk"
    }}
    if (-not (Test-Path $regPath)) {{
        # Windows only creates this policy key after the policy is changed.
        # Until then, fixed disks use the Windows default: write caching enabled.
        [PSCustomObject]@{{ ok = $true; diskName = $disk.FriendlyName; writeCacheEnabled = $true; defaultPolicy = $true }} | ConvertTo-Json -Compress
        return
    }}
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

    # Validate that payload is a dict before accessing keys
    if not isinstance(payload, dict):
        return Lever(lever_id, label, LeverStatus.unavailable, "Disk write-cache response is not a valid object", False)

    if not payload.get("ok"):
        return Lever(lever_id, label, LeverStatus.unavailable, str(payload.get("error", "unknown error")), False)

    disk_name = payload.get("diskName", "disk")
    if payload.get("defaultPolicy") is True:
        return Lever(
            lever_id,
            label,
            LeverStatus.ok,
            f"Write caching activo (política default de Windows) en {disk_name}",
            False,
        )

    enabled = bool(payload.get("writeCacheEnabled"))
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


_DEFENDER_EXCLUSIONS_SCRIPT = """
try {
    $prefs = Get-MpPreference -ErrorAction Stop
    [PSCustomObject]@{ ok = $true; exclusions = @($prefs.ExclusionPath) } | ConvertTo-Json -Compress
} catch {
    [PSCustomObject]@{ ok = $false; error = $_.Exception.Message } | ConvertTo-Json -Compress
}
""".strip()


def _normalize_path_for_compare(path: str) -> str:
    # Normalize path separators to backslashes, strip, remove trailing separators, lowercase
    normalized = path.strip().replace("/", "\\").rstrip("\\").lower()
    return normalized


# Estado verificado del fix de Defender. Un probe sin elevación NUNCA puede ver
# la lista de exclusiones (Defender la oculta), así que el estado "aplicado" se
# persiste cuando el script ELEVADO del fix verifica con Get-MpPreference que
# la exclusión quedó. Sin esto, un fix exitoso mostraba "Requiere administrador"
# para siempre (visto real en varias máquinas).
DEFENDER_MARKER_FILENAME = "defender-exclusion-verified.json"
DEFENDER_FIX_VERIFY_FAILED_EXIT = 3


def _defender_marker_path(runtime_path: str) -> Path:
    return Path(runtime_path) / DEFENDER_MARKER_FILENAME


def write_defender_marker(runtime_path: str) -> None:
    payload = {
        "path": runtime_path,
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        _defender_marker_path(runtime_path).write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        logger.warning("No se pudo persistir el marker de exclusión de Defender", exc_info=True)


def read_defender_verified_at(runtime_path: str) -> str | None:
    try:
        payload = json.loads(_defender_marker_path(runtime_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    marker_path = payload.get("path")
    if not isinstance(marker_path, str):
        return None
    if _normalize_path_for_compare(marker_path) != _normalize_path_for_compare(runtime_path):
        return None
    verified_at = payload.get("verified_at")
    return verified_at if isinstance(verified_at, str) and verified_at else None


def parse_defender_exclusion_json(
    raw_stdout: str, runtime_path: str, verified_at: str | None = None
) -> Lever:
    lever_id, label = "defender_exclusion", "Windows Defender exclusion on runtime/"
    try:
        payload = json.loads(raw_stdout)
    except (json.JSONDecodeError, TypeError):
        return Lever(lever_id, label, LeverStatus.unavailable, "Could not read Defender exclusions", False)

    # Validate that payload is a dict before accessing keys
    if not isinstance(payload, dict):
        return Lever(lever_id, label, LeverStatus.unavailable, "Defender response is not a valid object", False)

    if not payload.get("ok"):
        return Lever(lever_id, label, LeverStatus.needs_admin, str(payload.get("error", "unknown error")), True)

    # Validate that exclusions is a list of strings
    exclusions_raw = payload.get("exclusions", [])
    if not isinstance(exclusions_raw, list):
        return Lever(lever_id, label, LeverStatus.unavailable, "Defender exclusions is not a list", False)

    exclusions = set()
    for p in exclusions_raw:
        if not isinstance(p, str):
            return Lever(lever_id, label, LeverStatus.unavailable, "Defender exclusions contains non-string values", False)
        if p.strip().startswith("N/A:"):
            if verified_at:
                return Lever(
                    lever_id,
                    label,
                    LeverStatus.ok,
                    f"Exclusión aplicada y verificada con elevación ({verified_at}). Defender oculta la "
                    "lista a procesos sin elevación, así que se muestra el último estado verificado.",
                    False,
                )
            return Lever(
                lever_id,
                label,
                LeverStatus.needs_admin,
                "Defender oculta la lista de exclusiones a procesos sin elevación — no se puede verificar el estado "
                "real. El Fix aplica la exclusión y la verifica elevado; al terminar, este estado queda registrado.",
                True,
            )
        if p:
            exclusions.add(_normalize_path_for_compare(p))

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
    return parse_defender_exclusion_json(
        stdout.decode(errors="replace"), runtime_path, read_defender_verified_at(runtime_path)
    )


# ---------------------------------------------------------------------------
# Fix scripts + elevation runner. Unlike the probes above (read-only), these
# mutate machine state, so they always run through an elevated (UAC) child
# process -- the backend itself never runs elevated.
#
# Safety constraint: the disk write-cache fix writes the registry value ONLY.
# It must never call Disable-PnpDevice/Restart-Device or anything else that
# live-cycles a disk device -- the target disk can be the boot volume, and a
# live device reset on it is a real crash/hang risk. The change requires a
# reboot to take effect (see probe_disk_write_cache's detail message), which
# is the intended, safe path.
# ---------------------------------------------------------------------------

_HAGS_FIX_SCRIPT = (
    "Set-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers' "
    "-Name HwSchMode -Value 2"
)

_DISK_WRITE_CACHE_FIX_TEMPLATE = """
$target = {path_literal}
$drive = (Get-Item -LiteralPath $target -ErrorAction Stop).PSDrive.Name
$partition = Get-Partition -DriveLetter $drive -ErrorAction Stop
$disk = Get-Disk -Number $partition.DiskNumber -ErrorAction Stop
# Same -like prefix match as the probe script -- see its comment for why
# -eq never matches on RAID/Storport-backed disks.
$pnp = Get-PnpDevice -Class DiskDrive -ErrorAction Stop | Where-Object {{ $_.FriendlyName -like "$($disk.FriendlyName)*" }} | Select-Object -First 1
if (-not $pnp) {{ throw "No matching PnP disk device found for '$($disk.FriendlyName)'" }}
$regPath = "HKLM:\\SYSTEM\\CurrentControlSet\\Enum\\$($pnp.InstanceId)\\Device Parameters\\Disk"
if (-not (Test-Path $regPath)) {{
    New-Item -Path $regPath -Force | Out-Null
}}
Set-ItemProperty -Path $regPath -Name "UserWriteCacheSetting" -Value 1
""".strip()


# El fix de Defender verifica DENTRO de la misma elevación: Add-MpPreference no
# falla cuando Tamper Protection ignora el alta, y el probe sin elevación no
# puede mirar la lista. exit 0 = verificado aplicado; exit 3 = corrió pero la
# exclusión NO quedó.
_DEFENDER_FIX_TEMPLATE = """
$target = {path_literal}
Add-MpPreference -ExclusionPath $target -ErrorAction Stop
$ex = (Get-MpPreference).ExclusionPath
if ($ex -and @($ex | Where-Object {{ "$_".TrimEnd('\\') -ieq $target.TrimEnd('\\') }}).Count -gt 0) {{ exit 0 }}
exit {verify_failed_exit}
""".strip()


def build_fix_script(lever_id: str, runtime_path: str) -> str:
    if lever_id == "hags":
        return _HAGS_FIX_SCRIPT
    if lever_id == "defender_exclusion":
        return _DEFENDER_FIX_TEMPLATE.format(
            path_literal=_ps_single_quote(runtime_path),
            verify_failed_exit=DEFENDER_FIX_VERIFY_FAILED_EXIT,
        )
    if lever_id == "disk_write_cache":
        return _DISK_WRITE_CACHE_FIX_TEMPLATE.format(path_literal=_ps_single_quote(runtime_path))
    raise ValueError(f"Lever {lever_id!r} has no fix script (not fixable or unknown)")


# Start-Process -Verb RunAs launches the elevated child through ShellExecuteEx,
# which is NOT a child process of this (non-elevated) outer wrapper. If the
# outer run_guarded_process timeout fires first, killing the wrapper does not
# reliably terminate the elevated process or dismiss its UAC prompt, orphaning
# it on the user's desktop, untracked by the backend. So the wrapper itself
# tracks the elevated process (via -PassThru) and kills it on an inner
# deadline that is always shorter than the outer timeout, guaranteeing this
# fires first.
_ELEVATION_WAIT_MARGIN_SECONDS = 5.0
_MIN_ELEVATION_WAIT_SECONDS = 1.0


def _elevation_wait_milliseconds(timeout: float) -> int:
    wait_seconds = max(timeout - _ELEVATION_WAIT_MARGIN_SECONDS, _MIN_ELEVATION_WAIT_SECONDS)
    return int(wait_seconds * 1000)


async def _run_elevated(inner_script: str, timeout: float) -> tuple[int | None, str]:
    """(exit code del proceso elevado | None si ni corrió, mensaje de fallo).

    El exit code se preserva porque los scripts de fix lo usan para reportar
    resultados distintos de "anduvo/no anduvo" (ej. Defender: aplicó pero la
    verificación elevada dice que la exclusión no quedó).
    """
    # -EncodedCommand (base64 UTF-16LE) avoids nested PowerShell quoting bugs
    # that string concatenation into -Command would hit when passing an
    # already-quoted inner script through an outer Start-Process call.
    encoded = base64.b64encode(inner_script.encode("utf-16-le")).decode("ascii")
    wait_ms = _elevation_wait_milliseconds(timeout)
    outer = (
        "$p = Start-Process powershell.exe "
        f"-ArgumentList '-NoProfile','-EncodedCommand','{encoded}' "
        "-Verb RunAs -PassThru; "
        f"if (-not $p.WaitForExit({wait_ms})) {{ "
        "Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; exit 1 "
        "}; exit $p.ExitCode"
    )
    try:
        _, _, returncode = await run_guarded_process(["powershell.exe", "-NoProfile", "-Command", outer], timeout)
    except Exception:  # noqa: BLE001 -- elevation failures must degrade, never raise
        return None, "Elevation failed to run"
    if returncode != 0:
        return returncode, "Elevation was cancelled or failed"
    return returncode, ""


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
        returncode, message = await _run_elevated(script, self.settings.capability_fix_timeout_seconds)
        ok = returncode == 0
        if lever_id == "defender_exclusion":
            if ok:
                # Único momento en que el estado real se conoce: el script
                # elevado acaba de verificarlo con Get-MpPreference.
                write_defender_marker(runtime_path)
            elif returncode == DEFENDER_FIX_VERIFY_FAILED_EXIT:
                message = (
                    "el fix corrió elevado pero la exclusión no quedó aplicada — "
                    "probablemente Tamper Protection o una política la bloquea"
                )
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
