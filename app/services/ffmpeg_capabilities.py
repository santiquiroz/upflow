"""Capacidades de la build de ffmpeg para el modo CCTV (spec §4.2 paso 9).

Varios filtros de la lista blanca (`hqdn3d`, `fspp`, `spp`, `pp7`, `uspp`,
`eq`) y el encoder `libx264` solo existen en builds GPL, y `FFMPEG_BINARY`
puede apuntar a un ffmpeg LGPL o del sistema. Se sondea `-version`,
`-buildconf`, `-filters` y `-encoders` una vez por binario (cache por sha256)
y se registran el sha256 y las extensiones de CPU para el informe (§4.6).
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.services.cctv_chain import CCTV_CHAIN, OPEN_GATES, FilterSpec, StepSpec, is_open
from app.services.cctv_ingest import sha256_file

VERSION_ARGS: tuple[str, ...] = ("-version",)
BUILDCONF_ARGS: tuple[str, ...] = ("-hide_banner", "-buildconf")
FILTERS_ARGS: tuple[str, ...] = ("-hide_banner", "-filters")
ENCODERS_ARGS: tuple[str, ...] = ("-hide_banner", "-encoders")
PROBE_TIMEOUT_SECONDS = 30
LEGEND_END = "------"
UNKNOWN_VERSION = "unknown"
GPL_FLAG = "--enable-gpl"
REQUIRED_ENCODERS: tuple[str, ...] = ("ffv1", "libx264")
FILTER_UNAVAILABLE = "cctv.filterUnavailable"

# IDs PF_* de winnt.h que acepta IsProcessorFeaturePresent.
CPU_FEATURES: Mapping[str, int] = MappingProxyType(
    {"sse2": 10, "sse3": 13, "ssse3": 36, "sse4_1": 37, "sse4_2": 38, "avx": 39, "avx2": 40, "avx512f": 41}
)

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

Runner = Callable[[Path, Sequence[str]], str]
FeatureProbe = Callable[[int], Any]


class FfmpegProbeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FfmpegCapabilities:
    binary_sha256: str
    version: str
    configuration: tuple[str, ...]
    filters: frozenset[str]
    encoders: frozenset[str]
    cpu_extensions: tuple[str, ...]

    @property
    def is_gpl(self) -> bool:
        return GPL_FLAG in self.configuration

    def has_filter(self, name: str) -> bool:
        return name in self.filters

    def to_report(self) -> dict[str, Any]:
        return {
            "binarySha256": self.binary_sha256,
            "version": self.version,
            "gpl": self.is_gpl,
            "configuration": list(self.configuration),
            "cpuExtensions": list(self.cpu_extensions),
        }


@dataclass(frozen=True, slots=True)
class UnavailableFilter:
    step_id: str
    filter: str
    missing: tuple[str, ...]
    reason_key: str = FILTER_UNAVAILABLE

    @property
    def message(self) -> str:
        return f"Not available in this ffmpeg build (missing filter: {', '.join(self.missing)})."

    def to_json(self) -> dict[str, Any]:
        return {
            "stepId": self.step_id,
            "filter": self.filter,
            "missing": list(self.missing),
            "reasonKey": self.reason_key,
            "reason": self.message,
        }


def windows_feature_probe() -> FeatureProbe | None:
    if sys.platform != "win32":
        return None
    return ctypes.windll.kernel32.IsProcessorFeaturePresent


def cpu_extensions(is_present: FeatureProbe | None) -> tuple[str, ...]:
    if is_present is None:
        return ()
    return tuple(name for name, feature in CPU_FEATURES.items() if is_present(feature))


def host_cpu_extensions() -> tuple[str, ...]:
    return cpu_extensions(windows_feature_probe())


def run_ffmpeg_text(binary: Path, args: Sequence[str]) -> str:
    command = [str(binary), *args]
    try:
        result = subprocess.run(  # noqa: S603 - argumentos fijos de este modulo
            command, capture_output=True, timeout=PROBE_TIMEOUT_SECONDS, creationflags=_NO_WINDOW
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FfmpegProbeError(f"Could not run {binary} {' '.join(args)}: {exc}") from exc
    if result.returncode != 0:
        raise FfmpegProbeError(f"{binary} {' '.join(args)} failed with exit code {result.returncode}.")
    return result.stdout.decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class ProbeTools:
    run: Runner = run_ffmpeg_text
    hash_file: Callable[[Path], str] = sha256_file
    cpu_extensions: Callable[[], tuple[str, ...]] = host_cpu_extensions


def parse_version(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), UNKNOWN_VERSION)


def parse_buildconf(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if line.strip().startswith("--"))


def _entries_after_legend(text: str) -> list[str]:
    lines = text.splitlines()
    ends = [index for index, line in enumerate(lines) if line.strip() == LEGEND_END]
    return lines[ends[0] + 1 :] if ends else []


def _listed_names(text: str) -> frozenset[str]:
    rows = (line.split() for line in _entries_after_legend(text))
    return frozenset(parts[1] for parts in rows if len(parts) > 1)


def parse_filters(text: str) -> frozenset[str]:
    return _listed_names(text)


def parse_encoders(text: str) -> frozenset[str]:
    return _listed_names(text)


def _require_binary(binary: Path) -> None:
    if not binary.is_file():
        raise FfmpegProbeError(f"ffmpeg not found at {binary}.")


def probe_with_hash(binary: Path, binary_sha256: str, tools: ProbeTools) -> FfmpegCapabilities:
    run = tools.run
    return FfmpegCapabilities(
        binary_sha256=binary_sha256,
        version=parse_version(run(binary, VERSION_ARGS)),
        configuration=parse_buildconf(run(binary, BUILDCONF_ARGS)),
        filters=parse_filters(run(binary, FILTERS_ARGS)),
        encoders=parse_encoders(run(binary, ENCODERS_ARGS)),
        cpu_extensions=tools.cpu_extensions(),
    )


def probe(binary: Path, tools: ProbeTools = ProbeTools()) -> FfmpegCapabilities:
    _require_binary(binary)
    return probe_with_hash(binary, tools.hash_file(binary), tools)


StatKey = tuple[str, int, int]


def _stat_key(binary: Path) -> StatKey:
    _require_binary(binary)
    stat = binary.stat()
    return str(binary.resolve()), stat.st_size, stat.st_mtime_ns


class FfmpegCapabilityCache:
    def __init__(self, tools: ProbeTools = ProbeTools()) -> None:
        self._tools = tools
        self._lock = threading.Lock()
        self._sha_by_stat: dict[StatKey, str] = {}
        self._by_sha: dict[str, FfmpegCapabilities] = {}

    def get(self, binary: Path) -> FfmpegCapabilities:
        with self._lock:
            sha = self._sha_for(binary)
            if sha not in self._by_sha:
                self._by_sha[sha] = probe_with_hash(binary, sha, self._tools)
            return self._by_sha[sha]

    def _sha_for(self, binary: Path) -> str:
        key = _stat_key(binary)
        if key not in self._sha_by_stat:
            self._sha_by_stat[key] = self._tools.hash_file(binary)
        return self._sha_by_stat[key]


_PROCESS_CACHE = FfmpegCapabilityCache()


def cached_capabilities(binary: Path) -> FfmpegCapabilities:
    return _PROCESS_CACHE.get(binary)


def _missing_filters(spec: FilterSpec, caps: FfmpegCapabilities) -> tuple[str, ...]:
    return tuple(name for name in spec.ffmpeg_filters if not caps.has_filter(name))


def _step_unavailable_filters(step: StepSpec, caps: FfmpegCapabilities) -> tuple[UnavailableFilter, ...]:
    pairs = ((spec, _missing_filters(spec, caps)) for spec in step.filters)
    return tuple(UnavailableFilter(step.id, spec.name, missing) for spec, missing in pairs if missing)


def _offered(chain: Sequence[StepSpec]) -> tuple[StepSpec, ...]:
    return tuple(step for step in chain if step.lanes and is_open(step, OPEN_GATES))


def unavailable_filters(
    caps: FfmpegCapabilities, chain: Sequence[StepSpec] = CCTV_CHAIN
) -> tuple[UnavailableFilter, ...]:
    return tuple(item for step in _offered(chain) for item in _step_unavailable_filters(step, caps))


def _step_is_unavailable(step: StepSpec, caps: FfmpegCapabilities) -> bool:
    return all(_missing_filters(spec, caps) for spec in step.filters)


def unavailable_steps(caps: FfmpegCapabilities, chain: Sequence[StepSpec] = CCTV_CHAIN) -> tuple[str, ...]:
    return tuple(step.id for step in _offered(chain) if _step_is_unavailable(step, caps))


def missing_encoders(caps: FfmpegCapabilities) -> tuple[str, ...]:
    return tuple(name for name in REQUIRED_ENCODERS if name not in caps.encoders)


def cctv_mode_available(caps: FfmpegCapabilities) -> bool:
    return not missing_encoders(caps)


def mode_unavailable_reason(caps: FfmpegCapabilities) -> str | None:
    missing = missing_encoders(caps)
    if not missing:
        return None
    return f"This ffmpeg build has no {' or '.join(missing)} encoder, which CCTV mode needs."
