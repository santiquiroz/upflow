from __future__ import annotations

import json
import logging
import tempfile
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from app.config import MODEL_CATALOG, ModelOption, Settings
from app.models import utc_now

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "registry.json"

BUILTIN_MODEL_IDS = frozenset(option["key"] for option in MODEL_CATALOG)


class ModelKind(str, Enum):
    builtin_ncnn = "builtin-ncnn"
    onnx = "onnx"
    diffusion_onnx = "diffusion-onnx"


class ModelStatus(str, Enum):
    installed = "installed"
    converting = "converting"
    error = "error"


@dataclass(slots=True, kw_only=True)
class ModelEntry:
    id: str
    name: str
    kind: ModelKind
    source: str
    size_bytes: int
    scale: int | None = None
    arch: str | None = None
    file_path: str | None = None
    status: ModelStatus = ModelStatus.installed
    error: str | None = None
    created_at: datetime = field(default_factory=utc_now)


def _entry_to_json_dict(entry: ModelEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "name": entry.name,
        "kind": entry.kind.value,
        "source": entry.source,
        "scale": entry.scale,
        "arch": entry.arch,
        "file_path": entry.file_path,
        "size_bytes": entry.size_bytes,
        "status": entry.status.value,
        "error": entry.error,
        "created_at": entry.created_at.isoformat(),
    }


def _entry_from_json_dict(data: dict[str, Any]) -> ModelEntry:
    return ModelEntry(
        id=data["id"],
        name=data["name"],
        kind=ModelKind(data["kind"]),
        source=data["source"],
        size_bytes=data["size_bytes"],
        scale=data.get("scale"),
        arch=data.get("arch"),
        file_path=data.get("file_path"),
        status=ModelStatus(data["status"]),
        error=data.get("error"),
        created_at=datetime.fromisoformat(data["created_at"]),
    )


def _single_scale(option: ModelOption) -> int | None:
    scales = option["scales"]
    return scales[0] if len(scales) == 1 else None


def _builtin_entry_from_catalog(option: ModelOption) -> ModelEntry:
    return ModelEntry(
        id=option["key"],
        name=option["label"],
        kind=ModelKind.builtin_ncnn,
        source="builtin",
        size_bytes=0,
        scale=_single_scale(option),
        status=ModelStatus.installed,
    )


def _write_json_atomically(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp in the same directory (not the OS temp dir) so Path.replace is
    # an atomic rename on the same filesystem, never a cross-device copy.
    descriptor, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with open(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _reject_builtin_conflicts(entry: ModelEntry) -> None:
    if entry.id in BUILTIN_MODEL_IDS:
        raise ValueError(f"Cannot overwrite builtin model: {entry.id!r}")
    # Only the internal seed creates builtin entries; a caller-registered
    # builtin-ncnn entry would be an unremovable zombie.
    if entry.kind == ModelKind.builtin_ncnn:
        raise ValueError(f"Cannot register builtin-ncnn entry: {entry.id!r}")


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._registry_path = settings.models_path / REGISTRY_FILENAME
        # Guards the in-memory dict and the persist step: Task 5+ calls the
        # registry from asyncio.to_thread workers, and two unlocked os.replace
        # calls racing on the same target raise PermissionError on Windows.
        self._lock = threading.Lock()
        self._entries: dict[str, ModelEntry] = self._load()
        self._seed_builtins()

    def list(self) -> list[ModelEntry]:
        with self._lock:
            return list(self._entries.values())

    def get(self, model_id: str) -> ModelEntry | None:
        with self._lock:
            return self._entries.get(model_id)

    def register(self, entry: ModelEntry) -> ModelEntry:
        _reject_builtin_conflicts(entry)
        stored = replace(entry)
        with self._lock:
            self._entries[stored.id] = stored
            self._persist()
        return stored

    def remove(self, model_id: str) -> None:
        with self._lock:
            entry = self._require_entry(model_id)
            if entry.kind == ModelKind.builtin_ncnn:
                raise ValueError(f"Cannot remove builtin model: {model_id!r}")
            del self._entries[model_id]
            self._persist()

    def _require_entry(self, model_id: str) -> ModelEntry:
        entry = self._entries.get(model_id)
        if entry is None:
            raise ValueError(f"Unknown model id: {model_id!r}")
        return entry

    def _load(self) -> dict[str, ModelEntry]:
        if not self._registry_path.exists():
            return {}
        try:
            raw = json.loads(self._registry_path.read_text(encoding="utf-8"))
            return {item["id"]: _entry_from_json_dict(item) for item in raw}
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            self._backup_corrupt_registry(exc)
            return {}

    def _backup_corrupt_registry(self, exc: Exception) -> None:
        timestamp = utc_now().strftime("%Y%m%dT%H%M%S%f")
        backup_path = self._registry_path.with_name(
            f"{self._registry_path.name}.corrupt-{timestamp}"
        )
        self._registry_path.replace(backup_path)
        logger.warning(
            "Corrupt model registry at %s (%s); backed up to %s, reseeding builtins",
            self._registry_path,
            exc,
            backup_path,
        )

    def _seed_builtins(self) -> None:
        with self._lock:
            missing = [
                _builtin_entry_from_catalog(option)
                for option in MODEL_CATALOG
                if option["key"] not in self._entries
            ]
            if not missing:
                return
            for entry in missing:
                self._entries[entry.id] = entry
            self._persist()

    def _persist(self) -> None:
        # Callers must hold self._lock.
        payload = [_entry_to_json_dict(entry) for entry in self._entries.values()]
        _write_json_atomically(self._registry_path, payload)
