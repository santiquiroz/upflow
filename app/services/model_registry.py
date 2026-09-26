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
from app.services.classic_upscalers import catalog_entries as classic_catalog_entries

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "registry.json"

# El proposito de Spandrel ("SR", "Restoration", ...) de los reescaladores que trae
# la app. Los modelos convertidos guardan el que declare su arquitectura.
PURPOSE_SUPER_RESOLUTION = "SR"
RGB_CHANNELS = 3

CAPABILITY_FIELDS = (
    "purpose",
    "channels_in",
    "channels_out",
    "size_minimum",
    "size_multiple",
    "size_square",
    "tiling",
    "fp16_file",
)

CLASSIC_MODEL_IDS = frozenset(entry["key"] for entry in classic_catalog_entries())

# Ids que nadie puede sobrescribir ni borrar. Incluye los clasicos: no tienen archivos
# que borrar, asi que quitarlos solo los volveria a resembrar como zombies.
BUILTIN_MODEL_IDS = frozenset(option["key"] for option in MODEL_CATALOG) | CLASSIC_MODEL_IDS


class ModelKind(str, Enum):
    builtin_ncnn = "builtin-ncnn"
    onnx = "onnx"
    diffusion_onnx = "diffusion-onnx"
    # Reconocimiento de voz: el par encoder/decoder de un seq2seq exportado.
    # Se persiste, asi que una version anterior no va a saber leerlo -- por eso
    # _parse_entries SALTA la entrada en vez de invalidar el registro entero.
    asr_onnx = "asr-onnx"
    # Reescalado sin IA (swscale). No hay pesos ni archivo: nunca aparece en el
    # registro, solo como resolucion de un job. Existe como kind para que el ruteo de
    # dispositivo no lo confunda con builtin-ncnn, que EXIGE una GPU Vulkan.
    classic = "classic"


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
    # Solo instalaciones desde un checkpoint suelto: el archivo exacto del repo
    # de origen. El merge de inpainting lo necesita para volver a bajar LOS
    # MISMOS pesos; source solo guarda el repo.
    checkpoint_path: str | None = None
    status: ModelStatus = ModelStatus.installed
    error: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    # Si inventa textura. True por defecto: de un modelo sin declarar no se sabe.
    generative: bool = True
    # Capacidades que declara Spandrel al convertir. None = no se sabe: un .onnx
    # publicado o una entrada de una version anterior no las trae.
    purpose: str | None = None
    channels_in: int | None = None
    channels_out: int | None = None
    size_minimum: int | None = None
    size_multiple: int | None = None
    size_square: bool | None = None
    tiling: str | None = None
    # Relativo a models_path, como file_path. Solo si la variante fp16 existe.
    fp16_file: str | None = None


def _entry_to_json_dict(entry: ModelEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "name": entry.name,
        "kind": entry.kind.value,
        "source": entry.source,
        "scale": entry.scale,
        "arch": entry.arch,
        "file_path": entry.file_path,
        "checkpoint_path": entry.checkpoint_path,
        "size_bytes": entry.size_bytes,
        "status": entry.status.value,
        "error": entry.error,
        "created_at": entry.created_at.isoformat(),
        "generative": entry.generative,
        **{name: getattr(entry, name) for name in CAPABILITY_FIELDS},
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
        # .get: las entradas escritas antes de v0.57.0 no traen el campo.
        checkpoint_path=data.get("checkpoint_path"),
        status=ModelStatus(data["status"]),
        error=data.get("error"),
        created_at=datetime.fromisoformat(data["created_at"]),
        # .get: las entradas escritas antes del campo se tratan como generativas.
        generative=bool(data.get("generative", True)),
        # .get: las entradas escritas antes de MNT-03 no las traen y quedan "sin saber".
        **{name: data.get(name) for name in CAPABILITY_FIELDS},
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
        generative=option["generative"],
        purpose=PURPOSE_SUPER_RESOLUTION,
        channels_in=RGB_CHANNELS,
        channels_out=RGB_CHANNELS,
    )


def _refreshed_builtin(stored: ModelEntry, option: ModelOption) -> ModelEntry:
    return replace(_builtin_entry_from_catalog(option), created_at=stored.created_at)


def _lacks_purpose(entry: ModelEntry | None) -> bool:
    return entry is not None and entry.purpose is None


def _classic_entries() -> list[ModelEntry]:
    """Los upscalers clasicos, SIEMPRE en memoria y NUNCA en registry.json.

    Se derivan del catalogo en cada llamada en vez de persistirse, y la razon es de
    compatibilidad hacia ATRAS, no de estilo: el `_load` de las versiones ya publicadas
    (v0.16.2 incluida) es todo-o-nada, asi que al toparse con un kind desconocido como
    "classic" no saltea la entrada -- respalda el registry.json ENTERO y arranca vacio.
    Alguien que instale esta version y despues vuelva a la anterior perderia todos sus
    modelos instalados. El fix que saltea entradas ilegibles existe pero no esta en
    ninguna tag publicada, asi que no puede protegernos.

    Como no tienen archivos ni estado, no hay nada que persistir: derivarlos es
    equivalente y no toca el disco.
    """
    return [
        ModelEntry(
            id=option["key"],
            name=option["label"],
            kind=ModelKind.classic,
            source="builtin",
            size_bytes=0,
            scale=None,
            status=ModelStatus.installed,
            generative=option["generative"],
            purpose=PURPOSE_SUPER_RESOLUTION,
            channels_in=RGB_CHANNELS,
            channels_out=RGB_CHANNELS,
        )
        for option in classic_catalog_entries()
    ]




def is_generative(entry: ModelEntry | None) -> bool:
    return True if entry is None else entry.generative


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
    if entry.kind in (ModelKind.builtin_ncnn, ModelKind.classic):
        raise ValueError(f"Cannot register builtin entry: {entry.id!r} ({entry.kind.value})")


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
            return [*self._entries.values(), *_classic_entries()]

    def get(self, model_id: str) -> ModelEntry | None:
        with self._lock:
            entry = self._entries.get(model_id)
        if entry is not None:
            return entry
        return next((e for e in _classic_entries() if e.id == model_id), None)

    def register(self, entry: ModelEntry) -> ModelEntry:
        _reject_builtin_conflicts(entry)
        stored = replace(entry)
        with self._lock:
            self._entries[stored.id] = stored
            self._persist()
        return stored

    def remove(self, model_id: str) -> None:
        if model_id in CLASSIC_MODEL_IDS:
            # No viven en _entries (ver _classic_entries), asi que sin este chequeo el
            # error diria "id desconocido" y sonaria a un bug en vez de a una negativa.
            raise ValueError(f"Cannot remove builtin model: {model_id!r}")
        with self._lock:
            entry = self._require_entry(model_id)
            if entry.kind in (ModelKind.builtin_ncnn, ModelKind.classic):
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
            if not isinstance(raw, list):
                raise TypeError(f"expected a list of entries, got {type(raw).__name__}")
        except (json.JSONDecodeError, TypeError) as exc:
            # El ARCHIVO no se puede leer: ahi si hay que respaldarlo y resembrar.
            self._backup_corrupt_registry(exc)
            return {}
        return self._parse_entries(raw)

    def _parse_entries(self, raw: list[Any]) -> dict[str, ModelEntry]:
        """Una entrada ilegible se SALTA; no invalida el archivo entero.

        Importa por las versiones: ModelKind se persiste, asi que una version nueva
        que agrega un kind deja entradas que la anterior no sabe leer. Tratar eso
        como archivo corrupto renombraba el registro completo y el usuario perdia
        TODOS sus modelos instalados por una sola entrada que no entiende.
        """
        entries: dict[str, ModelEntry] = {}
        last_error: Exception | None = None
        for item in raw:
            try:
                entry = _entry_from_json_dict(item)
            except (KeyError, ValueError, TypeError) as exc:
                last_error = exc
                logger.warning(
                    "Skipping unreadable model registry entry %r: %s",
                    item.get("id") if isinstance(item, dict) else item,
                    exc,
                )
                continue
            entries[entry.id] = entry

        # Si NINGUNA entrada se pudo leer, el archivo esta roto de verdad: se
        # respalda para poder mirarlo despues. Con al menos una entrada buena, en
        # cambio, saltar las malas conserva lo que sirve.
        if raw and not entries and last_error is not None:
            self._backup_corrupt_registry(last_error)
        return entries

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
            seeded = [*self._missing_builtins(), *self._builtins_without_purpose()]
            if not seeded:
                return
            for entry in seeded:
                self._entries[entry.id] = entry
            self._persist()

    def _missing_builtins(self) -> list[ModelEntry]:
        return [
            _builtin_entry_from_catalog(option)
            for option in MODEL_CATALOG
            if option["key"] not in self._entries
        ]

    def _builtins_without_purpose(self) -> list[ModelEntry]:
        # Los sembro una version anterior a MNT-03: el catalogo sabe su proposito.
        return [
            _refreshed_builtin(self._entries[option["key"]], option)
            for option in MODEL_CATALOG
            if _lacks_purpose(self._entries.get(option["key"]))
        ]

    def _persist(self) -> None:
        # Callers must hold self._lock.
        payload = [_entry_to_json_dict(entry) for entry in self._entries.values()]
        _write_json_atomically(self._registry_path, payload)
