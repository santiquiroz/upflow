"""Prompts que guarda el usuario.

La distincion que define el diseño: un prompt guardado es DATO DEL USUARIO y no
copia de la app. Se guarda tal cual lo escribio y no se traduce nunca — es el
texto que va al modelo, y normalizarlo cambiaria lo que genera. Los presets de
fabrica (`prompt_presets.py`) si son copia y viajan como claves de traduccion.

La app tiene usuarios, asi que cada uno ve los suyos: lo que alguien guarda es
de la persona, no de la maquina.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import Settings
from app.services.json_store import backup_corrupt_file, write_json_atomically
from app.models import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SavedPrompt:
    id: str
    owner_id: str
    name: str
    prompt: str
    negative_prompt: str = ""
    mode: str = "text-to-image"
    created_at: datetime = field(default_factory=utc_now)


class SavedPromptStore:
    def __init__(self, settings: Settings) -> None:
        self.path = settings.auth_path / "saved_prompts.json"
        # Un lock por instancia alcanza: el archivo se reescribe entero en cada
        # cambio, y dos requests concurrentes del mismo proceso son el caso real.
        self._lock = threading.Lock()
        self._prompts: dict[str, SavedPrompt] = self._load()

    def list_for(self, owner_id: str) -> list[SavedPrompt]:
        # El ultimo guardado primero: es el que se acaba de usar. Se ordena por
        # ORDEN DE INSERCION y no por `created_at`, porque dos guardados en el
        # mismo instante empatan en el reloj y el orden queda al azar. El dict
        # conserva el orden de insercion, y `_load` lo reconstruye desde el
        # archivo, que se escribe en ese mismo orden.
        mine = [p for p in self._prompts.values() if p.owner_id == owner_id]
        mine.reverse()
        return mine

    def save(
        self,
        *,
        owner_id: str,
        name: str,
        prompt: str,
        negative_prompt: str = "",
        mode: str = "text-to-image",
    ) -> SavedPrompt:
        if not name.strip():
            raise ValueError("Un prompt guardado necesita un nombre para poder encontrarlo.")
        if not prompt.strip():
            raise ValueError("No hay nada que guardar: el prompt esta vacio.")

        saved = SavedPrompt(
            id=uuid4().hex,
            owner_id=owner_id,
            name=name.strip(),
            prompt=prompt.strip(),
            negative_prompt=negative_prompt.strip(),
            mode=mode,
        )
        with self._lock:
            self._prompts[saved.id] = saved
            self._persist()
        return saved

    def delete(self, *, owner_id: str, prompt_id: str) -> bool:
        with self._lock:
            existing = self._prompts.get(prompt_id)
            # Comprobar el dueño ACA y no en la ruta: si la comprobacion vive
            # solo en la capa web, cualquier otro llamador la saltea.
            if existing is None or existing.owner_id != owner_id:
                return False
            del self._prompts[prompt_id]
            self._persist()
            return True

    def _persist(self) -> None:
        payload = []
        for saved in self._prompts.values():
            item: dict[str, Any] = asdict(saved)
            item["created_at"] = saved.created_at.isoformat()
            payload.append(item)
        write_json_atomically(self.path, payload)

    def _load(self) -> dict[str, SavedPrompt]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return {
                item["id"]: SavedPrompt(
                    id=item["id"],
                    owner_id=item["owner_id"],
                    name=item["name"],
                    prompt=item["prompt"],
                    negative_prompt=item.get("negative_prompt", ""),
                    mode=item.get("mode", "text-to-image"),
                    created_at=datetime.fromisoformat(item["created_at"]),
                )
                for item in raw
            }
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            # Perder los prompts guardados es malo; no arrancar es peor. El
            # archivo roto queda respaldado para poder mirarlo.
            backup_corrupt_file(self.path, exc, logger)
            return {}
