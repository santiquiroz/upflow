"""Traducción local con los modelos OPUS-MT sobre ONNX.

Verificado el 2026-08-05 (`scripts/spike_translation.py`): 93% de parecido en la
ida y vuelta inglés → español → inglés, a 0,3 s por frase.

Un modelo POR PAR de idiomas, que es como los publica OPUS-MT. Traducir de
cualquier idioma a cualquier otro necesitaría un modelo multilingüe, y los
buenos (NLLB) son CC-BY-NC, o sea no comerciales.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Los subtitulos se traducen SEGMENTO POR SEGMENTO y no como un texto corrido:
# los tiempos pertenecen a cada segmento, y traducir todo junto los perderia.
MAX_TOKENS = 512


class TranslationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LanguagePair:
    source: str
    target: str

    @property
    def directory(self) -> str:
        return f"opus-mt-{self.source}-{self.target}"


def parse_pair(source: str, target: str) -> LanguagePair:
    source, target = source.strip().lower(), target.strip().lower()
    if not source or not target:
        raise TranslationUnavailable("Hacen falta el idioma de origen y el de destino.")
    if source == target:
        raise TranslationUnavailable("El idioma de origen y el de destino son el mismo.")
    return LanguagePair(source=source, target=target)


class TranslationEngine:
    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root
        self._cache: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.Lock()

    def available_pairs(self) -> list[LanguagePair]:
        if not self.models_root.is_dir():
            return []
        pares = []
        for carpeta in sorted(self.models_root.glob("opus-mt-*-*")):
            partes = carpeta.name.split("-")
            if len(partes) == 4 and (carpeta / "onnx").is_dir():
                pares.append(LanguagePair(source=partes[2], target=partes[3]))
        return pares

    def available(self, pair: LanguagePair) -> bool:
        return (self.models_root / pair.directory / "onnx").is_dir()

    def translate(self, texts: list[str], pair: LanguagePair) -> list[str]:
        """Traduce en lote, conservando el orden y los vacios.

        Los vacios se conservan en su posicion porque el llamador empareja el
        resultado con los segmentos por indice: descartarlos correria todo.
        """
        tokenizer, model = self._load(pair)
        salida: list[str] = []
        for texto in texts:
            if not texto.strip():
                salida.append(texto)
                continue
            entrada = tokenizer(texto, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
            generado = model.generate(**entrada, max_new_tokens=MAX_TOKENS)
            salida.append(tokenizer.batch_decode(generado, skip_special_tokens=True)[0].strip())
        return salida

    def _load(self, pair: LanguagePair) -> tuple[Any, Any]:
        with self._lock:
            cached = self._cache.get(pair.directory)
            if cached is not None:
                return cached

            directorio = self.models_root / pair.directory
            if not directorio.is_dir():
                raise TranslationUnavailable(
                    f"No hay modelo para traducir de {pair.source} a {pair.target}. "
                    f"Instalalo con scripts/download-translation.ps1 -Pair {pair.source}-{pair.target}"
                )
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            from transformers import AutoTokenizer

            # `use_merged=False` por la misma razon que en Whisper: el decoder
            # merged ya devolvio texto fluido y equivocado en este repo.
            par = (
                AutoTokenizer.from_pretrained(str(directorio)),
                ORTModelForSeq2SeqLM.from_pretrained(str(directorio), use_merged=False),
            )
            self._cache[pair.directory] = par
            return par
