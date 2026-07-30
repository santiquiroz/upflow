"""Spike de viabilidad: subtitulos por Whisper sobre ONNX Runtime.

Uso:
    .venv\\Scripts\\python scripts\\spike_whisper_onnx.py
    .venv\\Scripts\\python scripts\\spike_whisper_onnx.py --transcribe

La pregunta que responde NO es "existe la clase". Eso ya se sabe y no alcanza: en
este mismo repo optimum ejecuta `sana` aunque ORTSanaPipeline no exista, y al revés,
WhisperOnnxConfig existe pero TasksManager tiene el registro VACIO para todos los
model types en optimum 2.1.0 con optimum-onnx separado.

Las tres preguntas que si deciden:

  1. ¿Se puede CARGAR un repo Whisper ya exportado a ONNX, sin exportar nada?
     Si la respuesta es si, no hace falta el exportador y la capacidad entra por el
     mismo camino `ready_onnx` que ya usa el instalador.
  2. ¿Corre sobre DirectML, o solo CPU?
  3. ¿Transcribe de verdad? (--transcribe, genera un tono y mide que no explote)

Todo lo que descarga va a %TEMP% y se borra al terminar.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

# Repos Whisper YA exportados a ONNX. Se prueban en orden: el primero que cargue
# gana y define el camino.
CANDIDATE_REPOS = (
    "onnx-community/whisper-tiny.en",
    "onnx-community/whisper-tiny",
    "Xenova/whisper-tiny.en",
)

PROVIDERS_TO_TRY = ("DmlExecutionProvider", "CPUExecutionProvider")


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("optimum", "optimum-onnx", "onnxruntime-directml", "onnxruntime", "transformers"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "ausente"
    return out


def _available_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def _runtime_class_exists() -> bool:
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq  # noqa: F401
    except ImportError:
        return False
    return True


def _exporter_knows_whisper() -> dict[str, Any]:
    """Estado del registro del exportador.

    Se consulta un model type que SI deberia estar soportado (bert) como control:
    si ese tambien falla, el registro esta vacio y la respuesta sobre whisper no
    significa nada. Es la trampa en la que cae un chequeo ingenuo.
    """
    from optimum.exporters.tasks import TasksManager

    result: dict[str, Any] = {}
    for model_type in ("bert", "whisper"):
        try:
            tasks = TasksManager.get_supported_tasks_for_model_type(
                model_type, "onnx", library_name="transformers"
            )
            result[model_type] = sorted(tasks)
        except Exception as exc:  # noqa: BLE001 - el spike reporta, no falla
            result[model_type] = f"{type(exc).__name__}: {exc}"
    result["registro_vacio_para_todo"] = isinstance(result["bert"], str)
    return result


def _try_load(repo_id: str, provider: str, cache_dir: Path) -> dict[str, Any]:
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

    try:
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            repo_id,
            provider=provider,
            use_io_binding=False,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:  # noqa: BLE001 - el spike reporta, no falla
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"ok": True, "class": type(model).__name__, "model": model}


def _transcribe_smoke(model: Any, repo_id: str, cache_dir: Path) -> dict[str, Any]:
    """Transcribe un tono sintetico.

    No se busca que acierte palabras -- un tono no tiene ninguna -- sino que el
    grafo corra de punta a punta y devuelva texto sin explotar.
    """
    try:
        import numpy as np
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(repo_id, cache_dir=str(cache_dir))
        sample_rate = 16000
        seconds = 3
        t = np.linspace(0, seconds, sample_rate * seconds, endpoint=False)
        audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

        features = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        tokens = model.generate(input_features=features.input_features, max_new_tokens=16)
        text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"ok": True, "text": text.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Descarga un whisper-tiny ONNX y corre una transcripcion de humo.",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "versiones": _versions(),
        "providers_disponibles": _available_providers(),
        "clase_de_runtime_existe": _runtime_class_exists(),
        "estado_del_exportador": _exporter_knows_whisper(),
    }

    if args.transcribe:
        cache_dir = Path(tempfile.mkdtemp(prefix="upflow-whisper-spike-"))
        loads: dict[str, Any] = {}
        try:
            for repo_id in CANDIDATE_REPOS:
                for provider in PROVIDERS_TO_TRY:
                    if provider not in report["providers_disponibles"]:
                        loads[f"{repo_id} @ {provider}"] = {"ok": False, "error": "provider ausente"}
                        continue
                    outcome = _try_load(repo_id, provider, cache_dir)
                    model = outcome.pop("model", None)
                    loads[f"{repo_id} @ {provider}"] = outcome
                    if outcome["ok"] and model is not None:
                        loads[f"{repo_id} @ {provider}"]["transcripcion"] = _transcribe_smoke(
                            model, repo_id, cache_dir
                        )
                        del model
                if any(entry.get("ok") for entry in loads.values()):
                    break
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
        report["cargas"] = loads

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
