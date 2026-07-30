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
  2. ¿Corre sobre DirectML, o solo CPU? Y con QUE variante del decoder: con el
     merged, que es el default, DirectML devuelve basura sin fallar.
  3. ¿Transcribe VOZ REAL bien? (--real-speech, con una muestra de transcripcion
     conocida) -- es lo unico que dice algo sobre calidad. El modo --transcribe usa
     un tono sin habla y solo mide que el grafo corra.

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

# Muestra de voz REAL, en un repo con los audios sueltos (sin necesidad de la
# libreria `datasets`). El texto esperado sale de la frase famosa que contiene.
SPEECH_REPO = "Narsil/asr_dummy"
SPEECH_FILE = "i-know-kung-fu.mp3"
SPEECH_EXPECTED_WORDS = ("know", "kung", "fu")


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


def _try_load(
    repo_id: str, provider: str, cache_dir: Path, use_merged: bool = True
) -> dict[str, Any]:
    """Carga el modelo.

    use_merged=False NO es cosmetico: con el decoder merged (que es el DEFAULT),
    DirectML carga, corre sin error y devuelve BASURA. Medido 2026-07-29. Ver el
    doc de hallazgos.
    """
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

    kwargs: dict[str, Any] = {} if use_merged else {"use_merged": False}
    try:
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            repo_id,
            provider=provider,
            use_io_binding=False,
            cache_dir=str(cache_dir),
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - el spike reporta, no falla
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"ok": True, "class": type(model).__name__, "model": model}


def _load_real_speech(cache_dir: Path) -> Any:
    """Baja una muestra de voz real y la deja a 16 kHz mono, que es lo que Whisper espera."""
    import numpy as np
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=SPEECH_REPO,
        filename=SPEECH_FILE,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != 16000:
        # Remuestreo lineal: alcanza para un spike y evita traer librosa.
        target_length = int(len(mono) * 16000 / sample_rate)
        mono = np.interp(
            np.linspace(0, len(mono), target_length, endpoint=False),
            np.arange(len(mono)),
            mono,
        ).astype("float32")
    return mono


def _transcribe_real_speech(model: Any, repo_id: str, cache_dir: Path) -> dict[str, Any]:
    """Transcribe voz REAL y verifica que aparezcan las palabras esperadas.

    Esto SI dice algo sobre calidad, a diferencia del smoke con un tono: si el
    texto no contiene las palabras que la muestra dice, el camino no sirve por mas
    que el grafo corra.
    """
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(repo_id, cache_dir=str(cache_dir))
        audio = _load_real_speech(cache_dir)
        features = processor(audio, sampling_rate=16000, return_tensors="pt")
        tokens = model.generate(input_features=features.input_features, max_new_tokens=64)
        text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    lowered = text.lower()
    hits = [word for word in SPEECH_EXPECTED_WORDS if word in lowered]
    return {
        "ok": True,
        "text": text,
        "palabras_esperadas": list(SPEECH_EXPECTED_WORDS),
        "encontradas": hits,
        "calidad_aceptable": len(hits) == len(SPEECH_EXPECTED_WORDS),
    }


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
        help="Descarga un whisper-tiny ONNX y corre una transcripcion de humo (tono, sin habla).",
    )
    parser.add_argument(
        "--real-speech",
        action="store_true",
        help="Transcribe VOZ REAL. Es lo unico que dice algo sobre calidad.",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "versiones": _versions(),
        "providers_disponibles": _available_providers(),
        "clase_de_runtime_existe": _runtime_class_exists(),
        "estado_del_exportador": _exporter_knows_whisper(),
    }

    if args.transcribe or args.real_speech:
        cache_dir = Path(tempfile.mkdtemp(prefix="upflow-whisper-spike-"))
        loads: dict[str, Any] = {}
        try:
            for repo_id in CANDIDATE_REPOS:
                for provider in PROVIDERS_TO_TRY:
                    if provider not in report["providers_disponibles"]:
                        loads[f"{repo_id} @ {provider}"] = {"ok": False, "error": "provider ausente"}
                        continue
                    # Las DOS variantes del decoder: el merged es el default y en
                    # DirectML devuelve basura sin fallar.
                    for use_merged in (True, False):
                        label = f"{repo_id} @ {provider} merged={use_merged}"
                        outcome = _try_load(repo_id, provider, cache_dir, use_merged)
                        model = outcome.pop("model", None)
                        loads[label] = outcome
                        if outcome["ok"] and model is not None:
                            if args.transcribe:
                                loads[label]["tono_sin_habla"] = _transcribe_smoke(
                                    model, repo_id, cache_dir
                                )
                            if args.real_speech:
                                loads[label]["voz_real"] = _transcribe_real_speech(
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
