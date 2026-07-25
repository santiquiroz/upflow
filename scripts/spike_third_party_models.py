"""Spike reproducible para modelos de difusión de terceros.

Uso:
    .venv\Scripts\python scripts\spike_third_party_models.py
    .venv\Scripts\python scripts\spike_third_party_models.py --smoke-export

El modo smoke descarga y exporta un pipeline tiny exclusivamente bajo %TEMP%
y elimina tanto el snapshot como el resultado ONNX al terminar.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch


HF_BASE_URL = "https://huggingface.co"
TINY_MODEL_ID = "hf-internal-testing/tiny-stable-diffusion-torch"

REPOSITORIES = (
    {
        "variant": "SDXL",
        "repo_id": "amd/stable-diffusion-xl-1.0_io16_amdgpu",
        "evidence": "primary amd/",
    },
    {
        "variant": "SDXL Turbo",
        "repo_id": "amd/sdxl-turbo_amdgpu",
        "evidence": "primary amd/",
    },
    {
        "variant": "SDXL Turbo",
        "repo_id": "amd/stable-diffusion-sdxl-turbo-amdnpu-onnx",
        "evidence": "primary amd/",
    },
    {
        "variant": "SDXL Turbo",
        "repo_id": "stabilityai/sdxl-turbo",
        "evidence": "secondary public equivalent",
    },
    {
        "variant": "SD3.5",
        "repo_id": "amd/stable-diffusion-3.5-medium_amdgpu",
        "evidence": "primary amd/",
    },
    {
        "variant": "SD3.5",
        "repo_id": "stabilityai/stable-diffusion-3.5-medium",
        "evidence": "secondary official gated",
    },
    {
        "variant": "SD3.5",
        "repo_id": "adamo1139/stable-diffusion-3.5-medium-ungated",
        "evidence": "secondary public equivalent",
    },
)


def _request(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": "upflow-third-party-model-spike/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _request_json(url: str) -> tuple[int, Any | None]:
    status, payload = _request(url)
    if status != 200:
        return status, None
    return status, json.loads(payload)


def _declared_components(model_index: dict[str, Any] | None) -> list[str]:
    if model_index is None:
        return []
    return sorted(
        key
        for key, value in model_index.items()
        if not key.startswith("_") and isinstance(value, list) and len(value) == 2
    )


def _storage_kind(files: list[str]) -> list[str]:
    kinds: set[str] = set()
    for filename in files:
        lower = filename.lower()
        if lower.endswith(".onnx") or ".onnx." in lower or lower.endswith("onnx_data"):
            kinds.add("ONNX")
        if lower.endswith((".safetensors", ".bin", ".pt", ".pth")):
            kinds.add("PyTorch weights")
        if lower.endswith((".mxr", ".ctrlpkt", ".fconst", ".state", ".super")):
            kinds.add("hardware-specific compiled artifacts")
    return sorted(kinds) or ["metadata/config only"]


def inspect_repository(spec: dict[str, str]) -> dict[str, Any]:
    repo_id = spec["repo_id"]
    api_status, metadata = _request_json(f"{HF_BASE_URL}/api/models/{repo_id}?blobs=true")
    index_status, model_index = _request_json(f"{HF_BASE_URL}/{repo_id}/resolve/main/model_index.json")

    siblings = []
    if isinstance(metadata, dict):
        siblings = [item["rfilename"] for item in metadata.get("siblings", [])]

    components = _declared_components(model_index)
    component_storage = {
        component: _storage_kind(
            [filename for filename in siblings if filename.startswith(f"{component}/")]
        )
        for component in components
    }

    return {
        **spec,
        "api_status": api_status,
        "gated": metadata.get("gated") if isinstance(metadata, dict) else None,
        "model_index_status": index_status,
        "_class_name": model_index.get("_class_name") if isinstance(model_index, dict) else None,
        "declared_components": components,
        "component_storage": component_storage,
        "repository_storage": _storage_kind(siblings),
        "has_model_index": "model_index.json" in siblings,
    }


def _per_component_export_signatures() -> dict[str, str]:
    import optimum.exporters.onnx.convert as convert_module

    return {
        name: str(inspect.signature(getattr(convert_module, name)))
        for name in ("export_models", "export", "export_pytorch")
        if hasattr(convert_module, name)
    }


def inspect_optimum() -> dict[str, Any]:
    import onnxruntime as ort
    import optimum.onnxruntime as ort_module
    from optimum.exporters.onnx import main_export
    from optimum.utils.import_utils import is_onnxruntime_available

    diffusion_classes = [
        name for name in dir(ort_module) if "Diffusion" in name or "StableDiffusion" in name
    ]
    signature = inspect.signature(main_export)
    return {
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "optimum",
                "optimum-onnx",
                "onnxruntime-directml",
                "diffusers",
                "transformers",
            )
        },
        "onnxruntime_module": {
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
        },
        "optimum_detects_onnxruntime": is_onnxruntime_available(),
        "diffusion_classes_from_dir": diffusion_classes,
        "main_export_signature": str(signature),
        "main_export_has_callback": "callback" in signature.parameters,
        "per_component_export_signatures": _per_component_export_signatures(),
    }


def smoke_export(model_id: str) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from optimum.exporters.onnx import main_export
    from optimum.utils import logging as optimum_logging
    from optimum.utils.import_utils import is_onnxruntime_available

    temp_root = Path(tempfile.mkdtemp(prefix="upflow-third-party-spike-"))
    source_dir = temp_root / "source"
    cache_dir = temp_root / "cache"
    output_dir = temp_root / "onnx"
    result: dict[str, Any] | None = None

    try:
        optimum_logging.set_verbosity_info()
        snapshot_download(
            repo_id=model_id,
            local_dir=source_dir,
            cache_dir=cache_dir,
        )
        # optimum==2.1.0 no reconoce la distribución onnxruntime-directml aunque
        # el módulo onnxruntime sí importa. El guard solo protege un import que
        # ya verificamos; este patch permite probar main_export sin instalar el
        # paquete onnxruntime vanilla, que pisaría el backend DirectML.
        detector_workaround = not is_onnxruntime_available()
        with patch(
            "optimum.exporters.onnx.base.is_onnxruntime_available",
            return_value=True,
        ):
            main_export(
                str(source_dir),
                output_dir,
                task="text-to-image",
                device="cpu",
                cache_dir=str(cache_dir),
            )

        files = sorted(
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        onnx_files = [filename for filename in files if filename.endswith(".onnx")]
        if "model_index.json" not in files:
            raise RuntimeError("El export no produjo model_index.json")
        if not onnx_files:
            raise RuntimeError("El export no produjo archivos .onnx")

        exported_model_index = json.loads((output_dir / "model_index.json").read_text(encoding="utf-8"))
        result = {
            "model_id": model_id,
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "invocation": (
                "with patch('optimum.exporters.onnx.base.is_onnxruntime_available', "
                "return_value=True): main_export(str(source_dir), output_dir, "
                "task='text-to-image', device='cpu', cache_dir=str(cache_dir))"
            ),
            "directml_distribution_detector_workaround": detector_workaround,
            "exported_class_name": exported_model_index.get("_class_name"),
            "exported_components": _declared_components(exported_model_index),
            "files": files,
            "onnx_files": onnx_files,
        }
        print("SMOKE_EXPORT_RESULT")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        shutil.rmtree(temp_root)
        print(f"SMOKE_TEMP_REMOVED={not temp_root.exists()} path={temp_root}")

    if result is None:
        raise RuntimeError("El smoke export no produjo resultado")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-export",
        action="store_true",
        help="Descarga y exporta el pipeline tiny dentro de %%TEMP%%.",
    )
    parser.add_argument("--tiny-model", default=TINY_MODEL_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("HF_REPOSITORY_EVIDENCE")
    for spec in REPOSITORIES:
        print(json.dumps(inspect_repository(spec), indent=2, ensure_ascii=False))

    print("OPTIMUM_RUNTIME_EVIDENCE")
    print(json.dumps(inspect_optimum(), indent=2, ensure_ascii=False))

    if args.smoke_export:
        smoke_export(args.tiny_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
