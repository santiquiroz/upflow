from __future__ import annotations

from pathlib import Path

from app.config import MODEL_CATALOG
from app.services.cctv_ai_models import (
    GENERATIVE_LABEL,
    NON_GENERATIVE_LABEL,
    catalog_generative,
    export_on_disk,
    generative_label,
    runs_in_stream,
    stream_upscale_models,
)

CATALOG = [
    {"key": "realesr-animevideov3", "label": "AnimeVideo v3", "scales": [2, 3, 4], "generative": True},
    {"key": "realesrgan-x4plus", "label": "x4 Plus", "scales": [2, 3, 4], "generative": True},
    {"key": "classic-lanczos", "label": "Lanczos", "scales": [2, 3, 4], "generative": False},
    {"key": "tidy-denoiser", "label": "Tidy", "scales": [2], "generative": False},
]


def installed_only(*names: str):
    return lambda filename: filename in names


def test_the_generative_label_names_invented_texture() -> None:
    assert generative_label(True) == GENERATIVE_LABEL == "Generative (invents texture)"
    assert generative_label(False) == NON_GENERATIVE_LABEL == "Non-generative"


def test_only_builtin_onnx_exports_for_that_scale_run_in_the_stream() -> None:
    assert runs_in_stream("realesr-animevideov3", 3)
    assert runs_in_stream("realesrgan-x4plus", 2)
    assert not runs_in_stream("realesr-animevideov3-x2", 4)
    assert not runs_in_stream("classic-lanczos", 2)
    assert not runs_in_stream("someone/hf-upscaler", 2)
    assert not runs_in_stream("realesrgan-x4plus", 1)


def test_the_stream_upscaler_list_keeps_installed_scales_and_the_generative_label() -> None:
    installed = installed_only("realesr-animevideov3-x2-uint8.onnx", "realesrgan-x4plus-uint8.onnx")

    models = [model.to_json() for model in stream_upscale_models(CATALOG, installed)]

    assert models == [
        {
            "id": "realesr-animevideov3",
            "label": "AnimeVideo v3",
            "scales": [2],
            "generative": True,
            "generativeLabel": GENERATIVE_LABEL,
        },
        {
            "id": "realesrgan-x4plus",
            "label": "x4 Plus",
            "scales": [4],
            "generative": True,
            "generativeLabel": GENERATIVE_LABEL,
        },
    ]


def test_every_builtin_in_the_real_catalog_is_declared_generative() -> None:
    assert all(catalog_generative(MODEL_CATALOG, option["key"]) for option in MODEL_CATALOG)


def test_an_undeclared_model_counts_as_generative() -> None:
    assert catalog_generative(CATALOG, "unknown-model") is True
    assert catalog_generative(CATALOG, "tidy-denoiser") is False


def test_an_export_is_on_disk_only_as_a_file(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "folder.onnx").mkdir()

    assert export_on_disk(tmp_path, "model.onnx")
    assert not export_on_disk(tmp_path, "folder.onnx")
    assert not export_on_disk(tmp_path, "missing.onnx")
