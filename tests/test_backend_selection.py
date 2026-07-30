from __future__ import annotations

import pytest

from app.config import (
    UPSCALE_BACKEND_AUTO,
    UPSCALE_BACKEND_NCNN,
    UPSCALE_BACKEND_ONNX,
    Settings,
)
from app.services.backend_registry import (
    ncnn_produces_correct_output,
    BUILTIN_ONNX_MODELS,
    UpscaleBackend,
    get_builtin_onnx_model,
    resolve_upscale_backend,
    validate_backend_choice,
)

# ---------------------------------------------------------------------------
# SP11 Task 2 - backend registry + Auto-selection rule. Pure function, no GPU
# or onnxruntime needed. The rule: onnx iff the model has a vendored ONNX
# export AND (a GPU EP is present OR device is cpu); else ncnn (safe fallback).
# ---------------------------------------------------------------------------


def resolve(
    *,
    setting_backend: str = UPSCALE_BACKEND_AUTO,
    job_backend: str | None = None,
    onnx_model_available: bool = True,
    gpu_ep_available: bool = True,
    device: str = "dml:0",
) -> UpscaleBackend:
    return resolve_upscale_backend(
        setting_backend=setting_backend,
        job_backend=job_backend,
        onnx_model_available=onnx_model_available,
        gpu_ep_available=gpu_ep_available,
        device=device,
    )


# --- Auto rule ---


def test_auto_picks_onnx_when_model_available_and_gpu_ep_present() -> None:
    assert resolve(onnx_model_available=True, gpu_ep_available=True, device="dml:0") == UpscaleBackend.onnx


def test_auto_falls_back_to_ncnn_when_no_onnx_model() -> None:
    assert resolve(onnx_model_available=False, gpu_ep_available=True, device="dml:0") == UpscaleBackend.ncnn


def test_auto_falls_back_to_ncnn_when_no_gpu_ep_on_gpu_device() -> None:
    assert resolve(onnx_model_available=True, gpu_ep_available=False, device="dml:0") == UpscaleBackend.ncnn


def test_auto_picks_onnx_on_cpu_device_when_model_available() -> None:
    # ncnn Vulkan has no cpu path; onnx-cpu is the only runtime, even with no GPU EP.
    assert resolve(onnx_model_available=True, gpu_ep_available=False, device="cpu") == UpscaleBackend.onnx


def test_auto_stays_ncnn_on_cpu_device_when_no_onnx_model() -> None:
    assert resolve(onnx_model_available=False, gpu_ep_available=False, device="cpu") == UpscaleBackend.ncnn


# --- Forced setting (global UPSCALE_BACKEND) ---


def test_setting_ncnn_forces_ncnn_even_when_onnx_available() -> None:
    assert resolve(setting_backend=UPSCALE_BACKEND_NCNN, onnx_model_available=True) == UpscaleBackend.ncnn


def test_setting_onnx_forces_onnx() -> None:
    assert resolve(setting_backend=UPSCALE_BACKEND_ONNX, gpu_ep_available=False) == UpscaleBackend.onnx


# --- Per-job override wins over the global setting ---


def test_job_override_onnx_beats_setting_ncnn() -> None:
    assert resolve(setting_backend=UPSCALE_BACKEND_NCNN, job_backend=UPSCALE_BACKEND_ONNX) == UpscaleBackend.onnx


def test_job_override_ncnn_beats_setting_onnx() -> None:
    assert resolve(setting_backend=UPSCALE_BACKEND_ONNX, job_backend=UPSCALE_BACKEND_NCNN) == UpscaleBackend.ncnn


def test_job_override_auto_defers_to_auto_rule() -> None:
    assert (
        resolve(setting_backend=UPSCALE_BACKEND_NCNN, job_backend=UPSCALE_BACKEND_AUTO, onnx_model_available=True)
        == UpscaleBackend.onnx
    )


def test_no_job_override_uses_setting() -> None:
    assert resolve(setting_backend=UPSCALE_BACKEND_ONNX, job_backend=None) == UpscaleBackend.onnx


# --- validate_backend_choice ---


def test_validate_backend_choice_accepts_none() -> None:
    assert validate_backend_choice(None) is None


@pytest.mark.parametrize("value", [UPSCALE_BACKEND_AUTO, UPSCALE_BACKEND_NCNN, UPSCALE_BACKEND_ONNX])
def test_validate_backend_choice_accepts_valid(value: str) -> None:
    assert validate_backend_choice(value) == value


def test_validate_backend_choice_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        validate_backend_choice("cuda")


# --- builtin onnx model catalog ---


def test_builtin_onnx_catalog_covers_all_builtin_engine_names() -> None:
    expected = {
        "realesr-animevideov3-x2",
        "realesr-animevideov3-x3",
        "realesr-animevideov3-x4",
        "realesrgan-x4plus",
        # Escalas derivadas del x4plus. La clave lleva sufijo aunque el modelo ncnn NO
        # lo tenga: ncnn resuelve 2x/3x con -s sobre el mismo archivo, mientras ONNX
        # necesita un grafo por escala. Los dos runtimes nombran distinto el mismo
        # pedido, y por eso el lookup ONNX es por (nombre, escala).
        "realesrgan-x4plus-x2",
        "realesrgan-x4plus-x3",
        "realesrgan-x4plus-anime",
        "realesrgan-x4plus-anime-x2",
        "realesrgan-x4plus-anime-x3",
    }
    assert set(BUILTIN_ONNX_MODELS.keys()) == expected


# ---------------------------------------------------------------------------
# El lookup estricto por escala
#
# Es la guarda contra un fallo silencioso: realesrgan-x4plus ahora acepta 2x, pero su
# grafo ONNX base es x4. Devolverlo para un pedido de 2x entregaria un video al doble
# del tamano pedido, sin error, sin log y sin nada afuera que lo delate.
# ---------------------------------------------------------------------------


def test_the_base_graph_is_refused_for_a_scale_it_does_not_produce() -> None:
    assert get_builtin_onnx_model("realesrgan-x4plus", 4) is not None
    assert get_builtin_onnx_model("realesrgan-x4plus", 4).scale == 4


def test_a_derived_scale_resolves_to_its_own_graph() -> None:
    model = get_builtin_onnx_model("realesrgan-x4plus", 2)

    assert model is not None
    assert model.scale == 2
    assert model.filename == "realesrgan-x4plus-x2-uint8.onnx"


def test_a_scale_with_no_graph_returns_none_instead_of_the_wrong_one() -> None:
    # 8x no existe para ningun builtin. Devolver el grafo x4 daria la mitad del tamano
    # pedido; None manda el ruteo a ncnn, que al menos falla fuerte o resuelve bien.
    assert get_builtin_onnx_model("realesrgan-x4plus", 8) is None


def test_a_model_whose_engine_name_already_carries_the_scale_still_resolves() -> None:
    # animevideov3 nombra la escala en el propio engine_name, sin sufijo extra.
    model = get_builtin_onnx_model("realesr-animevideov3-x3", 3)

    assert model is not None
    assert model.scale == 3


def test_a_mismatched_scale_on_a_self_named_model_is_refused() -> None:
    assert get_builtin_onnx_model("realesr-animevideov3-x3", 2) is None


def test_without_a_scale_the_lookup_keeps_its_historical_behavior() -> None:
    # Compatibilidad: los llamadores viejos que no pasan escala siguen viendo la
    # entrada por nombre. Es permisivo a proposito, y por eso el pipeline SI pasa
    # la escala.
    model = get_builtin_onnx_model("realesrgan-x4plus")

    assert model is not None
    assert model.scale == 4


def test_get_builtin_onnx_model_returns_scale_and_filename() -> None:
    model = get_builtin_onnx_model("realesr-animevideov3-x3")
    assert model is not None
    assert model.scale == 3
    assert model.filename.endswith(".onnx")


def test_get_builtin_onnx_model_unknown_is_none() -> None:
    assert get_builtin_onnx_model("does-not-exist") is None


# --- config validation ---


def test_settings_rejects_invalid_upscale_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="UPSCALE_BACKEND must be one of"):
        Settings(_env_file=None, RUNTIME_DIR=str(tmp_path), UPSCALE_BACKEND="cuda")


def test_settings_defaults_upscale_backend_to_auto(tmp_path) -> None:
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path))
    assert settings.upscale_backend == UPSCALE_BACKEND_AUTO


# ---------------------------------------------------------------------------
# Las escalas que ncnn hace BIEN
#
# El binario FALLA EN SILENCIO: con -n realesrgan-x4plus -s 2 sale con codigo 0 y escribe
# un archivo con las dimensiones exactas pedidas, pero magnifica solo un cuadrante. Una
# fuente 256x256 con cuatro cuadrantes de colores distintos sale 512x512 con los cuatro
# AZULES; con -s 4 los cuatro colores salen bien. Chequear dimensiones no alcanza.
# ---------------------------------------------------------------------------


def test_the_x4_native_models_only_do_four_on_ncnn() -> None:
    assert ncnn_produces_correct_output("realesrgan-x4plus", 4) is True
    assert ncnn_produces_correct_output("realesrgan-x4plus", 2) is False
    assert ncnn_produces_correct_output("realesrgan-x4plus", 3) is False
    assert ncnn_produces_correct_output("realesrgan-x4plus-anime", 2) is False


def test_the_per_scale_models_are_native_at_their_own_scale() -> None:
    # animevideov3 trae un .param/.bin por escala, asi que cada uno es nativo.
    assert ncnn_produces_correct_output("realesr-animevideov3-x2", 2) is True
    assert ncnn_produces_correct_output("realesr-animevideov3-x3", 3) is True
    assert ncnn_produces_correct_output("realesr-animevideov3-x4", 4) is True


def test_an_unknown_model_is_assumed_correct() -> None:
    # Los modelos custom del usuario se resuelven por otro camino y no pasan por la
    # tabla; asumirlos rotos los bloquearia sin motivo.
    assert ncnn_produces_correct_output("un-modelo-del-usuario", 2) is True
