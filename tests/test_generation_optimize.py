from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app.models import JobStatus
from app.services.device_semaphores import DeviceSemaphores
from app.services.generation_converter import GenerationModelConverter
from app.services.generation_installer import GenerationModelInstaller
from app.services.generation_optimize import (
    GIB,
    SD15,
    SDXL,
    OptimizeUnsupportedError,
    architecture_for,
    ensure_enough_ram,
    is_inpaint_merge,
    is_optimized,
    optimized_display_name,
    optimized_model_id,
)
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.hf_client import HfFile
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.onnx_fusion_patch import _get_num_heads, patch_ort_attention_num_heads
from app.services.progress import build_optimize_stages
from test_generation_installer import (
    FakeHfClient,
    FakeValidationPipeline,
    make_settings,
)

REPO_ID = "owner/sdxl-model"
SOURCE_MODEL_ID = "gen--owner--sdxl-model"


# ---------------------------------------------------------------------------
# El parche del bug de ORT 1.24.4 + NumPy 2
# ---------------------------------------------------------------------------


class _FakeOnnxModel:
    def __init__(self, constant: object) -> None:
        self._constant = constant

    def get_constant_value(self, _name: str) -> object:
        return self._constant

    def get_parent(self, _node: object, _index: int) -> object:
        return types.SimpleNamespace(op_type="Concat", input=["a", "b", "c", "d"])


class _FakeFusion:
    def __init__(self, constant: object) -> None:
        self.model = _FakeOnnxModel(constant)


def _make_fusion_module(name: str, shared_class: type | None = None) -> type:
    module = types.ModuleType(name)
    cls = shared_class or type(
        "FusionAttentionUnet", (), {"get_num_heads": lambda self, q, is_torch2=False: -1}
    )
    module.FusionAttentionUnet = cls  # type: ignore[attr-defined]
    sys.modules[name] = module
    return cls


def test_patch_reaches_every_loaded_copy_of_the_fusion_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # onnxruntime.transformers mete su propio directorio en sys.path: el MISMO
    # modulo queda cargado con nombres distintos. Parchear solo uno deja la
    # fusion rota en silencio.
    monkeypatch.setattr(
        "app.services.onnx_fusion_patch._import_fusion_modules", lambda: None
    )
    alias_a = _make_fusion_module("fake_ort_transformers_a")
    alias_b = _make_fusion_module("fake_ort_transformers_b")
    try:
        patched = patch_ort_attention_num_heads()

        assert patched >= 2
        assert alias_a.get_num_heads is _get_num_heads
        assert alias_b.get_num_heads is _get_num_heads
    finally:
        sys.modules.pop("fake_ort_transformers_a", None)
        sys.modules.pop("fake_ort_transformers_b", None)


def test_patched_num_heads_reads_a_one_element_array_that_numpy2_rejects() -> None:
    # int(ndarray de shape [1]) es exactamente lo que NumPy 2 dejo de aceptar y
    # lo que mataba la fusion antes de fusionar una sola atencion.
    fusion = _FakeFusion(np.array([8], dtype=np.int64))

    assert _get_num_heads(fusion, object(), is_torch2=True) == 8


def test_patched_num_heads_returns_zero_when_the_constant_is_not_usable() -> None:
    assert _get_num_heads(_FakeFusion(None), object(), is_torch2=True) == 0


def test_patched_num_heads_reads_the_torch1_reshape_shape() -> None:
    fusion = _FakeFusion(np.array([2, 77, 12, 64], dtype=np.int64))

    assert _get_num_heads(fusion, types.SimpleNamespace(input=["x", "shape"])) == 12


# ---------------------------------------------------------------------------
# Elegibilidad: arquitectura y RAM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "class_name",
    ["StableDiffusionXLPipeline", "ORTStableDiffusionXLPipeline", "OnnxStableDiffusionXLPipeline"],
)
def test_architecture_for_recognises_every_sdxl_class_prefix(class_name: str) -> None:
    assert architecture_for(class_name) is SDXL


@pytest.mark.parametrize(
    "class_name",
    ["StableDiffusionPipeline", "ORTStableDiffusionInpaintPipeline"],
)
def test_architecture_for_recognises_sd15_classes(class_name: str) -> None:
    assert architecture_for(class_name) is SD15


@pytest.mark.parametrize(
    "class_name",
    ["StableDiffusion3Pipeline", "FluxPipeline", "LatentConsistencyModelPipeline"],
)
def test_architecture_for_rejects_families_the_fusion_cannot_guarantee(
    class_name: str,
) -> None:
    with pytest.raises(OptimizeUnsupportedError, match=class_name):
        architecture_for(class_name)


def test_ensure_enough_ram_rejects_with_both_numbers_in_the_message() -> None:
    with pytest.raises(OptimizeUnsupportedError) as excinfo:
        ensure_enough_ram(SDXL, 8 * GIB)

    message = str(excinfo.value)
    assert "50.0 GiB" in message
    assert "8.0 GiB" in message


def test_ensure_enough_ram_admits_when_there_is_headroom() -> None:
    ensure_enough_ram(SD15, SD15.peak_ram_bytes)


def test_ensure_enough_ram_fails_open_when_ram_cannot_be_measured() -> None:
    # Fail-open: no poder medir nunca puede ser motivo de bloqueo (mismo criterio
    # que el resto de las sondas de capacidad del repo).
    ensure_enough_ram(SDXL, None)


def test_variant_naming_marks_the_model_as_a_separate_entry() -> None:
    assert optimized_model_id(SOURCE_MODEL_ID) == f"{SOURCE_MODEL_ID}--optimized"
    assert optimized_display_name("owner/sdxl") == "owner/sdxl (optimized)"
    assert is_optimized(optimized_model_id(SOURCE_MODEL_ID))
    assert not is_optimized(SOURCE_MODEL_ID)
    assert is_inpaint_merge(f"{SOURCE_MODEL_ID}--inpainting")


def test_copying_the_pipeline_skips_the_unet_graph_but_keeps_its_config(
    tmp_path: Path,
) -> None:
    from app.services.generation_graph_fusion import copy_pipeline_without_unet_graph

    source = _write_installed_pipeline(tmp_path / "installed", "StableDiffusionXLPipeline")
    dest = tmp_path / "staging"

    copy_pipeline_without_unet_graph(source, dest)

    # El grafo del UNet se reemplaza entero: copiar 5.1 GB para borrarlos es
    # medio minuto de disco tirado.
    assert not (dest / "unet" / "model.onnx").exists()
    assert not (dest / "unet" / "model.onnx_data").exists()
    assert (dest / "unet" / "config.json").is_file()
    assert (dest / "vae" / "model.onnx").read_bytes() == b"vae"
    assert (dest / "model_index.json").is_file()


def test_optimize_stages_cover_the_four_real_phases_plus_validation() -> None:
    keys = [stage.key for stage in build_optimize_stages()]

    assert keys == ["downloading", "exporting", "fusing", "converting", "validating"]
    assert sum(stage.weight for stage in build_optimize_stages()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# El carril de trabajo
# ---------------------------------------------------------------------------


def _torch_repo_files() -> list[HfFile]:
    return [
        HfFile(path="model_index.json", size=100),
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=1000),
        HfFile(path="unet/config.json", size=10),
        HfFile(path="vae/diffusion_pytorch_model.safetensors", size=500),
        HfFile(path="vae/config.json", size=10),
    ]


def _write_installed_pipeline(root: Path, declared_class: str) -> Path:
    (root / "unet").mkdir(parents=True, exist_ok=True)
    (root / "vae").mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": declared_class,
                "unet": ["diffusers", "OnnxRuntimeModel"],
                "vae": ["diffusers", "OnnxRuntimeModel"],
            }
        ),
        encoding="utf-8",
    )
    (root / "unet" / "config.json").write_text("{}", encoding="utf-8")
    (root / "unet" / "model.onnx").write_bytes(b"installed-graph")
    (root / "unet" / "model.onnx_data").write_bytes(b"installed-weights")
    (root / "vae" / "model.onnx").write_bytes(b"vae")
    return root


def make_optimize_converter(
    tmp_path: Path,
    *,
    export_fn=None,
    optimize_unet_fn=None,
    declared_class: str = "StableDiffusionXLPipeline",
):
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    hf_client = FakeHfClient(files=_torch_repo_files())
    hf_client.download_bytes_by_path = {
        "model_index.json": json.dumps({"_class_name": declared_class}).encode("utf-8")
    }
    installer = GenerationModelInstaller(
        settings,
        registry,
        hf_client,
        GpuSessionCoordinator(),
        DeviceSemaphores(settings),
    )
    installer._create_validation_pipeline = (  # type: ignore[method-assign]
        lambda pipeline_dir: FakeValidationPipeline()
    )
    converter = GenerationModelConverter(
        settings,
        installer,
        hf_client,
        export_fn=export_fn or _fake_export,
        optimize_unet_fn=optimize_unet_fn or _fake_optimize_unet,
    )
    installed_dir = _write_installed_pipeline(
        settings.models_path / "generation" / SOURCE_MODEL_ID, declared_class
    )
    registry.register(
        ModelEntry(
            id=SOURCE_MODEL_ID,
            name=REPO_ID,
            kind=ModelKind.diffusion_onnx,
            source=f"hf:{REPO_ID}",
            size_bytes=1,
            file_path=f"generation/{SOURCE_MODEL_ID}",
        )
    )
    return converter, installer, settings, registry, installed_dir


def _fake_export(src_dir, out_dir, on_component, dtype, atol) -> list[str]:
    assert dtype == "fp32", "la fusion solo matchea sobre un export fp32"
    (out_dir / "unet").mkdir(parents=True, exist_ok=True)
    (out_dir / "unet" / "model.onnx").write_bytes(b"fp32-graph")
    on_component("1-UNet2DConditionModel")
    return ["1-UNet2DConditionModel"]


def _fake_optimize_unet(fp32_unet_onnx: Path, dest_unet_dir: Path, on_converting) -> dict:
    assert fp32_unet_onnx.read_bytes() == b"fp32-graph"
    on_converting()
    for stale in dest_unet_dir.iterdir():
        if stale.name != "config.json":
            stale.unlink()
    (dest_unet_dir / "model.onnx").write_bytes(b"fused-graph")
    return {"MultiHeadAttention": 140, "GroupNorm": 46, "SkipGroupNorm": 0}


def run_optimize(converter: GenerationModelConverter, installed_dir: Path) -> str:
    async def _run() -> str:
        conversion_id = await converter.optimize_installed(
            source_model_id=SOURCE_MODEL_ID,
            source_model_name=REPO_ID,
            repo_id=REPO_ID,
            checkpoint_path=None,
            installed_dir=installed_dir,
        )
        await converter._process_next()
        return conversion_id

    return asyncio.run(_run())


def test_optimize_registers_a_new_variant_and_leaves_the_original_installed(
    tmp_path: Path,
) -> None:
    converter, _installer, settings, registry, installed_dir = make_optimize_converter(
        tmp_path
    )

    conversion_id = run_optimize(converter, installed_dir)

    job = converter.status(conversion_id)
    assert job is not None and job.status is JobStatus.completed
    variant_id = optimized_model_id(SOURCE_MODEL_ID)
    variant = registry.get(variant_id)
    assert variant is not None
    assert variant.status is ModelStatus.installed
    assert variant.name == optimized_display_name(REPO_ID)
    original = registry.get(SOURCE_MODEL_ID)
    assert original is not None and original.status is ModelStatus.installed
    # El original NO se toca: la fusion cambia la imagen con la misma semilla.
    assert (installed_dir / "unet" / "model.onnx").read_bytes() == b"installed-graph"
    promoted = settings.models_path / "generation" / variant_id
    assert (promoted / "unet" / "model.onnx").read_bytes() == b"fused-graph"
    # Los pesos externos viejos no pueden sobrevivir al reemplazo del grafo.
    assert not (promoted / "unet" / "model.onnx_data").exists()


def _record_stages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import app.services.generation_converter as module

    stages: list[str] = []
    real = module.advance_optimize_stage

    def recording(job, stage_key: str) -> None:
        stages.append(stage_key)
        real(job, stage_key)

    monkeypatch.setattr(module, "advance_optimize_stage", recording)
    return stages


def test_optimize_walks_every_stage_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def recording_export(src_dir, out_dir, on_component, dtype, atol):
        seen.append("export")
        return _fake_export(src_dir, out_dir, on_component, dtype, atol)

    def recording_optimize(fp32_unet_onnx, dest_unet_dir, on_converting):
        seen.append("fuse")
        return _fake_optimize_unet(fp32_unet_onnx, dest_unet_dir, on_converting)

    converter, _installer, _settings, _registry, installed_dir = make_optimize_converter(
        tmp_path, export_fn=recording_export, optimize_unet_fn=recording_optimize
    )
    stages = _record_stages(monkeypatch)

    conversion_id = run_optimize(converter, installed_dir)

    job = converter.status(conversion_id)
    assert job is not None
    assert seen == ["export", "fuse"]
    # "converting" sale del callback que dispara el propio fusionador: fusion y
    # conversion a fp16 son las dos etapas largas (222 s y 104 s en SDXL) y
    # mostrarlas como una sola deja la barra clavada la mitad del tiempo.
    assert stages == ["downloading", "exporting", "fusing", "converting", "validating"]
    assert job.metadata["stage"] == "completed"
    assert job.metadata["progress"] == 1.0
    assert job.metadata["fusedOperators"]["SkipGroupNorm"] == 0


def test_optimize_progress_grows_monotonically_across_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter, _installer, _settings, _registry, installed_dir = make_optimize_converter(
        tmp_path
    )
    progress: list[float] = []
    import app.services.generation_converter as module

    real = module.advance_optimize_stage

    def recording(job, stage_key: str) -> None:
        real(job, stage_key)
        progress.append(job.metadata["progress"])

    monkeypatch.setattr(module, "advance_optimize_stage", recording)

    run_optimize(converter, installed_dir)

    assert progress == sorted(progress)
    assert progress[0] == 0.0
    assert 0.0 < progress[-1] < 1.0


def test_optimize_can_be_cancelled_between_stages(tmp_path: Path) -> None:
    converter, _installer, _settings, registry, installed_dir = make_optimize_converter(
        tmp_path
    )

    async def _run() -> str:
        conversion_id = await converter.optimize_installed(
            source_model_id=SOURCE_MODEL_ID,
            source_model_name=REPO_ID,
            repo_id=REPO_ID,
            checkpoint_path=None,
            installed_dir=installed_dir,
        )
        # El corte cae en el limite de etapa: el export corre dentro de una
        # libreria que no se puede interrumpir a la mitad.
        converter.cancel(conversion_id)
        await converter._process_next()
        return conversion_id

    conversion_id = asyncio.run(_run())

    job = converter.status(conversion_id)
    assert job is not None and job.status is JobStatus.cancelled
    assert job.error is None
    assert registry.get(optimized_model_id(SOURCE_MODEL_ID)) is None or (
        registry.get(optimized_model_id(SOURCE_MODEL_ID)).status  # type: ignore[union-attr]
        is not ModelStatus.installed
    )


def test_optimize_failing_validation_never_registers_an_installed_variant(
    tmp_path: Path,
) -> None:
    converter, installer, settings, registry, installed_dir = make_optimize_converter(
        tmp_path
    )

    def broken_pipeline(_pipeline_dir):
        raise RuntimeError("LoadModel failed: SkipGroupNorm no tiene kernel")

    installer._create_validation_pipeline = broken_pipeline  # type: ignore[method-assign]

    conversion_id = run_optimize(converter, installed_dir)

    job = converter.status(conversion_id)
    assert job is not None and job.status is JobStatus.failed
    variant_id = optimized_model_id(SOURCE_MODEL_ID)
    variant = registry.get(variant_id)
    assert variant is not None and variant.status is ModelStatus.error
    assert not (settings.models_path / "generation" / variant_id).exists()
    assert registry.get(SOURCE_MODEL_ID).status is ModelStatus.installed  # type: ignore[union-attr]
