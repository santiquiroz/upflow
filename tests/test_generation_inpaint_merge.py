from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from app.services.generation_inpaint_merge import (
    INPAINT_BASE_REPOS,
    InpaintMergeIncompatibleError,
    InpaintMergeUnsupportedError,
    merge_family_for,
    merge_to_inpaint_dir,
    merge_unet_state_dicts,
)

# ---------------------------------------------------------------------------
# Todo sintético: tensores chicos y dirs mínimos, sin descargar pesos ni GPU.
# ---------------------------------------------------------------------------


# --- (a) add-difference exacto en shapes iguales ---------------------------


def test_add_difference_is_exact_for_equal_shapes() -> None:
    user = {"w": torch.full((4, 4), 3.0)}
    base = {"w": torch.full((4, 4), 1.0)}
    inpaint = {"w": torch.full((4, 4), 10.0)}

    merged = merge_unet_state_dicts(user, base, inpaint)

    assert torch.equal(merged["w"], torch.full((4, 4), 12.0))
    assert merged["w"].dtype == inpaint["w"].dtype


# --- (b) conv_in: delta en los 4 originales, extras intactos ---------------


def test_conv_in_merges_originals_and_keeps_extra_channels_bit_exact() -> None:
    torch.manual_seed(0)
    user = {"conv_in.weight": torch.randn(8, 4, 3, 3)}
    base = {"conv_in.weight": torch.randn(8, 4, 3, 3)}
    inpaint = {"conv_in.weight": torch.randn(8, 9, 3, 3)}

    merged = merge_unet_state_dicts(user, base, inpaint)
    result = merged["conv_in.weight"]

    assert result.shape == (8, 9, 3, 3)
    expected_first4 = inpaint["conv_in.weight"][:, :4] + (
        user["conv_in.weight"] - base["conv_in.weight"]
    )
    assert torch.equal(result[:, :4], expected_first4)
    assert torch.equal(result[:, 4:], inpaint["conv_in.weight"][:, 4:])


# --- (c) claves asimétricas ------------------------------------------------


def test_a_key_only_in_the_inpaint_is_copied_verbatim() -> None:
    shared = torch.ones(2, 2)
    extra = torch.full((3,), 7.0)
    user = {"w": shared.clone()}
    base = {"w": shared.clone()}
    inpaint = {"w": shared.clone(), "solo_inpaint": extra}

    merged = merge_unet_state_dicts(user, base, inpaint)

    assert torch.equal(merged["solo_inpaint"], extra)


def test_a_key_only_in_the_user_is_a_clear_error() -> None:
    shared = torch.ones(2, 2)
    user = {"w": shared.clone(), "capa_inventada.weight": torch.ones(3)}
    base = {"w": shared.clone()}
    inpaint = {"w": shared.clone()}

    with pytest.raises(InpaintMergeIncompatibleError, match="capa_inventada"):
        merge_unet_state_dicts(user, base, inpaint)


def test_mismatched_shapes_outside_conv_in_are_a_clear_error() -> None:
    user = {"w": torch.ones(2, 2)}
    base = {"w": torch.ones(2, 2)}
    inpaint = {"w": torch.ones(4, 4)}

    with pytest.raises(InpaintMergeIncompatibleError, match="formas incompatibles"):
        merge_unet_state_dicts(user, base, inpaint)


def test_user_and_base_disagreeing_on_a_shape_is_a_clear_error() -> None:
    user = {"w": torch.ones(2, 2)}
    base = {"w": torch.ones(3, 3)}
    inpaint = {"w": torch.ones(2, 2)}

    with pytest.raises(InpaintMergeIncompatibleError, match="misma arquitectura"):
        merge_unet_state_dicts(user, base, inpaint)


# --- (d) aritmética float32: user == base devuelve el inpaint bit a bit ----


def test_user_equal_to_base_returns_the_inpaint_bit_exact_in_fp16() -> None:
    torch.manual_seed(1)
    base = {
        "w": torch.randn(4, 4).half(),
        "conv_in.weight": torch.randn(8, 4, 3, 3).half(),
    }
    user = {key: value.clone() for key, value in base.items()}
    inpaint = {
        "w": torch.randn(4, 4).half(),
        "conv_in.weight": torch.randn(8, 9, 3, 3).half(),
    }

    merged = merge_unet_state_dicts(user, base, inpaint)

    assert merged["w"].dtype == torch.float16
    assert torch.equal(merged["w"], inpaint["w"])
    assert torch.equal(merged["conv_in.weight"], inpaint["conv_in.weight"])


# --- (e) merge_family_for --------------------------------------------------


@pytest.mark.parametrize(
    ("declared", "family"),
    [
        ("StableDiffusionXLPipeline", "sdxl"),
        ("ORTStableDiffusionXLPipeline", "sdxl"),
        ("StableDiffusionPipeline", "sd15"),
        ("ORTStableDiffusionPipeline", "sd15"),
    ],
)
def test_known_classes_resolve_to_their_family(declared: str, family: str) -> None:
    assert merge_family_for(declared) == family
    assert set(INPAINT_BASE_REPOS[family]) == {"base", "inpaint"}


def test_architectures_without_official_inpaint_are_rejected() -> None:
    with pytest.raises(InpaintMergeUnsupportedError, match="no tiene checkpoint de inpainting"):
        merge_family_for("StableDiffusion3Pipeline")
    with pytest.raises(InpaintMergeUnsupportedError, match="no tiene checkpoint de inpainting"):
        merge_family_for("LatentConsistencyModelPipeline")


def test_an_unknown_class_gets_the_unknown_reason() -> None:
    with pytest.raises(InpaintMergeUnsupportedError, match="desconocida"):
        merge_family_for("KandinskyV22Pipeline")


# --- (f) merge_to_inpaint_dir con dirs sintéticos --------------------------


class _FakeUnet:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.state = state
        self.saved_to: Path | None = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.state

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        # Mismo contrato que el load_state_dict estricto de torch.
        assert set(state) == set(self.state)
        self.state = dict(state)

    def save_pretrained(self, path: str, safe_serialization: bool = False) -> None:
        assert safe_serialization is True
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text('{"in_channels": 9}', encoding="utf-8")
        self.saved_to = dest


def _make_pipeline_dirs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    user_dir = tmp_path / "user"
    base_dir = tmp_path / "base"
    inpaint_dir = tmp_path / "inpaint"
    out_dir = tmp_path / "out"
    # El user trae los componentes de un SD15 SIN feature_extractor ni
    # safety_checker: el model_index final tiene que soltarlos.
    for name in ("vae", "text_encoder", "tokenizer", "scheduler"):
        (user_dir / name).mkdir(parents=True)
        (user_dir / name / "config.json").write_text(
            json.dumps({"source": f"user-{name}"}), encoding="utf-8"
        )
    base_dir.mkdir()
    inpaint_dir.mkdir()
    (inpaint_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionInpaintPipeline",
                "_diffusers_version": "0.39.0",
                "feature_extractor": ["transformers", "CLIPImageProcessor"],
                "requires_safety_checker": False,
                "safety_checker": [None, None],
                "scheduler": ["diffusers", "PNDMScheduler"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        ),
        encoding="utf-8",
    )
    return user_dir, base_dir, inpaint_dir, out_dir


def _install_fake_unets(
    monkeypatch: pytest.MonkeyPatch,
    fakes: dict[Path, _FakeUnet],
) -> None:
    import diffusers

    def fake_from_pretrained(
        pretrained_path: str,
        *,
        subfolder: str,
        torch_dtype: torch.dtype,
        low_cpu_mem_usage: bool,
    ) -> _FakeUnet:
        assert subfolder == "unet"
        assert torch_dtype is torch.float16
        assert low_cpu_mem_usage is True
        return fakes[Path(pretrained_path)]

    monkeypatch.setattr(
        diffusers.UNet2DConditionModel, "from_pretrained", fake_from_pretrained
    )


def _make_fakes(dirs: tuple[Path, Path, Path]) -> dict[Path, _FakeUnet]:
    torch.manual_seed(2)
    user_dir, base_dir, inpaint_dir = dirs
    return {
        user_dir: _FakeUnet(
            {
                "conv_in.weight": torch.randn(8, 4, 3, 3).half(),
                "w": torch.randn(4, 4).half(),
            }
        ),
        base_dir: _FakeUnet(
            {
                "conv_in.weight": torch.randn(8, 4, 3, 3).half(),
                "w": torch.randn(4, 4).half(),
            }
        ),
        inpaint_dir: _FakeUnet(
            {
                "conv_in.weight": torch.randn(8, 9, 3, 3).half(),
                "w": torch.randn(4, 4).half(),
            }
        ),
    }


def test_merge_to_inpaint_dir_assembles_a_complete_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_dir, base_dir, inpaint_dir, out_dir = _make_pipeline_dirs(tmp_path)
    fakes = _make_fakes((user_dir, base_dir, inpaint_dir))
    inputs = {path: dict(fake.state) for path, fake in fakes.items()}
    _install_fake_unets(monkeypatch, fakes)
    stages: list[str] = []

    result = merge_to_inpaint_dir(
        user_dir, base_dir, inpaint_dir, out_dir, progress_cb=stages.append
    )

    assert result == out_dir
    assert stages == ["loading", "merging", "saving"]
    # El unet guardado es el del INPAINT (config con in_channels=9) con el
    # state dict mergeado adentro.
    assert fakes[inpaint_dir].saved_to == out_dir / "unet"
    expected = merge_unet_state_dicts(
        inputs[user_dir], inputs[base_dir], inputs[inpaint_dir]
    )
    saved = fakes[inpaint_dir].state
    assert set(saved) == set(expected)
    for key in expected:
        assert torch.equal(saved[key], expected[key])
    for name in ("vae", "text_encoder", "tokenizer", "scheduler"):
        content = json.loads(
            (out_dir / name / "config.json").read_text(encoding="utf-8")
        )
        assert content == {"source": f"user-{name}"}


def test_the_model_index_is_the_inpaints_intersected_with_whats_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_dir, base_dir, inpaint_dir, out_dir = _make_pipeline_dirs(tmp_path)
    _install_fake_unets(monkeypatch, _make_fakes((user_dir, base_dir, inpaint_dir)))

    merge_to_inpaint_dir(user_dir, base_dir, inpaint_dir, out_dir)

    index = json.loads((out_dir / "model_index.json").read_text(encoding="utf-8"))
    assert index["_class_name"] == "StableDiffusionInpaintPipeline"
    # Componentes que el user no trae desaparecen; los flags escalares quedan.
    assert set(index) == {
        "_class_name",
        "_diffusers_version",
        "requires_safety_checker",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "unet",
        "vae",
    }
    assert index["unet"] == ["diffusers", "UNet2DConditionModel"]
    assert index["requires_safety_checker"] is False


def test_merge_to_inpaint_dir_starts_from_a_clean_out_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_dir, base_dir, inpaint_dir, out_dir = _make_pipeline_dirs(tmp_path)
    _install_fake_unets(monkeypatch, _make_fakes((user_dir, base_dir, inpaint_dir)))
    out_dir.mkdir()
    stale = out_dir / "restos-de-un-merge-cortado.txt"
    stale.write_text("stale", encoding="utf-8")

    merge_to_inpaint_dir(user_dir, base_dir, inpaint_dir, out_dir)

    assert not stale.exists()
    assert (out_dir / "unet" / "config.json").is_file()
