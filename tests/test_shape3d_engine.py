from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.services.engines.shape3d import (
    Shape3dEngine,
    Shape3dUnavailable,
    _as_triangles,
    assert_usable_mesh,
)
from app.services.missing_pack import PACK_LABELS

# ---------------------------------------------------------------------------
# El fallo caro de estos modelos NO es una excepcion: es una salida con la forma
# correcta y el contenido roto. Ya paso dos veces en este repo — la variante fp16
# de Kokoro devolvia audio todo NaN con la duracion correcta, y la conversion de
# voz devolvia algo con la forma justa que sonaba a nada.
#
# Asi que la malla se revisa antes de entregarla, igual que el audio.
# ---------------------------------------------------------------------------


class FakeMesh:
    def __init__(self, verts, faces) -> None:
        self.verts = verts
        self.faces = faces


class TestMeshGuard:
    def test_a_mesh_with_nan_is_refused(self) -> None:
        with pytest.raises(Shape3dUnavailable, match="NaN|invalid"):
            assert_usable_mesh(np.array([[[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [0.0, 1.0, 0.0]]]))

    def test_a_mesh_with_infinity_is_refused(self) -> None:
        with pytest.raises(Shape3dUnavailable):
            assert_usable_mesh(np.array([[[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0], [0.0, 1.0, 0.0]]]))

    def test_an_empty_mesh_is_refused(self) -> None:
        with pytest.raises(Shape3dUnavailable, match="vacia"):
            assert_usable_mesh(np.empty((0, 3, 3)))

    def test_a_real_mesh_passes(self) -> None:
        assert_usable_mesh(np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))


class TestDecoderOutput:
    def test_vertices_and_faces_become_triangles(self) -> None:
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])

        triangulos = _as_triangles(FakeMesh(verts, faces))

        assert triangulos.shape == (2, 3, 3)
        assert triangulos[0][1] == pytest.approx([1.0, 0.0, 0.0])

    def test_an_empty_decoder_output_is_refused(self) -> None:
        with pytest.raises(Shape3dUnavailable):
            _as_triangles(FakeMesh(np.empty((0, 3)), np.empty((0, 3), dtype=int)))


class TestAvailability:
    def test_an_empty_folder_is_not_available(self, tmp_path: Path) -> None:
        assert not Shape3dEngine(tmp_path).available()

    def test_a_folder_with_the_index_is_available(self, tmp_path: Path) -> None:
        (tmp_path / "model_index.json").write_text(json.dumps({}), encoding="utf-8")

        assert Shape3dEngine(tmp_path).available()

    def test_generating_without_the_model_says_what_is_missing_without_dictating_commands(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(Shape3dUnavailable) as error:
            Shape3dEngine(tmp_path).generate_from_text("una pieza")

        # Identificar lo que falta, no la redaccion exacta: el mensaje se puede
        # retocar sin que nada este mal.
        assert PACK_LABELS["shap-e"] in str(error.value)
        # La regresion que hay que evitar: mandar al usuario a una terminal
        # cuando la app baja el paquete sola.
        assert ".ps1" not in str(error.value)

    def test_an_empty_prompt_is_refused_before_loading_anything(self, tmp_path: Path) -> None:
        # Cargar el modelo cuesta 80 segundos: rechazar antes es la diferencia
        # entre un error inmediato y un minuto y medio perdido.
        with pytest.raises(Shape3dUnavailable, match="descripcion"):
            Shape3dEngine(tmp_path).generate_from_text("   ")


def make_photo(tmp_path: Path, mode: str = "RGBA") -> Path:
    from PIL import Image

    destino = tmp_path / "foto.png"
    Image.new(mode, (8, 8), "white").save(destino)
    return destino


class TestImageAvailability:
    def test_without_an_image_dir_it_is_not_available(self, tmp_path: Path) -> None:
        assert not Shape3dEngine(tmp_path).available_image()

    def test_the_text_pack_alone_does_not_enable_photo(self, tmp_path: Path) -> None:
        # Son DOS repos de pesos: tener el de texto no habilita el de foto.
        (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")

        assert not Shape3dEngine(tmp_path, tmp_path / "img2img").available_image()

    def test_a_folder_with_the_image_index_is_available(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "img2img"
        image_dir.mkdir()
        (image_dir / "model_index.json").write_text("{}", encoding="utf-8")

        assert Shape3dEngine(tmp_path, image_dir).available_image()

    def test_generating_without_the_image_pack_says_what_is_missing(
        self, tmp_path: Path
    ) -> None:
        foto = make_photo(tmp_path)

        with pytest.raises(Shape3dUnavailable) as error:
            Shape3dEngine(tmp_path, tmp_path / "img2img").generate_from_image(foto)

        assert PACK_LABELS["shap-e-img2img"] in str(error.value)
        assert ".ps1" not in str(error.value)


class TestImagePipelineWiring:
    def make_engine(self, tmp_path: Path, pipe: Any) -> Shape3dEngine:
        image_dir = tmp_path / "img2img"
        image_dir.mkdir()
        (image_dir / "model_index.json").write_text("{}", encoding="utf-8")
        engine = Shape3dEngine(tmp_path / "texto", image_dir)
        engine._image_pipe = pipe
        return engine

    def test_the_photo_and_the_spike_settings_reach_the_model(self, tmp_path: Path) -> None:
        recibido: dict = {}

        class FakePipe:
            def __call__(self, image, **kwargs):
                recibido["image"] = image
                recibido.update(kwargs)
                verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                return type("S", (), {"images": [FakeMesh(verts, np.array([[0, 1, 2]]))]})()

        engine = self.make_engine(tmp_path, FakePipe())
        engine.generate_from_image(make_photo(tmp_path, mode="RGBA"))

        # La foto entra convertida a RGB: un PNG con alfa no puede reventar la
        # tuberia a los dos minutos de CPU.
        assert recibido["image"].mode == "RGB"
        # Los valores del spike: 16 pasos y guia 3.0 (mucho mas baja que texto).
        assert recibido["num_inference_steps"] == 16
        assert recibido["guidance_scale"] == 3.0
        assert recibido["frame_size"] == 64
        assert recibido["output_type"] == "mesh"

    def test_a_model_that_returns_garbage_is_caught(self, tmp_path: Path) -> None:
        class BrokenPipe:
            def __call__(self, image, **kwargs):
                verts = np.full((3, 3), np.nan)
                return type("S", (), {"images": [FakeMesh(verts, np.array([[0, 1, 2]]))]})()

        engine = self.make_engine(tmp_path, BrokenPipe())

        with pytest.raises(Shape3dUnavailable):
            engine.generate_from_image(make_photo(tmp_path))


class TestPipelineWiring:
    def make_engine(self, tmp_path: Path, pipe: Any) -> Shape3dEngine:
        (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
        engine = Shape3dEngine(tmp_path)
        engine._text_pipe = pipe
        return engine

    def test_the_prompt_and_the_settings_reach_the_model(self, tmp_path: Path) -> None:
        recibido: dict = {}

        class FakePipe:
            def __call__(self, prompt, **kwargs):
                recibido["prompt"] = prompt
                recibido.update(kwargs)
                verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                return type("S", (), {"images": [FakeMesh(verts, np.array([[0, 1, 2]]))]})()

        engine = self.make_engine(tmp_path, FakePipe())
        engine.generate_from_text("un soporte", steps=8, guidance=12.0, frame_size=32)

        assert recibido["prompt"] == "un soporte"
        assert recibido["num_inference_steps"] == 8
        assert recibido["guidance_scale"] == 12.0
        assert recibido["frame_size"] == 32
        # Sin esto vuelve una imagen renderizada en vez de geometria.
        assert recibido["output_type"] == "mesh"

    def test_a_model_that_returns_garbage_is_caught(self, tmp_path: Path) -> None:
        class BrokenPipe:
            def __call__(self, prompt, **kwargs):
                verts = np.full((3, 3), np.nan)
                return type("S", (), {"images": [FakeMesh(verts, np.array([[0, 1, 2]]))]})()

        engine = self.make_engine(tmp_path, BrokenPipe())

        with pytest.raises(Shape3dUnavailable):
            engine.generate_from_text("un soporte")
