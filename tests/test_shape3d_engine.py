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

    def test_generating_without_the_model_says_how_to_install_it(self, tmp_path: Path) -> None:
        with pytest.raises(Shape3dUnavailable, match="download-shap-e"):
            Shape3dEngine(tmp_path).generate_from_text("una pieza")

    def test_an_empty_prompt_is_refused_before_loading_anything(self, tmp_path: Path) -> None:
        # Cargar el modelo cuesta 80 segundos: rechazar antes es la diferencia
        # entre un error inmediato y un minuto y medio perdido.
        with pytest.raises(Shape3dUnavailable, match="descripcion"):
            Shape3dEngine(tmp_path).generate_from_text("   ")


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
