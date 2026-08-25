"""Tests de las tools MCP que hacen algo mas que reenviar la llamada.

Se intercepta el transporte HTTP igual que en test_mcp_server.py: lo que se
prueba es el contrato del cliente —rutas, forma del cuerpo, encadenado de dos
llamadas— y no la logica del servidor.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.mcp import client as mcp_client
from app.mcp.server import (
    upflow_delete_saved_prompt,
    upflow_generate_3d,
    upflow_reference_scene,
    upflow_repair_mesh,
    upflow_segment_object,
    upflow_sheet_views,
    upflow_update_setting,
)

PNG_FALSO = b"\x89PNG\r\n\x1a\n" + b"0" * 32


@pytest.fixture(autouse=True)
def reset_mcp_client():
    yield
    mcp_client._client = None
    mcp_client._login_attempted = False


def install_mock(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    transport = httpx.MockTransport(handler)
    mock = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    monkeypatch.setattr(mcp_client, "_get_client", lambda: mock)


def archivo(tmp_path: Path, nombre: str, contenido: bytes = b"datos") -> str:
    ruta = tmp_path / nombre
    ruta.write_bytes(contenido)
    return str(ruta)


async def test_segmentar_devuelve_un_token_reusable_y_no_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # La ruta devuelve un PNG crudo, que no sirve para encadenar. La tool tiene
    # que volver a subirlo y entregar el token que consume insert-object.
    vistas: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        vistas.append(request.url.path)
        if request.url.path == "/api/v1/editor/segment":
            cuerpo = json.loads(request.content)
            assert cuerpo == {"imageToken": "abc", "x": 10.0, "y": 20.0}
            return httpx.Response(200, content=PNG_FALSO, headers={"content-type": "image/png"})
        return httpx.Response(201, json={"initImageToken": "mask-token"})

    install_mock(monkeypatch, handler)
    destino = tmp_path / "mascara.png"
    salida = json.loads(await upflow_segment_object("abc", 10.0, 20.0, str(destino)))

    assert vistas == ["/api/v1/editor/segment", "/api/v1/generation/init-image"]
    assert salida["maskToken"] == "mask-token"
    assert salida["bytes"] == len(PNG_FALSO)
    assert destino.read_bytes() == PNG_FALSO


async def test_segmentar_sin_destino_no_escribe_nada(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/editor/segment":
            return httpx.Response(200, content=PNG_FALSO, headers={"content-type": "image/png"})
        return httpx.Response(201, json={"initImageToken": "mask-token"})

    install_mock(monkeypatch, handler)
    salida = json.loads(await upflow_segment_object("abc", 1, 2))

    assert "outputPath" not in salida
    assert not list(tmp_path.iterdir())


async def test_segmentar_sin_el_pack_devuelve_el_error_y_no_sube_nada(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sin MobileSAM la ruta contesta JSON, no PNG. Tratarlo como bytes subiria
    # un mensaje de error como si fuera una mascara.
    vistas: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        vistas.append(request.url.path)
        return httpx.Response(503, json={"detail": "Falta el modelo de seleccion por toque"})

    install_mock(monkeypatch, handler)
    salida = await upflow_segment_object("abc", 1, 2)

    assert salida.startswith("Error")
    assert vistas == ["/api/v1/editor/segment"]


async def test_la_escena_de_referencia_baja_el_blend_cuando_se_pide_destino(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            cuerpo = json.loads(request.content)
            assert cuerpo == {"token": "t1", "heightMeters": 1.7}
            return httpx.Response(
                201,
                json={"token": "t1", "downloadUrl": "/api/v1/model3d/scene/t1", "heightMeters": 1.7, "placed": []},
            )
        return httpx.Response(200, content=b"BLENDER-FAKE")

    install_mock(monkeypatch, handler)
    destino = tmp_path / "escena.blend"
    salida = json.loads(await upflow_reference_scene("t1", 1.7, str(destino)))

    assert salida["outputPath"] == str(destino)
    assert destino.read_bytes() == b"BLENDER-FAKE"


async def test_la_escena_sin_destino_no_descarga(monkeypatch: pytest.MonkeyPatch) -> None:
    metodos: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        metodos.append(request.method)
        return httpx.Response(201, json={"token": "t1", "downloadUrl": "/x", "heightMeters": 1.7, "placed": []})

    install_mock(monkeypatch, handler)
    await upflow_reference_scene("t1")

    assert metodos == ["POST"]


async def test_reparar_malla_baja_la_reparada_aunque_no_haya_cerrado(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Se entrega igual cuando quedo abierta: el reporte dice la verdad y quien
    # pide decide. Callarla seria el falso positivo que da confianza.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"canPrint": False, "watertight": False, "manifold": True,
                      "triangleCount": 10, "blockers": ["sigue abierta"],
                      "downloadUrl": "/api/v1/print/repaired/tok"},
            )
        return httpx.Response(200, content=b"STL")

    install_mock(monkeypatch, handler)
    destino = tmp_path / "reparada.stl"
    salida = json.loads(await upflow_repair_mesh(archivo(tmp_path, "rota.stl"), str(destino)))

    assert salida["canPrint"] is False
    assert salida["blockers"] == ["sigue abierta"]
    assert destino.read_bytes() == b"STL"


async def test_borrar_un_prompt_traduce_el_204_sin_cuerpo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/generation/saved-prompts/p1"
        return httpx.Response(204)

    install_mock(monkeypatch, handler)
    assert json.loads(await upflow_delete_saved_prompt("p1")) == {"ok": True}


async def test_cambiar_un_ajuste_manda_clave_y_valor_por_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert json.loads(request.content) == {"key": "CAD_LLM_MODEL", "value": "qwen"}
        return httpx.Response(200, json={"key": "CAD_LLM_MODEL"})

    install_mock(monkeypatch, handler)
    assert json.loads(await upflow_update_setting("CAD_LLM_MODEL", "qwen"))["key"] == "CAD_LLM_MODEL"


async def test_partir_la_hoja_manda_el_conteo_esperado(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/model3d/sheet/views"
        assert b"expectedViews" in request.content
        return httpx.Response(201, json={"token": "t", "views": [], "warnings": []})

    install_mock(monkeypatch, handler)
    salida = json.loads(await upflow_sheet_views(archivo(tmp_path, "hoja.png"), 3))

    assert salida["token"] == "t"


async def test_generar_3d_desde_foto_sube_la_imagen_y_cambia_el_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Pasar image_path tiene que implicar source=photo: dejar "mesh" mandaria la
    # foto al motor de texto, que es el bug que tenia la tuberia vieja.
    cuerpos: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/generation/init-image":
            return httpx.Response(201, json={"initImageToken": "img-1"})
        cuerpos.append(json.loads(request.content))
        return httpx.Response(202, json={"id": "j1", "status": "queued"})

    install_mock(monkeypatch, handler)
    await upflow_generate_3d("", "mesh", "ender-3", 0.0, archivo(tmp_path, "foto.png"))

    assert cuerpos[0]["source"] == "photo"
    assert cuerpos[0]["imageToken"] == "img-1"
