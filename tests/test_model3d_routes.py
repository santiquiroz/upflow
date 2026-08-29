from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image

from app.api.routes import (
    build_model3d_reference_scene,
    download_reference_scene,
    download_sheet_view,
    model3d_capabilities,
)
from app.config import Settings
from app.schemas import ReferenceSceneRequest
from app.api import routes
from app.services import model3d_service
from app.services.blender_service import BlenderBuild
from app.services.missing_pack import PACK_LABELS
from app.services.storage import StorageService

VIEW_SIZE = (60, 200)


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


def write_views(settings: Settings, token: str, names: tuple[str, ...]) -> Path:
    carpeta = settings.outputs_path / f"{token}.views"
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre in names:
        lienzo = Image.new("RGB", VIEW_SIZE, "white")
        for x in range(10, 50):
            for y in range(5, 195):
                lienzo.putpixel((x, y), (0, 0, 0))
        lienzo.save(carpeta / f"{nombre}.png")
    return carpeta


class RequestConDuenoLocal:
    """El camino de los tests de ruta: sin auth, el dueño es la instalación.

    Los tokens se registran por instancia y no en la clase: compartir el
    diccionario dejaria que un test le prestara sus tokens al siguiente.
    """

    def __init__(self, *tokens: str) -> None:
        estado = type("Estado", (), {"print_token_owners": {t: "local" for t in tokens}})()
        self.app = type("App", (), {"state": estado})()
        # Sin auth encendida no hay usuario en el request y el dueño es "local",
        # que es exactamente lo que hace la instalación de escritorio.
        self.state = type("EstadoRequest", (), {"current_user": None})()


class UsuarioLocal:
    id = "local"
    permissions: frozenset = frozenset()


@pytest.mark.asyncio
async def test_capabilities_reporta_ausencia_en_vez_de_fallar(tmp_path: Path, monkeypatch):
    # Preguntar por una capacidad ausente NO es un error: es la respuesta. Si
    # esta ruta tirara, la pantalla no podria decir que falta Blender.
    monkeypatch.setattr(routes, "blender_probe", lambda *_a, **_k: None)

    respuesta = await model3d_capabilities(settings_dep=make_settings(tmp_path))

    assert respuesta.blender.found is False
    assert respuesta.unlocked == []
    assert PACK_LABELS["blender"] in (respuesta.missing or "")


@pytest.mark.asyncio
async def test_capabilities_con_blender_viejo_no_desbloquea_nada(tmp_path: Path, monkeypatch):
    viejo = BlenderBuild(path=Path("blender.exe"), version=(3, 6, 0))
    monkeypatch.setattr(routes, "blender_probe", lambda *_a, **_k: viejo)

    respuesta = await model3d_capabilities(settings_dep=make_settings(tmp_path))

    assert respuesta.blender.found is True
    assert respuesta.blender.meets_minimum is False
    assert respuesta.unlocked == []
    assert "3.6.0" in (respuesta.missing or "")


@pytest.mark.asyncio
async def test_capabilities_con_blender_usable_desbloquea_el_carril(tmp_path: Path, monkeypatch):
    usable = BlenderBuild(path=Path("blender.exe"), version=(5, 2, 1))
    monkeypatch.setattr(routes, "blender_probe", lambda *_a, **_k: usable)

    respuesta = await model3d_capabilities(settings_dep=make_settings(tmp_path))

    assert respuesta.unlocked == ["audit", "referenceScene", "fit"]
    assert respuesta.missing is None


@pytest.mark.asyncio
async def test_capabilities_lista_los_motores_y_dice_que_le_falta_a_cada_uno(
    tmp_path: Path, monkeypatch
):
    """Un motor ausente NO es un error: es la respuesta.

    Y "no disponible" a secas manda a adivinar entre bajar varios GB de pesos
    y crear un entorno, asi que cada motor dice cual de los dos le falta.
    """
    usable = BlenderBuild(path=Path("blender.exe"), version=(5, 2, 1))
    monkeypatch.setattr(routes, "blender_probe", lambda *_a, **_k: usable)
    settings = make_settings(tmp_path)
    monkeypatch.setattr(type(settings), "mesh_engines_dir", property(lambda _s: tmp_path / "nada"))

    respuesta = await model3d_capabilities(settings_dep=settings)

    # Se afirma sobre el REGISTRO y no sobre una lista escrita a mano: esa
    # lista caduca en cuanto se agrega un motor, que es lo que pasó.
    from app.services.mesh_engine_service import ENGINES

    assert {motor.name for motor in respuesta.engines} == set(ENGINES)
    for motor in respuesta.engines:
        assert motor.ready is False, motor.name
        assert motor.license == ENGINES[motor.name]["license"], motor.name
        assert motor.missing and "entorno" in motor.missing, motor.name


@pytest.mark.asyncio
async def test_la_escena_no_le_cree_la_caja_de_tinta_al_cliente(tmp_path: Path, monkeypatch):
    # La escala sale de medir los recortes en disco. Si viniera del request,
    # cualquiera podria fijar una escala arbitraria en la escena.
    settings = make_settings(tmp_path)
    token = "a" * 32
    write_views(settings, token, ("front", "side"))
    recibido: dict = {}

    def fake_run(_settings, script, payload, **_kwargs):
        recibido["script"] = script
        recibido["payload"] = payload
        return {
            "blend": "x.blend",
            "heightMeters": payload["heightMeters"],
            "placed": [
                {
                    "view": nombre,
                    "image": datos["image"],
                    "inkHeightMeters": 1.7,
                    "planeHeightMeters": 1.7,
                    "planeWidthMeters": 0.5,
                    "scaledByInk": True,
                }
                for nombre, datos in payload["views"].items()
            ],
        }

    monkeypatch.setattr(model3d_service.blender_service, "run_script", fake_run)

    respuesta = await build_model3d_reference_scene(
        payload=ReferenceSceneRequest(token=token, heightMeters=1.7),
        request=RequestConDuenoLocal(token),
        settings_dep=settings,
        current_user=UsuarioLocal(),
    )

    assert recibido["script"] == "build_reference_scene.py"
    assert set(recibido["payload"]["views"]) == {"front", "side"}
    # Medida sobre el recorte, no heredada de la hoja ni recibida del cliente.
    assert recibido["payload"]["views"]["front"]["inkBox"] == [10, 5, 50, 195]
    assert respuesta.download_url.endswith(token)
    # La ruta absoluta del servidor no sale al cliente.
    assert all("/" not in colocada.image and "\\" not in colocada.image for colocada in respuesta.placed)


@pytest.mark.asyncio
async def test_un_token_mal_formado_es_404(tmp_path: Path):
    with pytest.raises(HTTPException) as exc_info:
        await build_model3d_reference_scene(
            payload=ReferenceSceneRequest(token="../../etc/passwd", heightMeters=1.7),
            request=RequestConDuenoLocal(),
            settings_dep=make_settings(tmp_path),
            current_user=UsuarioLocal(),
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_sin_recortes_en_disco_es_404(tmp_path: Path):
    with pytest.raises(HTTPException) as exc_info:
        await build_model3d_reference_scene(
            payload=ReferenceSceneRequest(token="b" * 32, heightMeters=1.7),
            request=RequestConDuenoLocal(),
            settings_dep=make_settings(tmp_path),
            current_user=UsuarioLocal(),
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_una_altura_no_positiva_es_400(tmp_path: Path):
    settings = make_settings(tmp_path)
    token = "c" * 32
    write_views(settings, token, ("front",))

    with pytest.raises(HTTPException) as exc_info:
        await build_model3d_reference_scene(
            payload=ReferenceSceneRequest(token=token, heightMeters=0.0),
            request=RequestConDuenoLocal(token),
            settings_dep=settings,
            current_user=UsuarioLocal(),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_bajar_una_escena_inexistente_es_404(tmp_path: Path):
    with pytest.raises(HTTPException) as exc_info:
        await download_reference_scene(
            token="d" * 32,
            request=RequestConDuenoLocal(),
            settings_dep=make_settings(tmp_path),
            current_user=UsuarioLocal(),
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_una_vista_con_nombre_inventado_es_404(tmp_path: Path):
    # El nombre no puede venir del cliente: cualquier cosa fuera de VIEW_ORDER
    # seria una ruta arbitraria adentro de la carpeta de salidas.
    settings = make_settings(tmp_path)
    token = "e" * 32
    write_views(settings, token, ("front",))

    with pytest.raises(HTTPException) as exc_info:
        await download_sheet_view(
            token=token,
            name="../../etc/passwd",
            request=RequestConDuenoLocal(token),
            settings_dep=settings,
            current_user=UsuarioLocal(),
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_una_vista_existente_se_sirve_como_png(tmp_path: Path):
    settings = make_settings(tmp_path)
    token = "f" * 32
    write_views(settings, token, ("front",))

    respuesta = await download_sheet_view(
        token=token,
        name="front",
        request=RequestConDuenoLocal(token),
        settings_dep=settings,
        current_user=UsuarioLocal(),
    )

    assert respuesta.media_type == "image/png"
    assert Path(respuesta.path).exists()


@pytest.mark.asyncio
async def test_una_vista_que_no_se_partio_es_404(tmp_path: Path):
    settings = make_settings(tmp_path)
    token = "0" * 32
    write_views(settings, token, ("front",))

    with pytest.raises(HTTPException) as exc_info:
        await download_sheet_view(
            token=token,
            name="back",
            request=RequestConDuenoLocal(token),
            settings_dep=settings,
            current_user=UsuarioLocal(),
        )

    assert exc_info.value.status_code == 404
