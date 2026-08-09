from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.auth_deps import off_mode_user
from app.api.routes import download_generated_part, generate_part, list_part_kinds
from app.config import Settings
from app.schemas import GeneratePartRequest
from app.services.storage import StorageService

# ---------------------------------------------------------------------------
# El carril de cotas exactas tiene que ser pedible desde afuera, y lo que
# devuelve tiene que venir YA verificado por el mismo banco que verifica un STL
# ajeno. Generar y verificar con la misma herramienta no probaria nada; generar
# con una y verificar con la otra, si.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


def fake_request() -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()), state=SimpleNamespace())


async def _generate(settings: Settings, payload: GeneratePartRequest, request=None):
    return await generate_part(
        payload=payload,
        request=request or fake_request(),
        settings_dep=settings,
        current_user=off_mode_user(),
    )


@pytest.mark.asyncio
async def test_a_spacer_comes_back_printable_and_measured(tmp_path: Path):
    settings = make_settings(tmp_path)

    respuesta = await _generate(
        settings,
        GeneratePartRequest(
            kind="tube",
            params={"outer_diameter": 20.0, "inner_diameter": 8.4, "height": 12.0},
            printer="ender-3",
        ),
    )

    assert respuesta.can_print
    assert respuesta.size_mm[2] == pytest.approx(12.0)
    assert respuesta.download_url


@pytest.mark.asyncio
async def test_the_generated_file_is_really_the_part(tmp_path: Path):
    settings = make_settings(tmp_path)
    req = fake_request()
    respuesta = await _generate(
        settings,
        GeneratePartRequest(
            kind="box", params={"x": 30.0, "y": 20.0, "z": 10.0}, printer="ender-3"
        ),
        request=req,
    )
    token = respuesta.download_url.rsplit("/", 1)[-1]

    archivo = await download_generated_part(
        token=token, request=req, settings_dep=settings, current_user=off_mode_user()
    )

    from app.services.mesh_inspect import inspect_mesh
    from app.services.stl_reader import read_stl

    reporte = inspect_mesh(read_stl(Path(archivo.path)))
    assert reporte.size == pytest.approx((30.0, 20.0, 10.0), abs=0.01)
    assert reporte.printable


@pytest.mark.asyncio
async def test_an_impossible_wall_is_a_400_with_the_reason(tmp_path: Path):
    settings = make_settings(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await _generate(
            settings,
            GeneratePartRequest(
                kind="tube",
                params={"outer_diameter": 20.0, "inner_diameter": 19.6, "height": 10.0},
                printer="ender-3",
            ),
        )

    assert exc_info.value.status_code == 400
    assert "pared" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_an_unknown_kind_lists_the_known_ones(tmp_path: Path):
    settings = make_settings(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await _generate(
            settings,
            GeneratePartRequest(kind="nave-espacial", params={}, printer="ender-3"),
        )

    assert exc_info.value.status_code == 400
    assert "tube" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_a_missing_parameter_is_a_400_not_a_500(tmp_path: Path):
    settings = make_settings(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await _generate(
            settings,
            GeneratePartRequest(kind="tube", params={"height": 10.0}, printer="ender-3"),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_a_part_too_big_for_the_bed_says_so_but_still_delivers(tmp_path: Path):
    # La pieza esta bien hecha: lo que no entra es en ESA maquina. Negarle el
    # archivo seria decidir por el usuario, que capaz tiene otra impresora.
    settings = make_settings(tmp_path)

    respuesta = await _generate(
        settings,
        GeneratePartRequest(
            kind="box", params={"x": 500.0, "y": 500.0, "z": 10.0}, printer="ender-3"
        ),
    )

    assert not respuesta.can_print
    assert respuesta.blockers
    assert respuesta.download_url


@pytest.mark.asyncio
async def test_the_catalog_describes_each_part(tmp_path: Path):
    respuesta = await list_part_kinds()

    por_id = {p.id: p for p in respuesta.kinds}
    assert "tube" in por_id
    assert por_id["tube"].params
    assert all(p.label_key for p in respuesta.kinds)
