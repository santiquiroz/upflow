from __future__ import annotations

import pytest

from tests.test_generation_converter import make_converter

# ---------------------------------------------------------------------------
# Reportado el 2026-08-06: "al intentar instalar y convertir un modelo y salir a
# otra ventana se pierde el progreso".
#
# La conversion NO se pierde — sigue corriendo en el servidor. Lo que se pierde
# es el hilo para volver a encontrarla: el id vivia solo en la pantalla, y al
# cambiar de seccion se iba con ella. Sin forma de listar las conversiones en
# curso, la barra no puede volver a engancharse y el usuario ve que "se perdio".
#
# Convertir un SDXL tarda cerca de media hora. Creer que se perdio y arrancarlo
# de nuevo es media hora tirada, ademas de dos conversiones compitiendo por la
# misma maquina.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_se_pueden_listar_las_conversiones_en_curso(tmp_path):
    convertidor, _installer, _settings, _registry = make_converter(tmp_path, lambda *a, **k: ["unet"])
    conversion_id = await convertidor.convert_from_hf("John6666/epicrealism-xl-vxv")

    activas = convertidor.active()

    assert [job.id for job in activas] == [conversion_id]


@pytest.mark.asyncio
async def test_la_conversion_lista_dice_de_que_repo_es(tmp_path):
    # Sin el repo, la pantalla no sabe a que tarjeta del buscador re-enganchar la
    # barra, y el usuario sigue sin ver su progreso.
    convertidor, _installer, _settings, _registry = make_converter(tmp_path, lambda *a, **k: ["unet"])
    await convertidor.convert_from_hf("John6666/epicrealism-xl-vxv")

    assert convertidor.active()[0].repo_id == "John6666/epicrealism-xl-vxv"


@pytest.mark.asyncio
async def test_las_terminadas_no_aparecen_como_en_curso(tmp_path):
    from app.models import JobStatus

    convertidor, _installer, _settings, _registry = make_converter(tmp_path, lambda *a, **k: ["unet"])
    conversion_id = await convertidor.convert_from_hf("algo/modelo")
    convertidor.status(conversion_id).status = JobStatus.completed

    assert convertidor.active() == []
