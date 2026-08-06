from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.exceptions import QueueFullError
from app.models import JobStatus
from app.services.engines.shape3d import Shape3dUnavailable
from app.services.missing_pack import PACK_LABELS
from app.services.shape3d_job_manager import Shape3dJobManager
from app.services.storage import StorageService

# ---------------------------------------------------------------------------
# Dos minutos de CPU no entran en una peticion sincronica: el navegador cortaria
# antes y el usuario no tendria a que volver. De ahi la cola.
#
# Lo que hace distinto a este manager: la malla NO se entrega cruda. Pasa por el
# banco y el veredicto viaja con el resultado.
# ---------------------------------------------------------------------------


def tetrahedron(scale: float = 10.0) -> np.ndarray:
    a, b, c, d = (0.0, 0.0, 0.0), (scale, 0.0, 0.0), (0.0, scale, 0.0), (0.0, 0.0, scale)
    return np.array([[a, c, b], [a, b, d], [a, d, c], [b, c, d]], dtype=np.float64)


class FakeEngine:
    def __init__(self, malla=None, error: Exception | None = None) -> None:
        self.malla = tetrahedron() if malla is None else malla
        self.error = error
        self.prompts: list[str] = []

    def available(self) -> bool:
        return True

    def generate_from_text(self, prompt: str, **_kwargs) -> np.ndarray:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.malla


class MissingEngine:
    def available(self) -> bool:
        return False


def make_manager(tmp_path: Path, engine=None) -> tuple[Shape3dJobManager, Settings]:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return Shape3dJobManager(settings, engine or FakeEngine()), settings


@pytest.mark.asyncio
async def test_a_generated_mesh_comes_back_with_its_verdict(tmp_path: Path):
    manager, _settings = make_manager(tmp_path)
    job = await manager.create_job(prompt="un soporte")

    await manager._process_next()

    assert job.status is JobStatus.completed
    assert job.can_print is True
    assert job.triangle_count == 4
    assert job.output_path.exists()


@pytest.mark.asyncio
async def test_a_mesh_that_does_not_close_is_reported_not_printable(tmp_path: Path):
    # El banco corre SIEMPRE, tambien sobre lo generado. Entregar una malla
    # abierta sin decirlo seria mandar a imprimir con confianza prestada.
    abierta = tetrahedron()[:-1]
    manager, _settings = make_manager(tmp_path, FakeEngine(malla=abierta))
    job = await manager.create_job(prompt="algo roto")

    await manager._process_next()

    assert job.status is JobStatus.completed
    assert job.can_print is False
    assert job.blockers


@pytest.mark.asyncio
async def test_the_target_size_reaches_the_mesh(tmp_path: Path):
    # Una malla generada no trae cotas: sin esto sale en la escala que se le
    # ocurrio al modelo.
    manager, _settings = make_manager(tmp_path)
    job = await manager.create_job(prompt="un soporte", target_mm=120.0)

    await manager._process_next()

    assert job.size_mm[0] == pytest.approx(120.0, abs=0.1)


@pytest.mark.asyncio
async def test_a_part_bigger_than_the_bed_is_reported(tmp_path: Path):
    manager, _settings = make_manager(tmp_path)
    job = await manager.create_job(prompt="algo enorme", target_mm=900.0)

    await manager._process_next()

    assert job.can_print is False
    assert job.blockers


@pytest.mark.asyncio
async def test_the_prompt_reaches_the_engine_trimmed(tmp_path: Path):
    engine = FakeEngine()
    manager, _settings = make_manager(tmp_path, engine)
    await manager.create_job(prompt="   un soporte en L   ")

    await manager._process_next()

    assert engine.prompts == ["un soporte en L"]


@pytest.mark.asyncio
async def test_an_engine_failure_fails_the_job_instead_of_the_app(tmp_path: Path):
    manager, _settings = make_manager(
        tmp_path, FakeEngine(error=Shape3dUnavailable("la malla salio toda NaN"))
    )
    job = await manager.create_job(prompt="algo")

    await manager._process_next()

    assert job.status is JobStatus.failed
    assert "NaN" in job.error


class TestValidation:
    @pytest.mark.asyncio
    async def test_an_empty_prompt_is_refused(self, tmp_path: Path):
        manager, _settings = make_manager(tmp_path)
        with pytest.raises(ValueError, match="descripcion"):
            await manager.create_job(prompt="   ")

    @pytest.mark.asyncio
    async def test_an_absurdly_long_prompt_is_refused(self, tmp_path: Path):
        manager, _settings = make_manager(tmp_path)
        with pytest.raises(ValueError, match="caracteres"):
            await manager.create_job(prompt="x" * 500)

    @pytest.mark.asyncio
    async def test_an_unknown_printer_lists_the_known_ones(self, tmp_path: Path):
        manager, _settings = make_manager(tmp_path)
        with pytest.raises(ValueError, match="ender-3"):
            await manager.create_job(prompt="algo", printer="no-existe")

    @pytest.mark.asyncio
    async def test_a_negative_size_is_refused(self, tmp_path: Path):
        manager, _settings = make_manager(tmp_path)
        with pytest.raises(ValueError):
            await manager.create_job(prompt="algo", target_mm=-5.0)

    @pytest.mark.asyncio
    async def test_dice_que_paquete_falta_sin_dictar_comandos(self, tmp_path: Path):
        # El usuario tiene que saber QUE falta; como se baja es asunto de la app,
        # no un comando que le toque copiar a una terminal.
        manager, _settings = make_manager(tmp_path, MissingEngine())
        with pytest.raises(Shape3dUnavailable) as excinfo:
            await manager.create_job(prompt="algo")

        mensaje = str(excinfo.value)
        assert PACK_LABELS["shap-e"] in mensaje
        assert ".ps1" not in mensaje

    @pytest.mark.asyncio
    async def test_a_full_queue_is_refused_not_dropped(self, tmp_path: Path):
        settings = Settings(RUNTIME_DIR=str(tmp_path), MAX_QUEUE_SIZE=1, _env_file=None)
        StorageService(settings).ensure_directories()
        manager = Shape3dJobManager(settings, FakeEngine())
        await manager.create_job(prompt="uno")

        with pytest.raises(QueueFullError):
            await manager.create_job(prompt="dos")


@pytest.mark.asyncio
async def test_cancelling_a_queued_job_stops_it_before_it_runs(tmp_path: Path):
    engine = FakeEngine()
    manager, _settings = make_manager(tmp_path, engine)
    job = await manager.create_job(prompt="algo")

    assert manager.cancel_job(job.id)
    await manager._process_next()

    assert job.status is JobStatus.cancelled
    assert engine.prompts == []


# ---------------------------------------------------------------------------
# El carril CAD: la descripcion pasa por un modelo que escribe OpenSCAD, y eso da
# COTAS. Es lo unico que un generador de malla no puede dar.
# ---------------------------------------------------------------------------


class ClienteCadFalso:
    def __init__(self, respuestas=None) -> None:
        self.respuestas = list(respuestas or ["cube([10,10,10]);"])
        self.pedidos: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.pedidos.append(user)
        return self.respuestas.pop(0) if self.respuestas else "cube([10,10,10]);"


def make_cad_manager(tmp_path: Path, cliente=None, con_openscad: bool = True):
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    if con_openscad:
        falso = tmp_path / "vendor" / "openscad" / "openscad.exe"
        falso.parent.mkdir(parents=True, exist_ok=True)
        falso.write_bytes(b"no-es-el-binario-real")
    return Shape3dJobManager(
        settings, FakeEngine(), cad_client=cliente or ClienteCadFalso()
    ), settings


@pytest.mark.asyncio
async def test_the_cad_lane_needs_a_model_server(tmp_path: Path):
    # Decir "no hay servidor" es mejor que fingir que si y fallar en el medio.
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    manager = Shape3dJobManager(settings, FakeEngine(), cad_client=None)

    with pytest.raises(Shape3dUnavailable, match="servidor"):
        await manager.create_job(prompt="un taco", source="cad")


@pytest.mark.asyncio
async def test_the_cad_lane_needs_openscad(tmp_path: Path):
    manager, settings = make_cad_manager(tmp_path, con_openscad=False)
    # El binario del sistema puede existir en esta maquina: se apunta a uno que no.
    object.__setattr__(settings, "runtime_dir", str(tmp_path / "sin-nada"))

    if not Path(settings.openscad_binary_path).exists():
        with pytest.raises(Shape3dUnavailable, match="OpenSCAD"):
            await manager.create_job(prompt="un taco", source="cad")


@pytest.mark.asyncio
async def test_an_unknown_source_is_refused(tmp_path: Path):
    manager, _settings = make_cad_manager(tmp_path)

    with pytest.raises(ValueError, match="Origen"):
        await manager.create_job(prompt="algo", source="magia")


@pytest.mark.asyncio
async def test_the_mesh_lane_does_not_need_the_cad_pieces(tmp_path: Path):
    # Los dos carriles son independientes: que falte uno no puede romper el otro.
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    manager = Shape3dJobManager(settings, FakeEngine(), cad_client=None)

    job = await manager.create_job(prompt="algo", source="mesh")
    await manager._process_next()

    assert job.status is JobStatus.completed
