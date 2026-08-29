from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.config import Settings
from app.services import mesh_engine_service
from app.services.blender_service import RESULT_SENTINEL
from app.services.mesh_engine_service import (
    ENGINES,
    EngineBuild,
    MeshEngineError,
    available,
    build_for,
    generate,
    script_path,
)


@pytest.fixture
def configuracion_motores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Settings, Path]:
    raiz = tmp_path / "motores"
    monkeypatch.setattr(
        Settings,
        "mesh_engines_dir",
        property(lambda _settings: raiz),
    )
    return Settings(_env_file=None), raiz


@pytest.fixture
def motor_listo(
    configuracion_motores: tuple[Settings, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Settings, EngineBuild]:
    settings, _ = configuracion_motores
    build = build_for(settings, "triposg")
    build.python.parent.mkdir(parents=True)
    build.python.write_bytes(b"")
    build.source.mkdir()

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "triposg_generate.py").write_text(
        "# script falso para el test\n", encoding="utf-8"
    )
    monkeypatch.setattr(mesh_engine_service, "ENGINE_SCRIPTS_DIR", scripts_dir)
    return settings, build


def test_build_for_calcula_rutas_sin_exigir_que_existan(
    configuracion_motores: tuple[Settings, Path],
) -> None:
    settings, raiz = configuracion_motores
    carpeta, ejecutable = mesh_engine_service._interpreter_name()
    assert not raiz.exists()

    build = build_for(settings, "triposg")

    assert build.python == raiz / "triposg-env" / carpeta / ejecutable
    assert build.source == raiz / "TripoSG"
    assert not build.python.exists()
    assert not build.source.exists()


def test_build_for_rechaza_un_motor_desconocido(
    configuracion_motores: tuple[Settings, Path],
) -> None:
    settings, _ = configuracion_motores

    with pytest.raises(MeshEngineError, match="motor desconocido: inventado"):
        build_for(settings, "inventado")


def test_missing_indica_que_falta_el_entorno_si_no_existe_python(
    tmp_path: Path,
) -> None:
    source = tmp_path / "TripoSG"
    source.mkdir()
    build = EngineBuild(
        name="triposg",
        python=tmp_path / "triposg-env" / "Scripts" / "python.exe",
        source=source,
        license="MIT",
    )

    missing = build.missing

    assert missing is not None
    assert "falta el entorno" in missing
    assert build.ready is False


def test_missing_indica_que_falta_el_codigo_si_solo_existe_python(
    tmp_path: Path,
) -> None:
    python = tmp_path / "triposg-env" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    build = EngineBuild(
        name="triposg",
        python=python,
        source=tmp_path / "TripoSG",
        license="MIT",
    )

    missing = build.missing

    assert missing is not None
    assert "falta el codigo" in missing
    assert build.ready is False


def test_missing_es_none_y_ready_es_true_si_entorno_y_codigo_existen(
    tmp_path: Path,
) -> None:
    python = tmp_path / "triposg-env" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    source = tmp_path / "TripoSG"
    source.mkdir()
    build = EngineBuild(
        name="triposg",
        python=python,
        source=source,
        license="MIT",
    )

    missing = build.missing
    ready = build.ready

    assert missing is None
    assert ready is True


def test_available_describe_todos_los_motores_aunque_no_haya_nada_instalado(
    configuracion_motores: tuple[Settings, Path],
) -> None:
    settings, raiz = configuracion_motores
    assert not raiz.exists()

    estado = available(settings)

    assert set(estado) == set(ENGINES)
    for nombre, ficha in ENGINES.items():
        assert set(estado[nombre]) == {"ready", "license", "device", "missing"}
        assert estado[nombre]["ready"] is False
        assert estado[nombre]["license"] == ficha["license"]
        assert estado[nombre]["device"] == ficha["device"]
        assert estado[nombre]["missing"] is not None


def test_script_path_rechaza_una_fuga_de_carpeta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    monkeypatch.setattr(mesh_engine_service, "ENGINE_SCRIPTS_DIR", scripts_dir)

    with pytest.raises(MeshEngineError, match="fuera de la carpeta"):
        script_path("../escape.py")


def test_script_path_rechaza_un_archivo_que_no_termina_en_py(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    monkeypatch.setattr(mesh_engine_service, "ENGINE_SCRIPTS_DIR", scripts_dir)

    with pytest.raises(MeshEngineError, match="fuera de la carpeta"):
        script_path("generar.txt")


def test_generate_rechaza_un_motor_no_listo_sin_llamar_a_subprocess(
    configuracion_motores: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, _ = configuracion_motores

    def proceso_inesperado(*args, **kwargs):
        raise AssertionError("subprocess.run no debe llamarse")

    monkeypatch.setattr(mesh_engine_service.subprocess, "run", proceso_inesperado)

    with pytest.raises(MeshEngineError, match="falta el entorno"):
        generate(settings, "triposg", {})


def test_generate_usa_el_ultimo_centinela_y_agrega_motor_y_licencia(
    motor_listo: tuple[Settings, EngineBuild], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, _ = motor_listo
    stdout = (
        "ruido del motor\n"
        f'{RESULT_SENTINEL}{json.dumps({"mesh": "provisional.glb"})}\n'
        "mas ruido\n"
        f'{RESULT_SENTINEL}{json.dumps({"mesh": "final.glb", "faces": 1200})}\n'
    )
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=stdout, stderr=""
    )
    monkeypatch.setattr(
        mesh_engine_service.subprocess, "run", lambda *args, **kwargs: completed
    )

    resultado = generate(settings, "triposg", {"image": "entrada.png"})

    assert resultado == {
        "mesh": "final.glb",
        "faces": 1200,
        "engine": "triposg",
        "license": "MIT",
    }


def test_generate_falla_si_el_subproceso_no_emite_un_centinela(
    motor_listo: tuple[Settings, EngineBuild], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, _ = motor_listo
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="terminado sin reporte\n", stderr=""
    )
    monkeypatch.setattr(
        mesh_engine_service.subprocess, "run", lambda *args, **kwargs: completed
    )

    with pytest.raises(MeshEngineError, match="sin reportar resultado"):
        generate(settings, "triposg", {})


def test_generate_falla_con_el_error_reportado_en_el_json(
    motor_listo: tuple[Settings, EngineBuild], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, _ = motor_listo
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=f'{RESULT_SENTINEL}{json.dumps({"error": "no pudo crear la malla"})}\n',
        stderr="detalle del motor\n",
    )
    monkeypatch.setattr(
        mesh_engine_service.subprocess, "run", lambda *args, **kwargs: completed
    )

    with pytest.raises(MeshEngineError, match="no pudo crear la malla"):
        generate(settings, "triposg", {})


def test_generate_convierte_timeout_expired_en_mesh_engine_error(
    motor_listo: tuple[Settings, EngineBuild], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, _ = motor_listo

    def proceso_agotado(command: list[str], **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(mesh_engine_service.subprocess, "run", proceso_agotado)

    with pytest.raises(MeshEngineError, match="paso los 12 s") as exc_info:
        generate(settings, "triposg", {}, timeout=12)

    assert isinstance(exc_info.value.__cause__, subprocess.TimeoutExpired)


def _sin_motores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    from app.config import Settings as _S

    settings = _S(RUNTIME_DIR=str(tmp_path), _env_file=None)
    monkeypatch.setattr(
        type(settings), "mesh_engines_dir", property(lambda _s: tmp_path / "sin-motores")
    )
    return settings


def test_lo_que_falta_no_lleva_rutas_absolutas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La ruta de la máquina va al log del servidor, nunca al cliente.

    Es la misma regla que ya rige en blender_service. `/model3d/capabilities` ni
    siquiera pide autenticación, así que un absoluto ahí filtra el disco a
    cualquiera. Y nombrar la variable de entorno es MÁS accionable que pegar un
    absoluto: dice dónde cambiarlo, no sólo dónde miró el proceso.
    """
    settings = _sin_motores(tmp_path, monkeypatch)

    faltante = build_for(settings, "triposg").missing

    assert faltante
    assert str(tmp_path) not in faltante
    assert "MESH_ENGINES_ROOT" in faltante


def test_el_error_de_motor_no_listo_tampoco_lleva_rutas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _sin_motores(tmp_path, monkeypatch)

    with pytest.raises(MeshEngineError) as capturado:
        generate(settings, "triposg", {"image": "x", "output": "y"})

    assert str(tmp_path) not in str(capturado.value)


def test_todos_los_motores_reportan_sin_rutas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vale para cada motor del registro, no sólo para el primero."""
    settings = _sin_motores(tmp_path, monkeypatch)

    for nombre, estado in available(settings).items():
        assert estado["missing"], nombre
        assert str(tmp_path) not in estado["missing"], nombre
