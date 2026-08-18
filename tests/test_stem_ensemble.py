"""Combinar varios separadores en uno: por que sirve y donde no.

Promediar baja el RUIDO DE ESTIMACION, que es distinto en cada arquitectura, no
el error comun: si todos se dejan la voz adentro del instrumental, el promedio
tambien la tiene. Estas pruebas fijan las dos mitades de esa frase, mas la regla
que hace que combinar signifique algo (los mismos stems en todos).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.services.audio_job_manager import MAX_ENSEMBLE_MODELS, AudioJobManager
from app.services.engines.separation_models import SEPARATION_MODELS
from app.services.engines.stem_ensemble import align_lengths, average_stem_files

SR = 44100


def escribir(path: Path, audio: np.ndarray) -> None:
    # FLOAT como los separadores: con el PCM_16 por default de soundfile, la
    # entrada del test ya llegaria cuantizada y taparia lo que se mide.
    sf.write(path, audio.astype(np.float32), SR, subtype="FLOAT")


def test_promediar_cancela_lo_que_no_comparten(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    senal = rng.standard_normal(SR).astype(np.float32) * 0.2
    # Mismo "modelo correcto" con artefactos independientes: es exactamente la
    # situacion que hace util el ensemble.
    entradas = []
    for indice in range(3):
        ruido = np.random.default_rng(100 + indice).standard_normal(SR) * 0.1
        ruta = tmp_path / f"m{indice}.wav"
        escribir(ruta, senal + ruido.astype(np.float32))
        entradas.append(ruta)

    average_stem_files(entradas, tmp_path / "mezcla.wav")

    salida, _ = sf.read(tmp_path / "mezcla.wav", dtype="float32", always_2d=True)
    error_promedio = float(np.abs(salida[:, 0] - senal).mean())
    error_de_uno, _ = sf.read(entradas[0], dtype="float32", always_2d=True)
    assert error_promedio < float(np.abs(error_de_uno[:, 0] - senal).mean())


def test_lo_que_todos_erran_igual_sigue_estando(tmp_path: Path) -> None:
    comun = np.full(SR, 0.3, dtype=np.float32)
    entradas = []
    for indice in range(3):
        ruta = tmp_path / f"m{indice}.wav"
        escribir(ruta, comun)
        entradas.append(ruta)

    average_stem_files(entradas, tmp_path / "mezcla.wav")

    salida, _ = sf.read(tmp_path / "mezcla.wav", dtype="float32", always_2d=True)
    # Promediar no inventa separacion que ninguno logro.
    assert np.allclose(salida[:, 0], comun, atol=1e-6)


def test_unas_muestras_de_diferencia_no_rompen_la_suma() -> None:
    # Cada modelo redondea el final a su tamaño de bloque; unas pocas muestras
    # de diferencia bastan para que la suma reviente.
    alineadas = align_lengths([np.zeros((100, 2)), np.zeros((97, 2)), np.zeros((99, 2))])

    assert {pista.shape[0] for pista in alineadas} == {97}


def test_se_recorta_y_no_se_rellena() -> None:
    largo = np.ones((10, 1))
    corto = np.ones((6, 1))

    alineadas = align_lengths([largo, corto])

    # Rellenar con ceros meteria un silencio que ningun modelo estimo.
    assert all(pista.shape[0] == 6 for pista in alineadas)
    assert float(alineadas[0].sum()) == 6.0


def test_combinar_nada_es_un_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        average_stem_files([], tmp_path / "x.wav")


# ---------------------------------------------------------------------------
# Que combinaciones tienen sentido
# ---------------------------------------------------------------------------


@pytest.fixture
def instalar(monkeypatch):
    """Finge que ciertos modelos estan bajados, sin escribir .onnx de mentira."""

    def _instalar(tmp_path: Path, instalados: list[str]) -> AudioJobManager:
        monkeypatch.setattr(
            Settings, "karaoke_installed_models", lambda self: list(instalados)
        )
        settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
        return AudioJobManager(settings, pipeline=None, device_semaphores=None)

    return _instalar


def dos_de_karaoke() -> list[str]:
    karaoke = [
        spec.id
        for spec in SEPARATION_MODELS.values()
        if spec.stem_ids() == ("instrumental", "vocals")
    ]
    assert len(karaoke) >= 2, "sin dos modelos de las mismas pistas no se prueba nada"
    return karaoke[:2]


def test_sin_extras_no_hay_ensemble(tmp_path: Path, instalar) -> None:
    principal = dos_de_karaoke()[0]

    assert instalar(tmp_path, [principal])._validate_ensemble(principal, []) == []


def test_repetir_el_principal_no_lo_convierte_en_ensemble(tmp_path: Path, instalar) -> None:
    principal = dos_de_karaoke()[0]

    # Correr dos veces el mismo modelo da el mismo resultado y cobra el doble.
    assert instalar(tmp_path, [principal])._validate_ensemble(principal, [principal]) == []


def test_dos_modelos_de_las_mismas_pistas_se_combinan(tmp_path: Path, instalar) -> None:
    primero, segundo = dos_de_karaoke()

    elegidos = instalar(tmp_path, [primero, segundo])._validate_ensemble(primero, [segundo])

    assert elegidos == [segundo]


def test_modelos_de_pistas_distintas_no_se_combinan(tmp_path: Path, instalar) -> None:
    karaoke = dos_de_karaoke()[0]
    otro = next(
        spec.id
        for spec in SEPARATION_MODELS.values()
        if spec.stem_ids() != ("instrumental", "vocals")
    )
    gestor = instalar(tmp_path, [karaoke, otro])

    with pytest.raises(ValueError) as error:
        gestor._validate_ensemble(karaoke, [otro])

    # Promediar un instrumental con un bajo no da un instrumental mejor: da una
    # suma que ademas nadie puede etiquetar.
    assert otro in str(error.value)


def test_un_modelo_sin_instalar_no_se_puede_combinar(tmp_path: Path, instalar) -> None:
    primero, segundo = dos_de_karaoke()
    gestor = instalar(tmp_path, [primero])

    with pytest.raises(ValueError):
        gestor._validate_ensemble(primero, [segundo])


def test_hay_un_techo_de_modelos(tmp_path: Path, instalar) -> None:
    todos = [spec.id for spec in SEPARATION_MODELS.values()]
    if len(todos) <= MAX_ENSEMBLE_MODELS:
        pytest.skip("el catalogo no tiene suficientes modelos para pasarse del techo")
    gestor = instalar(tmp_path, todos)

    with pytest.raises(ValueError):
        gestor._validate_ensemble(todos[0], todos[1 : MAX_ENSEMBLE_MODELS + 1])


def test_pedir_ensemble_sin_separar_es_un_error(tmp_path: Path, instalar) -> None:
    gestor = instalar(tmp_path, dos_de_karaoke())

    with pytest.raises(ValueError) as error:
        asyncio.run(
            gestor.create_job(
                source_path=tmp_path / "a.wav",
                original_filename="a.wav",
                ensemble_models=["inst_hq_3"],
            )
        )

    assert "separate=true" in str(error.value)
