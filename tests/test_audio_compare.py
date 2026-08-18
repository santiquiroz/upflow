"""Probar dos o tres separadores sobre el archivo del usuario, no sobre un dataset.

Que modelo anda mejor depende del material. Lo que estas pruebas fijan es que la
comparacion sea COMPARABLE: el mismo fragmento para todos, cortado una sola vez,
y del medio del tema y no del arranque.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.services.audio_compare import (
    MAX_MODELS_PER_COMPARISON,
    start_comparison,
    validate_models,
)
from app.services.audio_excerpt import (
    build_excerpt_command,
    centered_offset,
    parse_duration_seconds,
)

INSTALADOS = {"inst_hq_3", "voc_ft", "umx_4stem"}


def test_comparar_contra_si_mismo_no_es_comparar() -> None:
    with pytest.raises(ValueError):
        validate_models(["inst_hq_3"], INSTALADOS)
    # Repetido tampoco: dos corridas del mismo modelo dan el mismo resultado y
    # cobran el doble.
    with pytest.raises(ValueError):
        validate_models(["inst_hq_3", "inst_hq_3"], INSTALADOS)


def test_mas_de_tres_deja_de_ser_comparacion_y_es_cola() -> None:
    de_mas = [f"m{i}" for i in range(MAX_MODELS_PER_COMPARISON + 1)]
    with pytest.raises(ValueError):
        validate_models(de_mas, set(de_mas))


def test_un_modelo_sin_instalar_se_dice_por_su_nombre() -> None:
    with pytest.raises(ValueError) as error:
        validate_models(["inst_hq_3", "fantasma"], INSTALADOS)

    # "alguno no esta instalado" obliga a revisar los tres a mano.
    assert "fantasma" in str(error.value)
    assert "inst_hq_3" not in str(error.value)


def test_el_orden_pedido_se_respeta() -> None:
    assert validate_models(["voc_ft", "inst_hq_3"], INSTALADOS) == ["voc_ft", "inst_hq_3"]


def test_el_fragmento_sale_del_medio_y_no_del_arranque() -> None:
    # Las intros suelen ser instrumentales: juzgar un separador de VOCES en una
    # parte sin voces no dice nada.
    assert centered_offset(240.0, 30) == 105.0


def test_un_tema_mas_corto_que_el_fragmento_arranca_en_cero() -> None:
    assert centered_offset(20.0, 30) == 0.0


def test_sin_duracion_conocida_arranca_en_cero() -> None:
    # Peor lugar para juzgar, pero un offset mas alla del final da un archivo
    # vacio, que no es peor sino inservible.
    assert centered_offset(None, 30) == 0.0
    assert parse_duration_seconds({"format": {}}) is None
    assert parse_duration_seconds({"format": {"duration": "0"}}) is None


def test_el_corte_salta_al_punto_en_vez_de_decodificar_todo() -> None:
    comando = build_excerpt_command(
        ffmpeg=Path("ffmpeg"),
        source=Path("in.mp3"),
        destination=Path("out.wav"),
        offset_seconds=105.0,
        excerpt_seconds=30,
    )

    # `-ss` ANTES de `-i`: en un tema de 4 minutos es la diferencia entre cortar
    # al instante y cortar en segundos.
    assert comando.index("-ss") < comando.index("-i")
    assert comando[comando.index("-t") + 1] == "30"
    assert "-vn" in comando


class AudioJobsFalso:
    def __init__(self) -> None:
        self.pedidos: list[dict] = []

    async def create_job(self, **kwargs):
        self.pedidos.append(kwargs)
        return type("Job", (), {"id": kwargs["job_id"]})()


def settings_de_prueba(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


@pytest.fixture
def ffmpeg_falso(monkeypatch):
    """Corta de mentira: escribe el destino y reporta una duracion fija."""
    llamadas: list[list[str]] = []

    async def correr(command, _timeout, **_kwargs):
        llamadas.append(command)
        if "-show_entries" in command:
            return json.dumps({"format": {"duration": "240.0"}}).encode(), b"", 0
        Path(command[-1]).write_bytes(b"fragmento")
        return b"", b"", 0

    # Los dos: el corte lo lanza `audio_compare` y el ffprobe de duracion vive
    # en `audio_excerpt`, que es de donde tambien lo toma el karaoke.
    monkeypatch.setattr("app.services.audio_compare.run_guarded_process", correr)
    monkeypatch.setattr("app.services.audio_excerpt.run_guarded_process", correr)
    return llamadas


def comparar(tmp_path: Path, audio, **extra):
    fuente = tmp_path / "cancion.mp3"
    fuente.write_bytes(b"x")
    settings = settings_de_prueba(tmp_path)
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    return asyncio.run(
        start_comparison(
            fuente,
            "cancion.mp3",
            ["inst_hq_3", "voc_ft"],
            audio_jobs=audio,
            settings=settings,
            excerpt_seconds=30,
            **extra,
        )
    )


def test_el_fragmento_se_corta_una_sola_vez_para_todos(tmp_path: Path, ffmpeg_falso) -> None:
    audio = AudioJobsFalso()

    comparar(tmp_path, audio)

    cortes = [c for c in ffmpeg_falso if "-t" in c]
    # Si el corte se repitiera por modelo y llegara a variar, la comparacion
    # dejaria de ser entre modelos.
    assert len(cortes) == 1
    assert len(audio.pedidos) == 2


def test_cada_modelo_recibe_su_propio_archivo(tmp_path: Path, ffmpeg_falso) -> None:
    audio = AudioJobsFalso()

    comparar(tmp_path, audio)

    fuentes = [p["source_path"] for p in audio.pedidos]
    # Uno solo compartido no sirve: el primero que termina lo borra y el segundo
    # se queda sin entrada.
    assert len(set(fuentes)) == 2
    assert all(p.exists() for p in fuentes)
    assert [p["separation_model"] for p in audio.pedidos] == ["inst_hq_3", "voc_ft"]


def test_el_nombre_dice_que_modelo_lo_hizo(tmp_path: Path, ffmpeg_falso) -> None:
    audio = AudioJobsFalso()

    comparar(tmp_path, audio)

    # Dos entradas iguales en la cola no se pueden comparar: hay que saber cual
    # es cual sin abrir el detalle.
    assert "inst_hq_3" in audio.pedidos[0]["original_filename"]
    assert "voc_ft" in audio.pedidos[1]["original_filename"]


def test_el_offset_elegido_por_el_servidor_se_informa(tmp_path: Path, ffmpeg_falso) -> None:
    audio = AudioJobsFalso()

    resultado = comparar(tmp_path, audio)

    # Sin decirlo, el usuario no sabe que parte del tema esta oyendo.
    assert resultado.offset_seconds == 105.0
    assert resultado.excerpt_seconds == 30


def test_un_offset_pedido_gana_sobre_el_del_medio(tmp_path: Path, ffmpeg_falso) -> None:
    audio = AudioJobsFalso()

    resultado = comparar(tmp_path, audio, offset_seconds=12.5)

    assert resultado.offset_seconds == 12.5
    # Y no se gasta un ffprobe para algo que ya vino decidido.
    assert not any("-show_entries" in c for c in ffmpeg_falso)


def test_el_fragmento_compartido_no_queda_tirado_en_uploads(tmp_path: Path, ffmpeg_falso) -> None:
    audio = AudioJobsFalso()
    settings = settings_de_prueba(tmp_path)

    comparar(tmp_path, audio)

    # Cada trabajo ya tiene su nombre para el mismo contenido; el corte
    # compartido no lo borra nadie mas.
    quedan = {p.name for p in settings.uploads_path.iterdir()}
    assert quedan == {p["source_path"].name for p in audio.pedidos}
