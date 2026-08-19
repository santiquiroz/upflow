"""Una tarjeta que dice "no se puede" mientras la app lo hace es un bug visible.

Las tres que arregla este archivo llevaban meses mintiendo, y ninguna suite lo
notó porque cada una probaba su módulo por separado: `audio.stems` decía que
separar en varias pistas devolvía ruido mientras `audio.karaoke` separaba al
lado, y `generate.textTo3d` / `generate.imageTo3d` decían que no había camino a
un ONNX ejecutable mientras `print.generate` / `print.generatePhoto` lo hacían.

Lo que se fija acá no son estados concretos —eso depende de qué tenga instalado
la máquina— sino que **dos tarjetas con la misma implementación detrás digan lo
mismo**. Es la única forma de que esto no vuelva a envejecer en silencio.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.routes import capability_tree
from app.config import Settings
from app.services.capabilities import CATALOG
from app.services.model_registry import ModelRegistry


def make_settings(tmp_path: Path, **extra) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None, **extra)


async def estados(settings: Settings) -> dict[str, str]:
    tree = await capability_tree(settings, ModelRegistry(settings))

    def recorrer(nodo):
        if isinstance(nodo, dict):
            if "id" in nodo and "status" in nodo:
                yield nodo
            for valor in nodo.values():
                yield from recorrer(valor)
        elif isinstance(nodo, list):
            for valor in nodo:
                yield from recorrer(valor)

    return {c["id"]: c["status"] for c in recorrer(tree.model_dump())}


PARES_CON_LA_MISMA_IMPLEMENTACION = [
    # (visto desde generación, visto desde impresión)
    ("generate.textTo3d", "print.generate"),
    ("generate.imageTo3d", "print.generatePhoto"),
    # Los subtítulos SON la transcripción alineada y muxeada: mismo modelo,
    # mismo trabajo, misma respuesta.
    ("video.subtitles", "audio.transcribe"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("uno,otro", PARES_CON_LA_MISMA_IMPLEMENTACION)
async def test_dos_vistas_de_lo_mismo_no_pueden_contradecirse(
    tmp_path: Path, uno: str, otro: str
) -> None:
    resultado = await estados(make_settings(tmp_path))

    assert resultado[uno] == resultado[otro], (
        f"{uno} dice {resultado[uno]} y {otro} dice {resultado[otro]}, "
        "y son el mismo motor"
    )


@pytest.mark.asyncio
async def test_ninguna_tarjeta_de_3d_se_declara_imposible(tmp_path: Path) -> None:
    # Shap-E (MIT en código y pesos) genera mallas desde texto y desde foto
    # desde la v0.5x. Cualquier tarjeta que diga "no hay camino" sobre 3D está
    # vieja, esté instalado el pack o no.
    resultado = await estados(make_settings(tmp_path))

    for capacidad in ("generate.textTo3d", "generate.imageTo3d"):
        assert resultado[capacidad] != "not_implemented"


@pytest.mark.asyncio
async def test_separar_stems_dejo_de_ser_imposible(tmp_path: Path) -> None:
    # umxhq separa en cuatro pistas desde la v0.68.0. La tarjeta decía que un
    # intento "devolvió ruido" y que publicarlo sería publicar algo roto.
    resultado = await estados(make_settings(tmp_path))

    assert resultado["audio.stems"] != "not_implemented"


def test_separar_stems_pide_un_modelo_de_mas_de_dos_pistas() -> None:
    # Y no el de karaoke: tener instalado un modelo de voz/instrumental no
    # habilita la separación en cuatro, y si compartieran requisito la tarjeta
    # volvería a mentir, ahora al revés.
    stems = next(c for c in CATALOG if c.id == "audio.stems")
    karaoke = next(c for c in CATALOG if c.id == "audio.karaoke")

    assert stems.requirements != karaoke.requirements
    assert stems.requirements[0].setting_attr == "karaoke_installed_multistem_model"


def test_el_requisito_de_stems_solo_cuenta_modelos_completos(tmp_path: Path) -> None:
    from app.services.engines.separation_models import SEPARATION_MODELS

    settings = make_settings(tmp_path)
    carpeta = settings.karaoke_model_dir_path
    carpeta.mkdir(parents=True, exist_ok=True)
    multi = next(s for s in SEPARATION_MODELS.values() if len(s.stems) > 2)

    for archivo in multi.files[:-1]:
        (carpeta / archivo).write_bytes(b"x")
    assert settings.karaoke_installed_multistem_model == ""

    (carpeta / multi.files[-1]).write_bytes(b"x")
    assert settings.karaoke_installed_multistem_model != ""


def test_un_modelo_de_dos_pistas_no_habilita_la_de_cuatro(tmp_path: Path) -> None:
    from app.services.engines.separation_models import SEPARATION_MODELS

    settings = make_settings(tmp_path)
    carpeta = settings.karaoke_model_dir_path
    carpeta.mkdir(parents=True, exist_ok=True)
    dos = next(s for s in SEPARATION_MODELS.values() if len(s.stems) == 2)
    for archivo in dos.files:
        (carpeta / archivo).write_bytes(b"x")

    assert settings.karaoke_installed_model != ""
    assert settings.karaoke_installed_multistem_model == ""


@pytest.mark.asyncio
async def test_los_subtitulos_dejaron_de_ser_imposibles(tmp_path: Path) -> None:
    # La app devuelve .srt, .vtt, la pista muxeada y el quemado en la imagen
    # desde hace varias versiones, y hasta traducidos. La tarjeta decía que
    # faltaba "el paso de sincronizado".
    resultado = await estados(make_settings(tmp_path))

    assert resultado["video.subtitles"] != "not_implemented"


def test_no_quedan_claves_de_motivo_sin_capacidad_que_las_use() -> None:
    """Una clave de i18n que promete un "todavía no" sobre algo ya enviado.

    Es el mismo defecto que las tarjetas, una capa más abajo: el texto sigue en
    los dos idiomas aunque ninguna capacidad lo referencie, listo para volver a
    aparecer si alguien lo reusa por el nombre.
    """
    from pathlib import Path as _Path

    usadas = {c.unavailable_reason_key for c in CATALOG if c.unavailable_reason_key}
    for idioma in ("es", "en"):
        texto = _Path(f"frontend/src/i18n/{idioma}.ts").read_text(encoding="utf-8")
        declaradas = {
            linea.split('"')[1]
            for linea in texto.splitlines()
            if linea.strip().startswith('"capability.reason.')
        }
        assert declaradas == usadas, (
            f"{idioma}: sobran {declaradas - usadas}, faltan {usadas - declaradas}"
        )


# ---------------------------------------------------------------------------
# Una funcion con DOS motores: alcanza con tener uno
# ---------------------------------------------------------------------------


def _voice_convert_card(settings, monkeypatch, *, openvoice: bool, speecht5: bool):
    """La tarjeta de conversion de voz con cada motor presente o ausente."""
    import app.services.capabilities as cap_mod

    original = cap_mod._path_exists

    def falso(s, requirement):
        if requirement.setting_attr.startswith("openvoice"):
            return openvoice
        if requirement.setting_attr.startswith("voice_conversion"):
            return speecht5
        return original(s, requirement)

    monkeypatch.setattr(cap_mod, "_path_exists", falso)

    class RegistroVacio:
        def list(self):
            return []

    resueltas = cap_mod.resolve_capabilities(settings, RegistroVacio())
    return next(c for c in resueltas if c.id == "audio.voiceConvert")


def test_con_openvoice_solo_la_tarjeta_no_pide_bajar_nada(tmp_path: Path, monkeypatch):
    """El bug que motivo `AnyOfRequirement`.

    La tarjeta declaraba las tres piezas de SpeechT5 y no sabia que existia
    OpenVoice: con el pack nuevo instalado y funcionando, la pantalla decia
    "falta descargar" y mandaba a bajar 400+ MB del modelo viejo para una
    funcion que ya andaba.
    """
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)

    tarjeta = _voice_convert_card(settings, monkeypatch, openvoice=True, speecht5=False)

    assert tarjeta.status == "available"
    assert tarjeta.missing_packs == ()


def test_con_speecht5_solo_sigue_disponible(tmp_path: Path, monkeypatch):
    # Quien ya lo tenia bajado no puede ver "falta descargar" porque salio otro
    # motor: seria pedirle 128 MB para recuperar algo que le funcionaba.
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)

    tarjeta = _voice_convert_card(settings, monkeypatch, openvoice=False, speecht5=True)

    assert tarjeta.status == "available"


def test_sin_ningun_motor_ofrece_el_que_conviene(tmp_path: Path, monkeypatch):
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)

    tarjeta = _voice_convert_card(settings, monkeypatch, openvoice=False, speecht5=False)

    assert tarjeta.status == "needs_setup"
    # UNO solo, y el preferido: ofrecer los dos seria pedir dos motores para una
    # sola funcion, y ofrecer el viejo seria mandar a bajar 400 MB de mas.
    assert tarjeta.missing_packs == ("openvoice",)


def test_el_pack_que_ofrece_se_puede_instalar_desde_la_app(tmp_path: Path, monkeypatch):
    from app.services.pack_provisioner import PACK_SCRIPTS

    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    tarjeta = _voice_convert_card(settings, monkeypatch, openvoice=False, speecht5=False)

    # Nombrar un pack que el provisioner no conoce deja al usuario sin boton y
    # sin salida: la tarjeta le dice qué falta y nada se lo puede dar.
    for pack in tarjeta.missing_packs:
        assert pack in PACK_SCRIPTS
