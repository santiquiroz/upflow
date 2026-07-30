from __future__ import annotations

from pathlib import Path

import pytest

from app.services.fetch.options import (
    ALLOWED_MAX_HEIGHTS,
    DEFAULT_MAX_HEIGHT,
    MAX_PLAYLIST_ITEMS,
    FetchRequest,
    build_plan,
    format_selector,
    output_template,
    validate_max_height,
    validate_playlist_limit,
)

FFMPEG_BIN = Path("/vendor/ffmpeg/bin")


def make_request(**overrides) -> FetchRequest:
    fields = {"url": "https://example.com/watch?v=abc", "output_dir": Path("/out")}
    fields.update(overrides)
    return FetchRequest(**fields)


# ---------------------------------------------------------------------------
# El techo de altura
# ---------------------------------------------------------------------------


def test_the_default_ceiling_is_not_the_most_expensive_one():
    """El default caro es exactamente lo que costo 2,8 horas en el modulo de escalado.

    Bajar 4K cuando alcanzaba 1080p es el mismo error con otra ropa, asi que el pedido
    caro tiene que ser una eleccion y no un descuido.
    """
    assert DEFAULT_MAX_HEIGHT == 1080
    assert DEFAULT_MAX_HEIGHT != max(ALLOWED_MAX_HEIGHTS)


@pytest.mark.parametrize("height", ALLOWED_MAX_HEIGHTS)
def test_every_offered_height_validates(height: int):
    validate_max_height(height)


def test_a_height_outside_the_list_is_refused():
    # Una altura arbitraria produce un selector que no matchea nada y una descarga que
    # falla despues de empezar; rechazarla al crear es mucho mas barato.
    with pytest.raises(ValueError, match="max_height"):
        validate_max_height(999)


# ---------------------------------------------------------------------------
# El selector de formato
# ---------------------------------------------------------------------------


def test_video_asks_for_separate_tracks_and_falls_back_to_a_combined_one():
    """El `+` es lo que da la mejor calidad: en la mayoria de los sitios solo existe en
    pistas separadas. El fallback cubre a los que publican un archivo unico.
    """
    selector = format_selector(1080, audio_only=False)

    assert "bestvideo[height<=1080]+bestaudio" in selector
    assert "/best[height<=1080]" in selector


def test_the_height_ceiling_appears_in_both_branches_of_the_selector():
    # Si el techo faltara en el fallback, un sitio con archivo combinado devolveria 4K
    # aunque se hayan pedido 720p.
    selector = format_selector(720, audio_only=False)

    assert selector.count("height<=720") == 2


def test_audio_only_never_asks_for_video():
    selector = format_selector(1080, audio_only=True)

    assert "bestvideo" not in selector
    assert "height" not in selector


# ---------------------------------------------------------------------------
# Playlists
# ---------------------------------------------------------------------------


def test_a_playlist_url_is_one_item_unless_it_is_asked_for():
    """La queja mas repetida de los descargadores: pegar un link y que arranquen 500.

    Una URL de playlist es tambien una URL de video, asi que el default tiene que ser
    el item suelto.
    """
    plan = build_plan(make_request(), FFMPEG_BIN)

    assert plan.options["noplaylist"] is True
    assert "playlistend" not in plan.options


def test_asking_for_the_playlist_caps_how_many_items_run():
    plan = build_plan(
        make_request(include_playlist=True, playlist_limit=5), FFMPEG_BIN
    )

    assert plan.options["noplaylist"] is False
    assert plan.options["playlistend"] == 5


def test_a_limit_beyond_the_hard_ceiling_is_refused():
    with pytest.raises(ValueError, match="playlist_limit"):
        validate_playlist_limit(MAX_PLAYLIST_ITEMS + 1)


def test_a_non_positive_limit_is_refused():
    with pytest.raises(ValueError):
        validate_playlist_limit(0)


def test_the_limit_is_only_validated_when_the_playlist_is_wanted():
    # Un limite absurdo no deberia romper un pedido que ni siquiera usa playlist.
    build_plan(make_request(include_playlist=False, playlist_limit=9999), FFMPEG_BIN)


def test_playlist_items_are_numbered_so_the_order_survives_on_disk():
    assert output_template(include_playlist=True).startswith("%(playlist_index)")
    # En un item suelto el indice seria ruido.
    assert "playlist_index" not in output_template(include_playlist=False)


# ---------------------------------------------------------------------------
# ffmpeg: el detalle que rompe en produccion
# ---------------------------------------------------------------------------


def test_the_plan_demands_ffmpeg_on_path_and_not_only_as_an_option():
    """Medido: pasar solo `ffmpeg_location` falla.

    Un intento real murio con "ffmpeg is not installed" mientras el postprocesador
    reportaba available=True. El camino de descarga parcial usa otro chequeo, que mira
    el PATH. Devolver el directorio en el plan es lo que obliga al llamador a ponerlo.
    """
    plan = build_plan(make_request(), FFMPEG_BIN)

    assert plan.options["ffmpeg_location"] == str(FFMPEG_BIN)
    assert FFMPEG_BIN in plan.env_path_entries


# ---------------------------------------------------------------------------
# Contenedor, metadata y subtitulos
# ---------------------------------------------------------------------------


def test_video_is_merged_into_mp4():
    plan = build_plan(make_request(), FFMPEG_BIN)

    assert plan.options["merge_output_format"] == "mp4"


def test_audio_only_extracts_instead_of_merging():
    plan = build_plan(make_request(audio_only=True), FFMPEG_BIN)

    assert "merge_output_format" not in plan.options
    assert plan.options["postprocessors"][0]["key"] == "FFmpegExtractAudio"


def test_metadata_travels_by_default():
    # Es lo que la gente espera; pedirlo aparte es la friccion que vuelve burocraticos
    # a los descargadores.
    assert build_plan(make_request(), FFMPEG_BIN).options["embedmetadata"] is True


def test_subtitles_are_only_fetched_when_a_language_was_asked_for():
    without = build_plan(make_request(), FFMPEG_BIN).options
    assert "writesubtitles" not in without
    assert without["embedsubtitles"] is False

    with_subs = build_plan(make_request(subtitle_languages=("es", "en")), FFMPEG_BIN).options
    assert with_subs["writesubtitles"] is True
    assert with_subs["subtitleslangs"] == ["es", "en"]
    assert with_subs["embedsubtitles"] is True


def test_drm_protected_formats_are_never_accepted():
    """Es la linea que separa un uso defendible de uno que no lo es.

    yt-dlp ya se niega, pero dejarlo explicito evita que un cambio futuro de default lo
    active sin que nadie lo note.
    """
    assert build_plan(make_request(), FFMPEG_BIN).options["allow_unplayable_formats"] is False


def test_the_output_lands_where_it_was_asked_for():
    plan = build_plan(make_request(output_dir=Path("/descargas")), FFMPEG_BIN)

    assert str(Path("/descargas")) in plan.options["outtmpl"]
