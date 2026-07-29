from __future__ import annotations

import pytest

from app.services.voice_chain import (
    ChainStep,
    FfmpegStage,
    ModelStage,
    VoiceChainOptions,
    build_filter_argument,
    build_filter_chain,
    delivery_choices,
    plan_stages,
    step_catalog,
)


def _index_of(chain: list[str], needle: str) -> int:
    for position, entry in enumerate(chain):
        if needle in entry:
            return position
    raise AssertionError(f"{needle!r} no esta en {chain}")


# ---------------------------------------------------------------------------
# El orden de la cadena
# ---------------------------------------------------------------------------


def test_full_chain_is_in_the_documented_order():
    chain = build_filter_chain(
        VoiceChainOptions(
            denoise="fft",
            highpass_hz=80,
            compress=True,
            presence_db=3.0,
            deesser=True,
            delivery="streaming",
        )
    )
    assert _index_of(chain, "afftdn") < _index_of(chain, "highpass")
    assert _index_of(chain, "highpass") < _index_of(chain, "acompressor")
    assert _index_of(chain, "acompressor") < _index_of(chain, "equalizer")
    assert _index_of(chain, "equalizer") < _index_of(chain, "deesser")
    assert _index_of(chain, "deesser") < _index_of(chain, "loudnorm")


def test_loudnorm_is_always_last():
    # Mide loudness integrado de la senal TERMINADA: medir antes de procesar
    # daria un numero que ya no es cierto al salir.
    chain = build_filter_chain(
        VoiceChainOptions(denoise="fft", compress=True, deesser=True, delivery="ebu_r128")
    )
    assert "loudnorm" in chain[-1]


def test_denoise_is_always_first():
    # Quitar es menos destructivo que agregar, y todo lo posterior trabaja
    # sobre una senal mas limpia.
    chain = build_filter_chain(
        VoiceChainOptions(denoise="fft", highpass_hz=80, compress=True, deesser=True)
    )
    assert "afftdn" in chain[0]


def test_highpass_precedes_compression_even_without_denoise():
    chain = build_filter_chain(
        VoiceChainOptions(denoise="none", highpass_hz=100, compress=True)
    )
    assert _index_of(chain, "highpass") < _index_of(chain, "acompressor")


# ---------------------------------------------------------------------------
# Destinos de entrega
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delivery", "lufs", "true_peak"),
    [
        ("streaming", -14.0, -1.0),
        ("apple_music", -16.0, -1.0),
        ("ebu_r128", -23.0, -1.0),
        ("atsc_a85", -24.0, -2.0),
    ],
)
def test_delivery_targets_use_their_published_numbers(delivery, lufs, true_peak):
    chain = build_filter_chain(VoiceChainOptions(delivery=delivery, deesser=False))
    loudnorm = chain[_index_of(chain, "loudnorm")]
    assert f"I={lufs}" in loudnorm
    assert f"TP={true_peak}" in loudnorm


def test_atsc_has_a_looser_true_peak_ceiling_than_ebu():
    # -2 dBTP contra -1: no es un typo, son especificaciones distintas.
    atsc = build_filter_chain(VoiceChainOptions(delivery="atsc_a85", deesser=False))
    ebu = build_filter_chain(VoiceChainOptions(delivery="ebu_r128", deesser=False))
    assert "TP=-2.0" in atsc[-1]
    assert "TP=-1.0" in ebu[-1]


def test_delivery_none_adds_no_loudnorm():
    chain = build_filter_chain(VoiceChainOptions(delivery="none", deesser=True))
    assert not any("loudnorm" in entry for entry in chain)


def test_delivery_choices_expose_the_numbers_for_the_ui():
    choices = delivery_choices()
    by_id = {c["id"]: c for c in choices}
    assert by_id["streaming"]["lufs"] == -14.0
    assert by_id["atsc_a85"]["truePeakDb"] == -2.0
    assert all(c["label"] for c in choices)


# ---------------------------------------------------------------------------
# Pasos individuales
# ---------------------------------------------------------------------------


def test_every_step_can_be_turned_off_independently():
    chain = build_filter_chain(
        VoiceChainOptions(
            denoise="none", highpass_hz=None, compress=False,
            presence_db=None, deesser=False, delivery="none",
        )
    )
    assert chain == []


def test_no_steps_means_no_filter_argument():
    options = VoiceChainOptions(
        denoise="none", highpass_hz=None, compress=False,
        presence_db=None, deesser=False, delivery="none",
    )
    assert build_filter_argument(options) is None


def test_filter_argument_joins_with_commas():
    argument = build_filter_argument(
        VoiceChainOptions(denoise="none", highpass_hz=80, compress=False, deesser=True)
    )
    assert argument == "highpass=f=80,deesser"


def test_presence_boost_targets_the_human_voice_band():
    chain = build_filter_chain(
        VoiceChainOptions(denoise="none", compress=False, presence_db=4.5, deesser=False)
    )
    equalizer = chain[_index_of(chain, "equalizer")]
    assert "f=3000" in equalizer
    assert "g=4.5" in equalizer


def test_rnnoise_requires_its_model_path():
    with pytest.raises(ValueError, match="rnnoise"):
        build_filter_chain(VoiceChainOptions(denoise="rnnoise"))


def test_rnnoise_uses_the_model_path_it_receives():
    chain = build_filter_chain(
        VoiceChainOptions(denoise="rnnoise", highpass_hz=None, compress=False, deesser=False),
        rnnoise_model="C\\:/models/rnnoise.rnnn",
    )
    assert chain == ["arnndn=m=C\\:/models/rnnoise.rnnn"]


def test_defaults_are_a_sane_dialogue_chain():
    # Sin argumentos: highpass + compresor + de-esser. Es la base de una cadena
    # de dialogo sin tocar tono ni loudness.
    chain = build_filter_chain(VoiceChainOptions())
    assert any("highpass" in e for e in chain)
    assert any("acompressor" in e for e in chain)
    assert any("deesser" in e for e in chain)
    assert not any("loudnorm" in e for e in chain)
    assert not any("equalizer" in e for e in chain)


# ---------------------------------------------------------------------------
# Pasos mixtos DSP + modelo, y el plan de pasadas
# ---------------------------------------------------------------------------


def _filter(step_id: str, expr: str) -> ChainStep:
    return ChainStep(id=step_id, kind="filter", label=step_id, filter_expr=expr)


def _model(step_id: str, capability: str) -> ChainStep:
    return ChainStep(id=step_id, kind="model", label=step_id, model_capability=capability)


def test_consecutive_filters_collapse_into_one_pass():
    # Cada pasada de ffmpeg es un decode y encode completo del audio: fusionar
    # no es cosmetico.
    stages = plan_stages([
        _filter("highpass", "highpass=f=80"),
        _filter("compress", "acompressor"),
        _filter("deesser", "deesser"),
    ])
    assert len(stages) == 1
    assert isinstance(stages[0], FfmpegStage)
    assert stages[0].filter_argument == "highpass=f=80,acompressor,deesser"
    assert stages[0].step_ids == ("highpass", "compress", "deesser")


def test_a_model_step_splits_the_chain_in_two_passes():
    stages = plan_stages([
        _filter("highpass", "highpass=f=80"),
        _model("isolate", "audio.voice.isolate"),
        _filter("deesser", "deesser"),
    ])
    assert [type(s).__name__ for s in stages] == ["FfmpegStage", "ModelStage", "FfmpegStage"]
    assert stages[0].filter_argument == "highpass=f=80"
    assert stages[1].model_capability == "audio.voice.isolate"
    assert stages[2].filter_argument == "deesser"


def test_a_model_first_chain_starts_with_the_model_stage():
    # El caso real de mejora de voz: aislar la voz con un modelo y despues
    # trabajarla con DSP.
    stages = plan_stages([
        _model("isolate", "audio.voice.isolate"),
        _filter("highpass", "highpass=f=80"),
        _filter("loudnorm", "loudnorm=I=-23.0:TP=-1.0:LRA=11"),
    ])
    assert isinstance(stages[0], ModelStage)
    assert len(stages) == 2
    assert stages[1].step_ids == ("highpass", "loudnorm")


def test_consecutive_model_steps_do_not_merge():
    stages = plan_stages([
        _model("isolate", "audio.voice.isolate"),
        _model("denoise", "audio.denoise.deepfilternet"),
    ])
    assert len(stages) == 2
    assert all(isinstance(s, ModelStage) for s in stages)


def test_only_models_means_no_ffmpeg_pass():
    stages = plan_stages([_model("denoise", "audio.denoise.deepfilternet")])
    assert len(stages) == 1
    assert isinstance(stages[0], ModelStage)


def test_no_steps_means_no_stages():
    assert plan_stages([]) == []


def test_a_filter_step_without_an_expression_is_rejected():
    with pytest.raises(ValueError, match="expresion"):
        ChainStep(id="broken", kind="filter", label="broken")


def test_a_model_step_without_a_capability_is_rejected():
    with pytest.raises(ValueError, match="capacidad"):
        ChainStep(id="broken", kind="model", label="broken")


def test_step_order_is_preserved_verbatim():
    # plan_stages no reordena: quien arma la cadena es responsable del orden.
    steps = [
        _filter("deesser", "deesser"),
        _filter("highpass", "highpass=f=80"),
    ]
    stages = plan_stages(steps)
    assert stages[0].filter_argument == "deesser,highpass=f=80"


# ---------------------------------------------------------------------------
# Catalogo para publico no tecnico
# ---------------------------------------------------------------------------


def test_every_step_has_plain_language_help():
    for step in step_catalog():
        assert step.label and not step.label.endswith(":")
        # La descripcion tiene que explicar, no repetir la etiqueta.
        assert len(step.description) > 40
        assert step.description != step.label


def test_every_delivery_target_has_plain_language_help():
    for choice in delivery_choices():
        assert len(str(choice["description"])) > 40


def test_catalog_order_matches_the_real_chain_order():
    # Si el catalogo se desincroniza del orden real, la UI mostraria una cadena
    # que no es la que se ejecuta. Este test es el que lo impide.
    chain = build_filter_chain(
        VoiceChainOptions(
            denoise="fft", highpass_hz=80, compress=True,
            presence_db=3.0, deesser=True, delivery="streaming",
        )
    )
    expr_by_step = {
        "denoise": "afftdn",
        "highpass": "highpass",
        "compress": "acompressor",
        "presence": "equalizer",
        "deesser": "deesser",
        "loudness": "loudnorm",
    }
    catalog_ids = [s.id for s in step_catalog()]
    positions = [_index_of(chain, expr_by_step[step_id]) for step_id in catalog_ids]
    assert positions == sorted(positions)


def test_catalog_covers_every_step_the_chain_can_emit():
    catalog_ids = {s.id for s in step_catalog()}
    assert catalog_ids == {"denoise", "highpass", "compress", "presence", "deesser", "loudness"}


def test_defaults_in_the_catalog_match_the_options_defaults():
    defaults = {s.id: s.default_enabled for s in step_catalog()}
    options = VoiceChainOptions()
    assert defaults["highpass"] is bool(options.highpass_hz)
    assert defaults["compress"] is options.compress
    assert defaults["deesser"] is options.deesser
    assert defaults["denoise"] is (options.denoise != "none")
    assert defaults["presence"] is (options.presence_db is not None)
    assert defaults["loudness"] is (options.delivery != "none")
