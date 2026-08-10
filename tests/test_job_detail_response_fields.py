from __future__ import annotations

from pathlib import Path

from app.api.routes import audio_job_to_response, transcribe_job_to_response
from app.models import AudioJob, TranscribeJob

# ---------------------------------------------------------------------------
# El modal de detalle de trabajo muestra lo que el usuario eligio y lo que el
# pipeline decidio solo. Varias de esas cosas vivian en el job y no viajaban en
# la respuesta: el acabado y la cadena de voz (audio), el modo de salida y el
# idioma de doblaje (transcripcion), y la metadata entera de audio -- que es
# donde quedan `masteringSkipped` y la sonoridad medida. Sin ellas la pantalla
# no puede decir por que el resultado sono o salio distinto de lo pedido.
# ---------------------------------------------------------------------------


def _audio_job(**overrides) -> AudioJob:
    job = AudioJob(source_path=Path("cancion.mp3"), original_filename="cancion.mp3")
    for name, value in overrides.items():
        setattr(job, name, value)
    return job


def test_audio_response_carries_the_finishing_and_voice_choices():
    job = _audio_job(
        master="broadcast",
        voice_steps=["deesser", "presence"],
        voice_delivery="podcast",
        voice_presence_db=2.0,
    )

    response = audio_job_to_response(job)

    assert response.master == "broadcast"
    assert response.voice_steps == ["deesser", "presence"]
    assert response.voice_delivery == "podcast"
    assert response.voice_presence_db == 2.0


def test_audio_response_carries_the_pipeline_notes_in_metadata():
    job = _audio_job(master="streaming")
    job.metadata = {
        "loudnessBefore": -21.5,
        "loudnessTarget": -16.0,
        "masteringSkipped": "no se pudo medir la sonoridad",
    }

    response = audio_job_to_response(job)

    assert response.metadata["loudnessBefore"] == -21.5
    assert response.metadata["loudnessTarget"] == -16.0
    assert response.metadata["masteringSkipped"] == "no se pudo medir la sonoridad"


def test_audio_response_uses_camel_case_on_the_wire():
    job = _audio_job(voice_steps=["deesser"], voice_delivery="podcast", voice_presence_db=1.5)
    job.metadata = {"loudnessBefore": -20.0}

    payload = audio_job_to_response(job).model_dump(by_alias=True)

    assert payload["voiceSteps"] == ["deesser"]
    assert payload["voiceDelivery"] == "podcast"
    assert payload["voicePresenceDb"] == 1.5
    assert payload["metadata"]["loudnessBefore"] == -20.0


def test_audio_response_defaults_do_not_break_when_nothing_was_chosen():
    payload = audio_job_to_response(_audio_job()).model_dump(by_alias=True)

    assert payload["master"] is None
    assert payload["voiceSteps"] == []
    assert payload["voiceDelivery"] is None
    assert payload["voicePresenceDb"] is None
    assert payload["metadata"] == {}


def test_transcribe_response_carries_the_output_mode_and_dubbing_language():
    job = TranscribeJob(
        source_path=Path("charla.mp4"),
        original_filename="charla.mp4",
        model_id="whisper-small",
        output_mode="dubbed_video",
        target_language="en",
    )

    response = transcribe_job_to_response(job)
    payload = response.model_dump(by_alias=True)

    assert response.output_mode == "dubbed_video"
    assert response.target_language == "en"
    assert payload["outputMode"] == "dubbed_video"
    assert payload["targetLanguage"] == "en"


def test_transcribe_response_defaults_to_plain_text_without_a_dubbing_language():
    job = TranscribeJob(
        source_path=Path("charla.mp4"),
        original_filename="charla.mp4",
        model_id="whisper-small",
    )

    payload = transcribe_job_to_response(job).model_dump(by_alias=True)

    assert payload["outputMode"] == "text"
    assert payload["targetLanguage"] is None
