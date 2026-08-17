"""Tests de integración del servidor MCP contra una API simulada.

Se intercepta el transporte HTTP (httpx.MockTransport) en vez de levantar la
app: lo que se prueba es el contrato del cliente MCP — rutas correctas, forma
normalizada de la salida y manejo de errores — no la lógica del servidor.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.mcp import client as mcp_client
from app.mcp.server import (
    upflow_download_result,
    upflow_job_status,
    upflow_list_jobs,
    upflow_status,
    upflow_upscale_image,
    upflow_wait_job,
)


@pytest.fixture(autouse=True)
def reset_mcp_client():
    yield
    mcp_client._client = None
    mcp_client._login_attempted = False


def install_mock(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    transport = httpx.MockTransport(handler)
    mock = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    monkeypatch.setattr(mcp_client, "_get_client", lambda: mock)


async def test_status_aggregates_four_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        payloads = {
            "/api/v1/health": {"status": "ok", "queueDepth": 0},
            "/api/v1/engine": {"engine": "realesrgan", "available": True},
            "/api/v1/devices": {"devices": [], "defaultDeviceId": "dml:0"},
            "/api/v1/auth/me": {"username": "local", "role": "admin"},
        }
        return httpx.Response(200, json=payloads[request.url.path])

    install_mock(monkeypatch, handler)
    result = json.loads(await upflow_status())

    assert result["health"]["status"] == "ok"
    assert result["me"]["role"] == "admin"
    assert len(seen) == 4


async def test_job_status_normalizes_video_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/video/jobs/v1"
        return httpx.Response(
            200,
            json={
                "jobId": "v1",
                "status": "running",
                "progressPct": 42.5,
                "metadata": {"stage": "encode"},
            },
        )

    install_mock(monkeypatch, handler)
    result = json.loads(await upflow_job_status("video", "v1"))

    assert result["family"] == "video"
    assert result["jobId"] == "v1"
    assert result["stage"] == "encode"


async def test_job_status_unknown_family_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    install_mock(monkeypatch, lambda request: httpx.Response(200, json={}))
    result = await upflow_job_status("nope", "x")
    assert result.startswith("Error")
    assert "image" in result and "shape3d" in result


async def test_job_status_404_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    install_mock(
        monkeypatch,
        lambda request: httpx.Response(404, json={"detail": "Job not found"}),
    )
    result = await upflow_job_status("image", "missing")
    assert result.startswith("Error")
    assert "Job not found" in result


async def test_wait_job_returns_terminal_state(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        status = "completed" if calls["n"] >= 2 else "running"
        return httpx.Response(200, json={"jobId": "a1", "status": status})

    install_mock(monkeypatch, handler)
    monkeypatch.setattr("app.mcp.server.WAIT_POLL_SECONDS", 0.01)
    result = json.loads(await upflow_wait_job("image", "a1", timeout_seconds=30))

    assert result["status"] == "completed"
    assert "waitTimedOut" not in result


async def test_upscale_image_uploads_waits_and_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "foto.png"
    source.write_bytes(b"png-bytes")
    output_dir = tmp_path / "out"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/jobs" and request.method == "POST":
            assert b"png-bytes" in request.read()
            return httpx.Response(202, json={"jobId": "img1", "status": "queued"})
        if request.url.path == "/api/v1/jobs/img1":
            return httpx.Response(200, json={"jobId": "img1", "status": "completed"})
        if request.url.path == "/api/v1/jobs/img1/download":
            return httpx.Response(200, content=b"resultado")
        raise AssertionError(f"ruta inesperada: {request.url.path}")

    install_mock(monkeypatch, handler)
    monkeypatch.setattr("app.mcp.server.WAIT_POLL_SECONDS", 0.01)
    result = json.loads(
        await upflow_upscale_image(str(source), destination_path=str(output_dir))
    )

    assert result["status"] == "completed"
    saved = Path(result["outputPath"])
    assert saved.read_bytes() == b"resultado"


async def test_upscale_image_missing_file_is_actionable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_mock(monkeypatch, lambda request: httpx.Response(200, json={}))
    result = await upflow_upscale_image(str(tmp_path / "no-existe.png"))
    assert result.startswith("Error")
    assert "no-existe.png" in result


def listing_mock(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Un listado de un job por familia, anotando que rutas se pidieron."""
    visitadas: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        visitadas.append(request.url.path)
        return httpx.Response(200, json={"jobs": [{"id": "x1", "jobId": "x1", "status": "running"}]})

    install_mock(monkeypatch, handler)
    return visitadas


async def test_list_jobs_lists_the_families_that_had_no_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Antes devolvian un error pidiendo guardar el jobId: un job de esas tres
    familias quedaba inalcanzable si el agente lo perdia."""
    visitadas = listing_mock(monkeypatch)

    for name, path in (
        ("transcribe", "/api/v1/transcribe/jobs"),
        ("download", "/api/v1/download/jobs"),
        ("shape3d", "/api/v1/print/generate"),
    ):
        result = json.loads(await upflow_list_jobs(name))
        assert result[name][0]["jobId"] == "x1"
        assert visitadas[-1] == path


async def test_list_jobs_without_family_covers_the_seven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing_mock(monkeypatch)

    result = json.loads(await upflow_list_jobs())

    assert set(result) == {
        "image", "video", "audio", "generation", "transcribe", "download", "shape3d",
    }


async def test_download_result_transcribe_passes_format_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/transcribe/jobs/t1/download"
        assert request.url.params["fmt"] == "srt"
        assert request.url.params["translate_to"] == "es"
        return httpx.Response(200, content=b"1\n00:00:00,000 --> 00:00:01,000\nhola\n")

    install_mock(monkeypatch, handler)
    result = json.loads(
        await upflow_download_result(
            "transcribe",
            "t1",
            str(tmp_path),
            transcript_format="srt",
            translate_to="es",
        )
    )

    saved = Path(result["outputPath"])
    assert saved.name == "transcript.srt"
    assert saved.exists()


async def test_process_audio_separate_sends_karaoke_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "cancion.mp3"
    source.write_bytes(b"mp3-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/audio/jobs"
        body = request.read()
        assert b'name="separate"' in body and b"true" in body
        return httpx.Response(202, json={"jobId": "a1", "status": "queued"})

    install_mock(monkeypatch, handler)
    from app.mcp.server import upflow_process_audio

    result = json.loads(await upflow_process_audio(str(source), separate=True))
    assert result["jobId"] == "a1"


async def test_process_audio_forwards_the_cleanup_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "cancion.mp3"
    source.write_bytes(b"mp3-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/audio/jobs"
        body = request.read()
        # Viaja tal cual llego: el ORDEN lo fija el catalogo del backend, asi
        # que la tool no reordena ni valida por su cuenta.
        assert b'name="cleanup_steps"' in body
        assert b"reverb_hq,denoise" in body
        # Se combina con el resto de la cadena en el MISMO job.
        assert b'name="master"' in body
        return httpx.Response(202, json={"jobId": "a2", "status": "queued"})

    install_mock(monkeypatch, handler)
    from app.mcp.server import upflow_process_audio

    result = json.loads(
        await upflow_process_audio(
            str(source), cleanup_steps="reverb_hq,denoise", master="streaming"
        )
    )
    assert result["jobId"] == "a2"


async def test_process_audio_redundant_cleanup_propagates_the_api_400(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Sin whitelist local: la exclusion por familia es una regla del catalogo
    # del backend, y el 400 de la API es la unica fuente de verdad.
    source = tmp_path / "cancion.mp3"
    source.write_bytes(b"mp3-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "detail": (
                    "Pasos de limpieza redundantes: 'deecho_normal' y "
                    "'deecho_aggressive' hacen la misma tarea (quitar eco)."
                )
            },
        )

    install_mock(monkeypatch, handler)
    from app.mcp.server import upflow_process_audio

    result = await upflow_process_audio(
        str(source), cleanup_steps="deecho_normal,deecho_aggressive"
    )
    assert result.startswith("Error")
    assert "redundantes" in result


async def test_process_audio_can_be_a_pure_format_conversion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Sin ningun paso: el form solo lleva formato y calidad, y la API lo acepta.
    source = tmp_path / "cancion.flac"
    source.write_bytes(b"flac-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        assert b'name="output_format"' in body and b"m4a" in body
        assert b'name="lossy_quality"' in body and b"balanced" in body
        assert b'name="denoise"' not in body
        assert b'name="master"' not in body
        return httpx.Response(202, json={"jobId": "a9", "status": "queued"})

    install_mock(monkeypatch, handler)
    from app.mcp.server import upflow_process_audio

    result = json.loads(
        await upflow_process_audio(str(source), output_format="m4a", lossy_quality="balanced")
    )
    assert result["jobId"] == "a9"


async def test_process_audio_docstring_documents_the_conversion_contract() -> None:
    # La docstring ES el contrato para un agente: sin decir que la conversion
    # pura existe y que un resample forzado queda en metadata, un agente asume
    # que hace falta un paso y que la tasa siempre se conserva.
    from app.mcp.server import upflow_process_audio

    doc = upflow_process_audio.__doc__ or ""
    assert "CONVERSIÓN PURA" in doc
    assert "conversionResampled" in doc
    assert "m4a" in doc
    assert "lossy_quality" in doc


async def test_process_audio_cleanup_docstring_states_the_fixed_order() -> None:
    # La docstring ES el contrato para un agente: si no dice que el orden es
    # fijo y que hay exclusividad, el agente va a intentar imponer los suyos.
    from app.mcp.server import upflow_process_audio

    doc = upflow_process_audio.__doc__ or ""
    assert "ORDEN es FIJO" in doc
    assert "EXCLUSIVIDAD" in doc
    assert "deecho_dereverb" in doc


async def test_download_result_audio_stem_passes_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/audio/jobs/a1/download"
        assert request.url.params["stem"] == "vocals"
        return httpx.Response(200, content=b"wav-bytes")

    install_mock(monkeypatch, handler)
    result = json.loads(
        await upflow_download_result("audio", "a1", str(tmp_path), stem="vocals")
    )
    assert Path(result["outputPath"]).name == "vocals.flac"


async def test_process_audio_separation_model_without_separate_propagates_400(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "cancion.mp3"
    source.write_bytes(b"mp3-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        assert b'name="separation_model"' in body
        return httpx.Response(
            400, json={"detail": "separation_model solo aplica cuando separate=true."}
        )

    install_mock(monkeypatch, handler)
    from app.mcp.server import upflow_process_audio

    result = await upflow_process_audio(str(source), separation_model="voc_ft")
    assert result.startswith("Error")
    assert "separate=true" in result


async def test_download_result_unknown_stem_propagates_api_400(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Sin whitelist local: los stems dependen del modelo del job (karaoke usa
    # instrumental/vocals, reverb_hq usa dry/wet) — el 400 de la API es la
    # verdad y llega al cliente MCP con los válidos.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["stem"] == "drums"
        return httpx.Response(
            400, json={"detail": "stem inválido; válidos: dry, wet"}
        )

    install_mock(monkeypatch, handler)
    result = await upflow_download_result("audio", "a1", str(tmp_path), stem="drums")
    assert result.startswith("Error")
    assert "dry, wet" in result


async def test_connection_refused_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("rechazado")

    install_mock(monkeypatch, handler)
    result = await upflow_job_status("image", "x")
    assert result.startswith("Error")
    assert "UPFLOW_URL" in result
