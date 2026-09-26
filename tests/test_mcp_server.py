"""Tests de integración del servidor MCP contra una API simulada.

Se intercepta el transporte HTTP (httpx.MockTransport) en vez de levantar la
app: lo que se prueba es el contrato del cliente MCP — rutas correctas, forma
normalizada de la salida y manejo de errores — no la lógica del servidor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from app.mcp import client as mcp_client
from app.mcp import headless_tools
from app.mcp.server import (
    upflow_cctv_check_unchanged,
    upflow_cctv_clarify,
    upflow_cctv_probe,
    upflow_cctv_roi_fuse,
    upflow_download_result,
    upflow_job_status,
    upflow_list_jobs,
    upflow_restore_analyze,
    upflow_restore_photo,
    upflow_restore_recompose,
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


async def test_restore_analyze_uploads_the_photo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "abuela.jpg"
    source.write_bytes(b"jpg-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/restore/analyze"
        assert b"jpg-bytes" in request.read()
        return httpx.Response(200, json={"token": "t" * 32, "proposedSteps": ["repair"]})

    install_mock(monkeypatch, handler)
    result = json.loads(await upflow_restore_analyze(str(source)))
    assert result["proposedSteps"] == ["repair"]


async def test_restore_photo_with_token_sends_the_form_waits_and_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/restore/jobs":
            form = {key: values[0] for key, values in parse_qs(request.read().decode()).items()}
            assert form == {
                "token": "tok123",
                "restore_steps": "denoise,tone",
                "restore_options": '{"tone": {"strength": 0.4}}',
                "scale": "1",
                "model_name": "realesrgan-x4plus",
                "output_format": "png",
            }
            return httpx.Response(202, json={"jobId": "r1", "status": "queued", "restoreSteps": ["denoise", "tone"]})
        if request.url.path == "/api/v1/jobs/r1":
            return httpx.Response(
                200, json={"jobId": "r1", "status": "completed", "restoreSteps": ["denoise", "tone"], "metadata": {}}
            )
        if request.url.path == "/api/v1/jobs/r1/download":
            return httpx.Response(200, content=b"restaurada")
        raise AssertionError(f"ruta inesperada: {request.url.path}")

    install_mock(monkeypatch, handler)
    monkeypatch.setattr("app.mcp.server.WAIT_POLL_SECONDS", 0.01)
    result = json.loads(
        await upflow_restore_photo(
            token="tok123",
            steps=["denoise", "tone"],
            options={"tone": {"strength": 0.4}},
            destination_path=str(tmp_path / "out"),
        )
    )

    assert result["status"] == "completed"
    assert result["restoreSteps"] == ["denoise", "tone"]
    assert Path(result["outputPath"]).read_bytes() == b"restaurada"


async def test_restore_photo_needs_one_source_and_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    install_mock(monkeypatch, lambda request: pytest.fail("no request expected"))
    assert (await upflow_restore_photo(steps=["tone"])).startswith("Error")
    assert (await upflow_restore_photo(file_path="a.png", token="t", steps=["tone"])).startswith("Error")
    assert "upflow_restore_analyze" in await upflow_restore_photo(token="t")


async def test_restore_recompose_posts_faces_and_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/restore/jobs/r1/recompose":
            assert json.loads(request.read()) == {"faces": {"0": {"enabled": False, "blend": 0.3}}}
            return httpx.Response(200, json={"sidecar": {"faces": []}})
        if request.url.path == "/api/v1/jobs/r1/download":
            return httpx.Response(200, content=b"recompuesta")
        raise AssertionError(f"ruta inesperada: {request.url.path}")

    install_mock(monkeypatch, handler)
    faces = {"0": {"enabled": False, "blend": 0.3}}
    result = json.loads(await upflow_restore_recompose("r1", faces, destination_path=str(tmp_path / "r.png")))
    assert result["jobId"] == "r1"
    assert result["sidecar"] == {"faces": []}
    assert Path(result["outputPath"]).read_bytes() == b"recompuesta"


async def test_restore_photo_runs_in_process_when_the_server_is_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    seen: dict[str, object] = {}

    async def inprocess(**kwargs) -> str:
        seen.update(kwargs)
        return json.dumps({"ok": True})

    install_mock(monkeypatch, refused)
    monkeypatch.delenv("UPFLOW_MCP_MODE", raising=False)
    monkeypatch.setattr("app.mcp.headless_tools.upflow_restore_photo_headless", inprocess)
    source = tmp_path / "foto.png"
    source.write_bytes(b"png")
    result = json.loads(await upflow_restore_photo(file_path=str(source), steps=["tone"], scale=2))
    assert result == {"ok": True}
    assert seen["file_path"] == str(source)
    assert seen["steps"] == ["tone"]
    assert seen["scale"] == 2


# ---------------------------------------------------------------- CCTV


CCTV_ANALYSIS = {
    "token": "tok123",
    "sourceSha256": "ab" * 32,
    "video": {"lite": None},
    "quality": {"interlace": {"interlaced": True}},
    "suggestedPreset": "day",
    "modeAvailable": True,
    "warnings": [],
}


@pytest.fixture
def server_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(headless_tools.MODE_ENV, raising=False)


def step_ids(steps: list[dict]) -> list[str]:
    return [step["id"] for step in steps]


async def test_cctv_probe_waits_for_a_long_analysis_and_adds_the_preset_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server_mode: None
) -> None:
    clip = tmp_path / "camara.mp4"
    clip.write_bytes(b"clip-bytes")
    polls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/video/cctv/analyze":
            assert b"clip-bytes" in request.read()
            return httpx.Response(
                202,
                json={"analysisJobId": "an1", "status": "running", "statusUrl": "/api/v1/video/cctv/analysis/an1"},
            )
        assert request.url.path == "/api/v1/video/cctv/analysis/an1"
        polls["n"] += 1
        status = "completed" if polls["n"] >= 2 else "running"
        result = CCTV_ANALYSIS if status == "completed" else None
        return httpx.Response(200, json={"analysisJobId": "an1", "status": status, "statusUrl": "/api/v1/video/cctv/analysis/an1", "result": result})

    install_mock(monkeypatch, handler)
    monkeypatch.setattr("app.mcp.server.WAIT_POLL_SECONDS", 0.01)
    result = json.loads(await upflow_cctv_probe(str(clip)))

    assert result["token"] == "tok123" and polls["n"] == 2
    assert set(result["presetSteps"]) == {"day", "night_ir", "analog", "low_res"}
    assert "deinterlace" in step_ids(result["presetSteps"]["day"])


async def test_cctv_probe_returns_a_short_analysis_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server_mode: None
) -> None:
    clip = tmp_path / "camara.mp4"
    clip.write_bytes(b"clip")
    install_mock(monkeypatch, lambda request: httpx.Response(200, json={**CCTV_ANALYSIS, "quality": None}))

    result = json.loads(await upflow_cctv_probe(str(clip)))

    assert result["sourceSha256"] == "ab" * 32
    assert "deinterlace" not in step_ids(result["presetSteps"]["day"])


async def test_cctv_probe_reports_a_failed_analysis_with_its_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server_mode: None
) -> None:
    clip = tmp_path / "camara.mp4"
    clip.write_bytes(b"clip")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/video/cctv/analyze":
            return httpx.Response(202, json={"analysisJobId": "an1", "status": "running", "statusUrl": "/s/an1"})
        return httpx.Response(
            200,
            json={"status": "failed", "statusUrl": "/s/an1", "error": "The file has no video stream.", "errorKey": "cctv.error.noVideoStream"},
        )

    install_mock(monkeypatch, handler)
    monkeypatch.setattr("app.mcp.server.WAIT_POLL_SECONDS", 0.01)
    result = await upflow_cctv_probe(str(clip))

    assert result.startswith("Error") and "cctv.error.noVideoStream" in result


async def test_cctv_clarify_posts_the_json_contract_and_keeps_the_cctv_summary(
    monkeypatch: pytest.MonkeyPatch, server_mode: None
) -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/video/cctv/jobs" and request.method == "POST"
        sent.append(json.loads(request.read()))
        summary = {"task": "clarify", "sourceSha256": "ab" * 32, "artifacts": [], "verifyUrl": None}
        return httpx.Response(202, json={"jobId": "v9", "status": "queued", "cctv": summary})

    install_mock(monkeypatch, handler)
    result = json.loads(
        await upflow_cctv_clarify(
            "tok123",
            preset="day",
            steps=[{"id": "denoise", "params": {"filter": "hqdn3d"}}, {"id": "deblock"}],
            osd_boxes=[[0, 0, 96, 24]],
            osd_confirmed=True,
            trim=[3, 40],
            still_frames=[5, 30],
            acquisition={"recorderMake": "HiLook"},
        )
    )

    assert result["jobId"] == "v9" and result["family"] == "video"
    assert result["cctv"]["sourceSha256"] == "ab" * 32
    assert sent == [
        {
            "token": "tok123",
            "task": "clarify",
            "preset": "day",
            "steps": [{"id": "denoise", "params": {"filter": "hqdn3d"}}, {"id": "deblock", "params": {}}],
            "osdBoxes": [[0, 0, 96, 24]],
            "osdBoxesConfirmed": True,
            "noOsd": False,
            "trim": [3, 40],
            "stillFrames": [5, 30],
            "acquisition": {"recorderMake": "HiLook"},
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"preset": "day"}, "presetSteps"),
        ({"steps": [], "trim": [1, 2, 3]}, "trim"),
        ({"steps": ["denoise"]}, "'id'"),
    ],
)
async def test_cctv_clarify_rejects_bad_choices_before_calling_the_api(
    monkeypatch: pytest.MonkeyPatch, server_mode: None, kwargs: dict, fragment: str
) -> None:
    install_mock(monkeypatch, lambda request: pytest.fail("the API must not be called"))

    result = json.loads(await upflow_cctv_clarify("tok123", no_osd=True, **kwargs))

    assert result["ok"] is False and result["code"] == 2 and fragment in result["error"]


async def test_cctv_clarify_propagates_the_keyed_api_400(
    monkeypatch: pytest.MonkeyPatch, server_mode: None
) -> None:
    detail = {"key": "cctv.error.osdUnconfirmed", "reason": "Confirm the on-screen text boxes"}
    install_mock(monkeypatch, lambda request: httpx.Response(400, json={"detail": detail}))

    result = await upflow_cctv_clarify("tok123", steps=[])

    assert result.startswith("Error") and "cctv.error.osdUnconfirmed" in result


async def test_cctv_clarify_falls_back_in_process_when_the_server_is_down(
    monkeypatch: pytest.MonkeyPatch, server_mode: None
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("rechazado")

    seen: list[tuple] = []

    async def inprocess(token, choices, destination_dir=""):
        seen.append((token, choices.no_osd, destination_dir))
        return json.dumps({"ok": True, "jobId": "inline"})

    install_mock(monkeypatch, handler)
    monkeypatch.setattr(headless_tools, "upflow_cctv_clarify_headless", inprocess)
    result = json.loads(await upflow_cctv_clarify("tok123", steps=[], no_osd=True, destination_dir="C:/caso"))

    assert result["jobId"] == "inline" and seen == [("tok123", True, "C:/caso")]


async def test_cctv_check_unchanged_posts_verify(monkeypatch: pytest.MonkeyPatch, server_mode: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/video/jobs/v9/verify" and request.method == "POST"
        return httpx.Response(200, json={"ok": False, "checked": 3, "mismatches": ["02_processed/a.mkv"], "missing": []})

    install_mock(monkeypatch, handler)
    result = json.loads(await upflow_cctv_check_unchanged("v9"))

    assert result["ok"] is False and result["mismatches"] == ["02_processed/a.mkv"]


async def test_cctv_check_unchanged_rejects_a_path_as_job_id(monkeypatch: pytest.MonkeyPatch, server_mode: None) -> None:
    install_mock(monkeypatch, lambda request: pytest.fail("the API must not be called"))

    result = json.loads(await upflow_cctv_check_unchanged("../v9"))

    assert result["ok"] is False and result["code"] == 2


async def test_cctv_check_unchanged_reads_a_moved_folder_in_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server_mode: None
) -> None:
    install_mock(monkeypatch, lambda request: pytest.fail("a moved folder is checked without the server"))
    folder = tmp_path / "v9.cctv"
    folder.mkdir()
    (folder / "report.json").write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(b"{}").hexdigest()
    (folder / "SHA256SUMS.txt").write_text(f"{digest} *report.json\n", encoding="utf-8")
    monkeypatch.setattr(headless_tools, "get_context", lambda: pytest.fail("no context needed"))

    result = json.loads(await upflow_cctv_check_unchanged("v9", output_dir=str(folder)))

    assert result["ok"] is True and result["checked"] == 1


# ---------------------------------------------------------------- CCTV: foto multi-cuadro de una ROI


async def test_cctv_roi_fuse_posts_the_json_contract(monkeypatch: pytest.MonkeyPatch, server_mode: None) -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/video/cctv/jobs" and request.method == "POST"
        sent.append(json.loads(request.read()))
        summary = {"task": "roi_fusion", "sourceSha256": "ab" * 32, "artifacts": [], "roi": None}
        return httpx.Response(202, json={"jobId": "v7", "status": "queued", "cctv": summary})

    install_mock(monkeypatch, handler)
    result = json.loads(
        await upflow_cctv_roi_fuse(
            "tok123",
            frames=[10, 40],
            reference=22,
            box=[100, 80, 64, 24],
            kind="plate",
            scale=3,
            method="trimmed_mean",
            preset="night_ir",
            steps=[{"id": "deinterlace", "params": {"mode": "send_frame"}}],
            acquisition={"recorderMake": "HiLook"},
        )
    )

    assert result["jobId"] == "v7" and result["family"] == "video" and result["cctv"]["task"] == "roi_fusion"
    assert sent == [
        {
            "token": "tok123",
            "task": "roi_fusion",
            "preset": "night_ir",
            "steps": [{"id": "deinterlace", "params": {"mode": "send_frame"}}],
            "roi": {
                "firstFrame": 10,
                "lastFrame": 40,
                "referenceFrame": 22,
                "box": [100, 80, 64, 24],
                "kind": "plate",
                "scale": 3,
                "method": "trimmed_mean",
            },
            "acquisition": {"recorderMake": "HiLook"},
        }
    ]


async def test_cctv_roi_fuse_without_reference_asks_the_server_for_one(
    monkeypatch: pytest.MonkeyPatch, server_mode: None
) -> None:
    sent: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, json.loads(request.read())))
        if request.url.path.endswith("/roi/reference"):
            return httpx.Response(200, json={"referenceFrame": 7})
        return httpx.Response(202, json={"jobId": "v7", "status": "queued"})

    install_mock(monkeypatch, handler)
    await upflow_cctv_roi_fuse(
        "tok123", frames=[4, 9], box=[0, 0, 40, 40], kind="plate", steps=[{"id": "deblock", "params": {}}]
    )

    (reference_path, reference_body), (job_path, job_body) = sent
    assert reference_path == "/api/v1/video/cctv/tok123/roi/reference"
    assert reference_body == {
        "firstFrame": 4, "lastFrame": 9, "box": [0, 0, 40, 40], "steps": [{"id": "deblock", "params": {}}],
    }  # fmt: skip
    assert job_path == "/api/v1/video/cctv/jobs" and job_body["roi"]["referenceFrame"] == 7


async def test_cctv_roi_fuse_defaults_to_2x_median_without_prefilters(
    monkeypatch: pytest.MonkeyPatch, server_mode: None
) -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return httpx.Response(202, json={"jobId": "v7", "status": "queued"})

    install_mock(monkeypatch, handler)
    await upflow_cctv_roi_fuse("tok123", frames=[0, 5], reference=0, box=[0, 0, 40, 40], kind="face_or_object")

    body = sent[0]
    assert body["steps"] == [] and body["preset"] is None
    assert (body["roi"]["scale"], body["roi"]["method"]) == (2, "median")


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"frames": [10]}, "frames"),
        ({"box": [0, 0, 40]}, "box"),
        ({"kind": "car"}, "Region type"),
        ({"scale": 5}, "Region type"),
        ({"reference": 41}, "reference"),
        ({"preset": "day"}, "presetSteps"),
        ({"steps": ["deblock"]}, "'id'"),
    ],
)
async def test_cctv_roi_fuse_rejects_bad_choices_before_calling_the_api(
    monkeypatch: pytest.MonkeyPatch, server_mode: None, kwargs: dict, fragment: str
) -> None:
    install_mock(monkeypatch, lambda request: pytest.fail("the API must not be called"))
    choices = {"frames": [10, 40], "reference": 22, "box": [0, 0, 40, 40], "kind": "plate", **kwargs}

    result = json.loads(await upflow_cctv_roi_fuse("tok123", **choices))

    assert result["ok"] is False and result["code"] == 2 and fragment in result["error"]


async def test_cctv_roi_fuse_propagates_the_keyed_api_400(monkeypatch: pytest.MonkeyPatch, server_mode: None) -> None:
    detail = {"key": "cctv.error.roiOdd", "reason": "The region must have an even width and height."}
    install_mock(monkeypatch, lambda request: httpx.Response(400, json={"detail": detail}))

    result = await upflow_cctv_roi_fuse("tok123", frames=[0, 5], reference=2, box=[0, 0, 41, 40], kind="plate")

    assert result.startswith("Error") and "cctv.error.roiOdd" in result


async def test_cctv_roi_fuse_falls_back_in_process_when_the_server_is_down(
    monkeypatch: pytest.MonkeyPatch, server_mode: None
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("rechazado")

    seen: list[tuple] = []

    async def inprocess(token, choices, destination_dir=""):
        seen.append((token, choices.roi.kind, choices.roi.reference_frame, destination_dir))
        return json.dumps({"ok": True, "jobId": "inline"})

    install_mock(monkeypatch, handler)
    monkeypatch.setattr(headless_tools, "upflow_cctv_roi_fuse_headless", inprocess)
    result = json.loads(
        await upflow_cctv_roi_fuse(
            "tok123", frames=[0, 5], reference=3, box=[0, 0, 40, 40], kind="plate", destination_dir="C:/caso"
        )
    )

    assert result["jobId"] == "inline" and seen == [("tok123", "plate", 3, "C:/caso")]


async def test_cctv_roi_fuse_runs_in_process_without_calling_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(headless_tools.MODE_ENV, "inprocess")
    install_mock(monkeypatch, lambda request: pytest.fail("in-process mode never calls the API"))
    seen: list[tuple] = []

    async def fake_roi(ctx, token, choices, out_dir=None):
        seen.append((ctx, token, choices.roi.first_frame, out_dir))
        return {"ok": True, "jobId": "inline", "roi": {"framesUsed": 5}}

    monkeypatch.setattr(headless_tools, "get_context", lambda: "ctx")
    monkeypatch.setattr(headless_tools.headless, "cctv_roi", fake_roi)
    result = json.loads(
        await upflow_cctv_roi_fuse(
            "tok123", frames=[4, 9], reference=5, box=[0, 0, 40, 40], kind="plate", destination_dir="C:/caso"
        )
    )

    assert result["roi"] == {"framesUsed": 5} and seen == [("ctx", "tok123", 4, Path("C:/caso"))]
