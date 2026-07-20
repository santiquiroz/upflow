# Selección de pistas de audio, subtítulos y restauración calidad-first — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dejar de descartar pistas de audio/subtítulos al reescalar video, y que Apollo/AudioSR restauren en estéreo/surround real (M/S + RMS-match) en vez de downmixear a mono — más un selector de formato de salida amigable en video y en el módulo Audio.

**Architecture:** Fase A añade un endpoint de análisis (`ffprobe`, ya existe como `MediaTools.ffprobe_json`) + selección de pistas por índice + mapeo múltiple de audio/subtítulos en el mux final, con auto-upgrade de contenedor a `.mkv` cuando hace falta preservar algo sin pérdida. Fase B reemplaza el downmix-a-mono de ambos motores de restauración por un wrapper M/S (mid-side) reutilizable, con RMS-matching post-restauración. Fase C expone el `output_container`/codec de audio ya existente como una elección explícita con copy amigable, tanto en jobs de video como en el módulo Audio standalone.

**Tech Stack:** FastAPI, ffmpeg/ffprobe (ya vendored), numpy, soundfile, React/TS (Vite), vitest, pytest.

## Global Constraints

- Todos los campos nuevos son opt-in con default = comportamiento actual (`audio_track_indices=None`, `keep_subtitles=False`, `audio_output_format="auto"`). Ningún job existente cambia de comportamiento sin que el usuario elija algo nuevo.
- Enhance/restore de audio se aplica SIEMPRE a una sola pista "primaria" (la default detectada, o la primera de `audio_track_indices` si no hay default) — nunca se multiplica el costo de un modelo de difusión por N pistas.
- Auto-upgrade de contenedor (`mp4`→`mkv`) nunca es silencioso: siempre se anota en `job.metadata["containerUpgradedReason"]`.
- `restore_multichannel`'s `restore_mono` callable siempre recibe/devuelve arrays 1D `(samples,)` float32 — el contrato mono de `AudioSrRestorer`/`ApolloRestorer` no cambia, solo quién lo llama.
- Layouts de canal desconocidos caen a mono con `logger.warning` + nota en `job.metadata["multichannelFallback"]` — nunca degradan en silencio.
- Commits en español, formato por capas del repo, sin Co-Authored-By. Sin pushear hasta el final (branch de feature).
- Suites completas verdes al cierre: backend (`.venv/Scripts/python.exe -m pytest -q`) y frontend (`cd frontend && npm test -- --run`).

---

## Fase A — Análisis de pistas + selección + subtítulos

### Task 1: parser de streams + endpoint `POST /video/analyze`

**Files:**
- Create: `app/services/stream_analysis.py`
- Test: `tests/test_stream_analysis.py`
- Modify: `app/schemas.py` (agregar `AudioTrackInfo`, `SubtitleTrackInfo`, `AnalyzeVideoResponse`)
- Modify: `app/api/routes.py` (agregar `POST /video/analyze`)
- Test: `tests/test_routes_video_analyze.py`

**Interfaces:**
- Produces: `parse_audio_tracks(probe: dict) -> list[AudioTrackInfo]`, `parse_subtitle_tracks(probe: dict) -> list[SubtitleTrackInfo]` — funciones puras, consumen el mismo dict que devuelve `MediaTools.ffprobe_json` (ya usado en `video_job_manager.py::_validate_video`).

- [ ] **Step 1: Write the failing test para `parse_audio_tracks`/`parse_subtitle_tracks`**

```python
# tests/test_stream_analysis.py
from app.services.stream_analysis import parse_audio_tracks, parse_subtitle_tracks

FAKE_PROBE = {
    "streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "ac3",
            "channels": 2,
            "disposition": {"default": 1},
            "tags": {"language": "jpn"},
        },
        {
            "index": 2,
            "codec_type": "audio",
            "codec_name": "aac",
            "channels": 6,
            "disposition": {"default": 0},
            "tags": {"language": "eng"},
        },
        {
            "index": 3,
            "codec_type": "subtitle",
            "codec_name": "ass",
            "disposition": {"default": 0},
            "tags": {"language": "eng"},
        },
    ]
}


def test_parse_audio_tracks_returns_one_entry_per_audio_stream_in_order():
    tracks = parse_audio_tracks(FAKE_PROBE)
    assert [t.index for t in tracks] == [1, 2]


def test_parse_audio_tracks_reads_language_channels_and_default_flag():
    tracks = parse_audio_tracks(FAKE_PROBE)
    assert tracks[0].language == "jpn"
    assert tracks[0].channels == 2
    assert tracks[0].is_default is True
    assert tracks[1].language == "eng"
    assert tracks[1].channels == 6
    assert tracks[1].is_default is False


def test_parse_audio_tracks_missing_language_tag_is_none():
    probe = {"streams": [{"index": 0, "codec_type": "audio", "codec_name": "aac", "channels": 2, "disposition": {}}]}
    tracks = parse_audio_tracks(probe)
    assert tracks[0].language is None


def test_parse_audio_tracks_no_audio_streams_returns_empty_list():
    probe = {"streams": [{"index": 0, "codec_type": "video", "codec_name": "h264"}]}
    assert parse_audio_tracks(probe) == []


def test_parse_subtitle_tracks_returns_one_entry_per_subtitle_stream():
    tracks = parse_subtitle_tracks(FAKE_PROBE)
    assert len(tracks) == 1
    assert tracks[0].index == 3
    assert tracks[0].language == "eng"
    assert tracks[0].codec == "ass"


def test_parse_subtitle_tracks_no_subtitle_streams_returns_empty_list():
    assert parse_subtitle_tracks({"streams": []}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stream_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.stream_analysis'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/stream_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AudioTrackInfo:
    index: int
    codec: str
    channels: int
    is_default: bool
    language: str | None


@dataclass(frozen=True)
class SubtitleTrackInfo:
    index: int
    codec: str
    language: str | None


def _language_tag(stream: dict[str, Any]) -> str | None:
    return stream.get("tags", {}).get("language")


def _is_default(stream: dict[str, Any]) -> bool:
    return bool(stream.get("disposition", {}).get("default"))


def parse_audio_tracks(probe: dict[str, Any]) -> list[AudioTrackInfo]:
    return [
        AudioTrackInfo(
            index=stream["index"],
            codec=stream.get("codec_name", "unknown"),
            channels=int(stream.get("channels", 1)),
            is_default=_is_default(stream),
            language=_language_tag(stream),
        )
        for stream in probe.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]


def parse_subtitle_tracks(probe: dict[str, Any]) -> list[SubtitleTrackInfo]:
    return [
        SubtitleTrackInfo(
            index=stream["index"],
            codec=stream.get("codec_name", "unknown"),
            language=_language_tag(stream),
        )
        for stream in probe.get("streams", [])
        if stream.get("codec_type") == "subtitle"
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_stream_analysis.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/stream_analysis.py tests/test_stream_analysis.py
git commit -m "Dominio: parser de pistas de audio/subtitulos desde ffprobe (Fase A Task 1)"
```

- [ ] **Step 6: Write the failing test para el endpoint**

Leer primero cómo se testea el endpoint `/video/jobs` existente (usa el mismo `TestClient`/fixtures) — buscar `tests/test_routes_video.py` o equivalente y copiar su fixture de app/cliente. El endpoint sube un archivo real chico (usar cualquier fixture de video ya usado en otros tests, p.ej. `tests/fixtures/tiny.mp4` si existe, o el mismo patrón de creación de video sintético que usan los tests de `video_job_manager`).

```python
# tests/test_routes_video_analyze.py
import io

from app.services.stream_analysis import AudioTrackInfo, SubtitleTrackInfo


def test_analyze_video_returns_audio_and_subtitle_tracks(client, monkeypatch, tiny_video_bytes):
    async def fake_ffprobe_json(self, path):
        return {
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264"},
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "disposition": {"default": 1},
                    "tags": {"language": "jpn"},
                },
            ]
        }

    from app.services.media_tools import MediaTools

    monkeypatch.setattr(MediaTools, "ffprobe_json", fake_ffprobe_json)

    response = client.post(
        "/api/v1/video/analyze",
        files={"file": ("clip.mp4", io.BytesIO(tiny_video_bytes), "video/mp4")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "uploadToken" in body
    assert body["audioTracks"] == [
        {"index": 1, "codec": "aac", "channels": 2, "isDefault": True, "language": "jpn"}
    ]
    assert body["subtitleTracks"] == []


def test_analyze_video_rejects_non_video_upload(client, monkeypatch):
    from app.services.media_tools import MediaTools
    import subprocess

    async def fake_ffprobe_json_raises(self, path):
        raise subprocess.CalledProcessError(1, ["ffprobe"])

    monkeypatch.setattr(MediaTools, "ffprobe_json", fake_ffprobe_json_raises)

    response = client.post(
        "/api/v1/video/analyze",
        files={"file": ("not-a-video.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
```

Nota: `client`, `tiny_video_bytes` deben ser los mismos fixtures que ya usan `tests/test_routes_video.py` (o el archivo de tests de rutas de video que exista) — reusar, no reinventar.

- [ ] **Step 7: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_routes_video_analyze.py -v`
Expected: FAIL with 404 (route doesn't exist yet)

- [ ] **Step 8: Add schemas**

En `app/schemas.py`, agregar (junto a los otros schemas de response, siguiendo el mismo patrón de `serialization_alias` camelCase ya usado en el archivo):

```python
class AudioTrackResponse(BaseModel):
    index: int
    codec: str
    channels: int
    is_default: bool = Field(serialization_alias="isDefault")
    language: str | None = None


class SubtitleTrackResponse(BaseModel):
    index: int
    codec: str
    language: str | None = None


class AnalyzeVideoResponse(BaseModel):
    upload_token: str = Field(serialization_alias="uploadToken")
    audio_tracks: list[AudioTrackResponse] = Field(serialization_alias="audioTracks")
    subtitle_tracks: list[SubtitleTrackResponse] = Field(serialization_alias="subtitleTracks")

    model_config = {"populate_by_name": True}
```

- [ ] **Step 9: Add the route**

En `app/api/routes.py`, cerca de `create_video_job`, agregar (reusa `sanitize_filename`, `uuid4`, `settings.uploads_path` exactamente como hace `create_video_job` — ver líneas 438-441 de este mismo archivo para el patrón exacto de guardado con token):

```python
@router.post("/video/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    media_tools: MediaTools = Depends(get_media_tools),
) -> AnalyzeVideoResponse:
    original_name = Path(file.filename or "upload.mp4").name
    safe_name = sanitize_filename(original_name, default="upload.mp4")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    await storage.save_upload(file, destination)

    try:
        probe = await media_tools.ffprobe_json(destination)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid video") from exc

    audio_tracks = parse_audio_tracks(probe)
    subtitle_tracks = parse_subtitle_tracks(probe)
    return AnalyzeVideoResponse(
        upload_token=token,
        audio_tracks=[
            AudioTrackResponse(
                index=t.index, codec=t.codec, channels=t.channels, is_default=t.is_default, language=t.language
            )
            for t in audio_tracks
        ],
        subtitle_tracks=[
            SubtitleTrackResponse(index=t.index, codec=t.codec, language=t.language) for t in subtitle_tracks
        ],
    )
```

Agregar los imports que falten al tope del archivo: `from app.services.stream_analysis import parse_audio_tracks, parse_subtitle_tracks`, `from app.services.media_tools import MediaTools`, `get_media_tools` (buscar cómo se inyecta `MediaTools` en `create_video_job` o en `app.main` — seguir el mismo patrón `Depends(...)` que ya exista para ese servicio; si no hay un `get_media_tools` en `app/api/dependencies.py`, agregarlo siguiendo el patrón de `get_video_job_manager`/`get_storage` del mismo archivo).

- [ ] **Step 10: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_routes_video_analyze.py -v`
Expected: 2 passed

- [ ] **Step 11: Run the focused suite + full backend suite once**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: todos los tests previos (924+) + los 8 nuevos de este task, 0 failures.

- [ ] **Step 12: Commit**

```bash
git add app/schemas.py app/api/routes.py tests/test_routes_video_analyze.py
git commit -m "Aplicacion: endpoint POST /video/analyze (Fase A Task 1)"
```

---

### Task 2: `upload_token` + `audio_track_indices` + `keep_subtitles` en el modelo de job

**Files:**
- Modify: `app/models.py:53-93` (`VideoUpscaleJob`)
- Modify: `app/services/video_job_manager.py` (`create_job`, resolución de `upload_token`)
- Modify: `app/api/routes.py` (`create_video_job`)
- Modify: `app/schemas.py` (request/response)
- Test: `tests/test_video_job_manager.py`

**Interfaces:**
- Consumes: `AnalyzeVideoResponse.upload_token` (Task 1) — el frontend lo reenvía en `POST /video/jobs`.
- Produces: `VideoUpscaleJob.audio_track_indices: list[int] | None`, `VideoUpscaleJob.keep_subtitles: bool` — consumidos por Task 3.

- [ ] **Step 1: Write the failing test para resolución de `upload_token`**

Ubicar el archivo de tests de `VideoJobManager.create_job` existente (`tests/test_video_job_manager.py`) y revisar su fixture de `manager`/`tmp_path` antes de escribir — seguir exactamente ese patrón de construcción del manager con dobles fake.

```python
# agregar a tests/test_video_job_manager.py
import pytest


@pytest.mark.asyncio
async def test_create_job_resolves_source_path_from_upload_token(manager, tmp_path):
    staged = manager.settings.uploads_path / "abc123-clip.mp4"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"fake-mp4-bytes")

    job = await manager.create_job(
        source_path=None,
        upload_token="abc123",
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=True,
    )
    assert job.source_path == staged


@pytest.mark.asyncio
async def test_create_job_raises_when_upload_token_has_no_staged_file(manager):
    with pytest.raises(ValueError, match="upload_token"):
        await manager.create_job(
            source_path=None,
            upload_token="does-not-exist",
            original_filename="clip.mp4",
            model_name="realesr-animevideov3-x2",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=True,
        )


@pytest.mark.asyncio
async def test_create_job_defaults_audio_track_indices_and_keep_subtitles(manager):
    job = await manager.create_job(
        source_path=manager.settings.uploads_path / "existing.mp4",
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=True,
    )
    assert job.audio_track_indices is None
    assert job.keep_subtitles is False


@pytest.mark.asyncio
async def test_create_job_upgrades_container_to_mkv_when_keep_subtitles(manager):
    job = await manager.create_job(
        source_path=manager.settings.uploads_path / "existing.mp4",
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=True,
        keep_subtitles=True,
    )
    assert job.output_container == "mkv"
    assert "subtitles" in job.metadata["containerUpgradedReason"]
```

(Nota: `existing.mp4` en los últimos dos tests debe ya existir como archivo real chico en el `tmp_path` del fixture del manager — revisar cómo los tests existentes de `create_job` proveen un `source_path` válido y reusar ese mismo helper/fixture.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_job_manager.py -k upload_token or audio_track_indices or keep_subtitles -v`
Expected: FAIL (`create_job() got an unexpected keyword argument 'upload_token'`)

- [ ] **Step 3: `VideoUpscaleJob` — nuevos campos**

En `app/models.py`, agregar después de `audio_restore: str | None = None` (línea 67):

```python
    audio_track_indices: list[int] | None = None
    keep_subtitles: bool = False
```

- [ ] **Step 4: `create_job` — resolver `upload_token`, aplicar auto-upgrade de contenedor**

En `app/services/video_job_manager.py`, modificar la firma de `create_job` (agregar tras `source_path: Path`, y hacerlo opcional ya que ahora puede venir por token):

```python
    async def create_job(
        self,
        *,
        source_path: Path | None = None,
        upload_token: str | None = None,
        original_filename: str,
        model_name: str,
        scale: int,
        output_container: str,
        video_codec: str,
        video_preset: str,
        crf: int,
        keep_audio: bool,
        fps_multiplier: int = 1,
        target_fps: str | None = None,
        audio_enhance: str | None = None,
        audio_restore: str | None = None,
        audio_track_indices: list[int] | None = None,
        keep_subtitles: bool = False,
        interp_engine: str = RIFE_ENGINE,
        model_id: str | None = None,
        device: str | None = None,
        backend: str | None = None,
        video_encoder: str = "auto",
        job_id: str | None = None,
    ) -> VideoUpscaleJob:
        resolved_source_path = self._resolve_source_path(source_path, upload_token)
        source_fps, probe = await self._validate_video(resolved_source_path)
```

Reemplazar cada uso posterior de `source_path` dentro del método (en `self._validate_video(source_path)` y en la construcción de `VideoUpscaleJob(source_path=source_path, ...)`) por `resolved_source_path`.

Justo antes de construir el `VideoUpscaleJob`, resolver el contenedor y las notas de metadata:

```python
        resolved_container, container_upgrade_reason = self._resolve_output_container(
            output_container, keep_subtitles
        )
```

Y en la construcción de `VideoUpscaleJob(...)`, usar `output_container=resolved_container`, agregar `audio_track_indices=audio_track_indices, keep_subtitles=keep_subtitles,`. Después de construir `job` (antes de `self._enqueue(job)`), agregar:

```python
        if container_upgrade_reason is not None:
            job.metadata["containerUpgradedReason"] = container_upgrade_reason
```

Nuevos métodos privados (agregar cerca de `_validate_video`):

```python
    def _resolve_source_path(self, source_path: Path | None, upload_token: str | None) -> Path:
        if upload_token is not None:
            matches = sorted(self.settings.uploads_path.glob(f"{upload_token}-*"))
            if not matches:
                raise ValueError(f"No staged upload found for upload_token={upload_token!r}")
            return matches[0]
        if source_path is None:
            raise ValueError("Either source_path or upload_token must be provided")
        return source_path

    @staticmethod
    def _resolve_output_container(output_container: str, keep_subtitles: bool) -> tuple[str, str | None]:
        if keep_subtitles and output_container != "mkv":
            return "mkv", "Output container upgraded to mkv to preserve subtitles without quality loss"
        return output_container, None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_job_manager.py -k upload_token or audio_track_indices or keep_subtitles -v`
Expected: 4 passed

- [ ] **Step 6: Wire the route (`create_video_job`)**

En `app/api/routes.py`, en la firma de `create_video_job` agregar:

```python
    upload_token: str | None = Form(default=None),
    audio_track_indices: str | None = Form(default=None),  # CSV de indices, p.ej. "1,2"
    keep_subtitles: bool = Form(default=False),
```

Validar que exactamente uno de `file`/`upload_token` venga (después del chequeo de `profile`):

```python
    if bool(file and file.filename) == bool(upload_token):
        raise HTTPException(status_code=400, detail="Provide exactly one of file or upload_token")
```

Parsear `audio_track_indices` (CSV → `list[int] | None`):

```python
    parsed_audio_track_indices = (
        [int(i) for i in audio_track_indices.split(",") if i.strip()] if audio_track_indices else None
    )
```

Cuando venga `upload_token`, saltar el guardado del archivo (el bloque que arma `destination`/escribe el `file`) — solo pasar `upload_token=upload_token, source_path=None` a `video_jobs.create_job(...)` en vez de `source_path=destination`. Cuando venga `file`, seguir el flujo actual sin cambios (`source_path=destination, upload_token=None`). Agregar `audio_track_indices=parsed_audio_track_indices, keep_subtitles=keep_subtitles,` a la llamada de `create_job`.

- [ ] **Step 7: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures.

- [ ] **Step 8: Commit**

```bash
git add app/models.py app/services/video_job_manager.py app/api/routes.py tests/test_video_job_manager.py
git commit -m "Dominio+Aplicacion: upload_token, audio_track_indices, keep_subtitles + auto-upgrade a mkv (Fase A Task 2)"
```

---

### Task 3: mapeo múltiple de audio + subtítulos en el mux final

**Files:**
- Modify: `app/services/video_upscaler.py` (`_build_encode_command`, `_prepare_audio` y alrededores)
- Test: `tests/test_video_upscaler.py`

**Interfaces:**
- Consumes: `job.audio_track_indices: list[int] | None`, `job.keep_subtitles: bool` (Task 2), `job.probe: dict` (ya existe).
- Produces: `_build_encode_command` ahora acepta `job` completo (ya lo hace) y arma `-map`/`-i` adicionales para pistas extra + subtítulos.

**Diseño de mapeo**: la pista PRIMARIA (enhance/restore aplicado o copy simple) sigue viniendo de `audio_mux_path` (input index 1, sin cambios). Cuando hay pistas de audio EXTRA (`audio_track_indices` con más de un elemento) o `keep_subtitles=True`, se agrega el `job.source_path` ORIGINAL como un input adicional (`-i <source_path>`) y se mapean sus streams de audio/subtítulos directamente con `-c:s copy`/`-c:a:N copy` sin volver a extraerlos.

- [ ] **Step 1: Write the failing test**

Revisar primero `tests/test_video_upscaler.py` para el patrón exacto de test de `_build_encode_command` (probablemente ya hay tests para el caso con/sin audio — copiar su estilo de construcción de `job`/`self` fake).

```python
# agregar a tests/test_video_upscaler.py
def test_build_encode_command_maps_extra_audio_tracks_from_source(upscaler, tmp_path):
    job = make_video_job(  # usar el mismo helper que ya usan los tests existentes de este archivo
        audio_track_indices=[1, 2],
        keep_subtitles=False,
        source_path=tmp_path / "source.mkv",
    )
    audio_mux_path = tmp_path / "audio.m4a"
    audio_mux_path.write_bytes(b"fake")
    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", audio_mux_path, ["-c:a", "copy"], tmp_path / "out.mkv", "libx264"
    )
    assert str(job.source_path) in cmd
    assert "-map" in cmd
    idx = cmd.index(str(job.source_path))
    assert cmd[idx - 1] == "-i"
    # la pista extra (index=2 en el archivo original) se mapea explicitamente
    assert any(arg.startswith("2:a:") or arg == "2:1" for arg in cmd)


def test_build_encode_command_maps_subtitles_when_keep_subtitles(upscaler, tmp_path):
    job = make_video_job(
        audio_track_indices=None,
        keep_subtitles=True,
        source_path=tmp_path / "source.mkv",
    )
    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", None, [], tmp_path / "out.mkv", "libx264"
    )
    assert "-c:s" in cmd
    assert "copy" in cmd
    assert str(job.source_path) in cmd


def test_build_encode_command_no_extra_tracks_unchanged(upscaler, tmp_path):
    job = make_video_job(audio_track_indices=None, keep_subtitles=False, source_path=tmp_path / "source.mp4")
    audio_mux_path = tmp_path / "audio.m4a"
    audio_mux_path.write_bytes(b"fake")
    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", audio_mux_path, ["-c:a", "copy"], tmp_path / "out.mp4", "libx264"
    )
    assert str(job.source_path) not in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_upscaler.py -k extra_audio or keep_subtitles or no_extra_tracks -v`
Expected: FAIL (asserts don't hold — `_build_encode_command` doesn't reference `job.source_path` yet, `make_video_job` helper probably needs the two new kwargs too — si `make_video_job` no acepta `audio_track_indices`/`keep_subtitles`, agregarlos a ese helper primero con default `None`/`False`).

- [ ] **Step 3: Implement**

En `app/services/video_upscaler.py`, modificar `_build_encode_command` (líneas 659-683):

```python
    def _build_encode_command(
        self,
        job: VideoUpscaleJob,
        encode_frames_dir: Path,
        encode_fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
        output_path: Path,
        encoder: str,
    ) -> list[str]:
        cmd = [
            str(self.settings.ffmpeg_binary_path),
            "-y",
            "-framerate",
            encode_fps,
            "-i",
            str(encode_frames_dir / "%08d.png"),
        ]
        next_input_index = 1
        if audio_mux_path is not None:
            cmd += ["-i", str(audio_mux_path), "-map", "0:v:0", f"-map", f"{next_input_index}:a:0"]
            next_input_index += 1
        source_input_index = self._maybe_add_source_input(cmd, job, next_input_index)
        if source_input_index is not None:
            self._map_extra_audio_tracks(cmd, job, source_input_index)
            self._map_subtitles(cmd, job, source_input_index)
        cmd += self._build_video_encode_options(job, encoder)
        if audio_mux_path is not None:
            cmd += audio_codec_args
        if job.keep_subtitles:
            cmd += ["-c:s", "copy"]
        cmd.append(str(output_path))
        return cmd

    def _extra_audio_track_indices(self, job: VideoUpscaleJob) -> list[int]:
        if not job.audio_track_indices or len(job.audio_track_indices) <= 1:
            return []
        primary = job.audio_track_indices[0]
        return [i for i in job.audio_track_indices[1:] if i != primary]

    def _maybe_add_source_input(self, cmd: list[str], job: VideoUpscaleJob, input_index: int) -> int | None:
        if not self._extra_audio_track_indices(job) and not job.keep_subtitles:
            return None
        cmd += ["-i", str(job.source_path)]
        return input_index

    def _map_extra_audio_tracks(self, cmd: list[str], job: VideoUpscaleJob, source_input_index: int) -> None:
        for stream_index in self._extra_audio_track_indices(job):
            cmd += ["-map", f"{source_input_index}:{stream_index}"]

    def _map_subtitles(self, cmd: list[str], job: VideoUpscaleJob, source_input_index: int) -> None:
        if not job.keep_subtitles:
            return
        cmd += ["-map", f"{source_input_index}:s?"]
```

Nota sobre `_map_extra_audio_tracks`: usa el índice de stream ABSOLUTO del archivo original (`f"{source_input_index}:{stream_index}"`, p.ej. `2:2` para el stream índice 2 del input 2) en vez de la sintaxis relativa `2:a:N` — es más simple y evita tener que re-enumerar cuál "audio Nº" es cada índice absoluto. ffmpeg soporta mapear por índice absoluto de stream directamente.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_upscaler.py -k extra_audio or keep_subtitles or no_extra_tracks -v`
Expected: 3 passed (ajustar los asserts del Step 1 si la sintaxis exacta de mapeo elegida en Step 3 difiere — el índice absoluto `f"{source_input_index}:{stream_index}"` es la fuente de verdad, no la primera versión especulativa del test).

- [ ] **Step 5: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add app/services/video_upscaler.py tests/test_video_upscaler.py
git commit -m "Aplicacion: mapea pistas de audio extra y subtitulos desde el archivo original al mux final (Fase A Task 3)"
```

---

### Task 4: frontend — análisis + selección de pistas/subtítulos

**Files:**
- Modify: `frontend/src/lib/api.ts` (nuevo `analyzeVideo`, extender `CreateVideoJobParams`/`buildVideoJobFormData`)
- Modify: `frontend/src/lib/apiTypes.ts` (tipos de respuesta de analyze)
- Create: `frontend/src/modules/enhance/TrackSelector.tsx`
- Create: `frontend/src/modules/enhance/TrackSelector.test.tsx`
- Modify: `frontend/src/modules/enhance/VideoPanel.tsx`
- Modify: `frontend/src/modules/enhance/VideoPanel.test.tsx`

**Interfaces:**
- Consumes: `POST /video/analyze` (Task 1), `upload_token`/`audio_track_indices`/`keep_subtitles` en `POST /video/jobs` (Task 2).
- Produces: `TrackSelector` — componente puro `{ audioTracks, subtitleTracks, selectedAudioIndices, onChangeAudioIndices, keepSubtitles, onChangeKeepSubtitles }`.

- [ ] **Step 1: tipos + llamada de API**

En `frontend/src/lib/apiTypes.ts`, agregar:

```typescript
export interface AudioTrackInfo {
  index: number;
  codec: string;
  channels: number;
  isDefault: boolean;
  language: string | null;
}

export interface SubtitleTrackInfo {
  index: number;
  codec: string;
  language: string | null;
}

export interface AnalyzeVideoResponse {
  uploadToken: string;
  audioTracks: AudioTrackInfo[];
  subtitleTracks: SubtitleTrackInfo[];
}
```

En `frontend/src/lib/api.ts`, agregar (siguiendo el patrón de `apiPostForm` ya usado por `createVideoJob`):

```typescript
export function analyzeVideo(file: File): Promise<AnalyzeVideoResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return apiPostForm<AnalyzeVideoResponse>("/video/analyze", formData);
}
```

Extender `CreateVideoJobParams` (agregar campos, todos opcionales para no romper llamadas existentes):

```typescript
  uploadToken?: string;
  audioTrackIndices?: number[];
  keepSubtitles?: boolean;
```

Extender `buildVideoJobFormData`: cuando `params.uploadToken` esté presente, usar `formData.append("upload_token", params.uploadToken)` EN VEZ DE `formData.append("file", params.file)` (no ambos — el backend rechaza si vienen los dos). Agregar:

```typescript
  if (params.audioTrackIndices && params.audioTrackIndices.length > 0) {
    formData.append("audio_track_indices", params.audioTrackIndices.join(","));
  }
  if (params.keepSubtitles) {
    formData.append("keep_subtitles", "true");
  }
```

- [ ] **Step 2: `TrackSelector` — test primero**

```tsx
// frontend/src/modules/enhance/TrackSelector.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TrackSelector } from "./TrackSelector";

const audioTracks = [
  { index: 1, codec: "ac3", channels: 2, isDefault: true, language: "jpn" },
  { index: 2, codec: "aac", channels: 6, isDefault: false, language: "eng" },
];
const subtitleTracks = [{ index: 3, codec: "ass", language: "eng" }];

describe("TrackSelector", () => {
  it("shows one checkbox per audio track with language and default badge", () => {
    render(
      <TrackSelector
        audioTracks={audioTracks}
        subtitleTracks={subtitleTracks}
        selectedAudioIndices={[1]}
        onChangeAudioIndices={vi.fn()}
        keepSubtitles={false}
        onChangeKeepSubtitles={vi.fn()}
      />
    );
    expect(screen.getByLabelText(/jpn/i)).toBeChecked();
    expect(screen.getByLabelText(/eng/i)).not.toBeChecked();
    expect(screen.getByText(/default/i)).toBeInTheDocument();
  });

  it("calls onChangeAudioIndices with the toggled track added", () => {
    const onChange = vi.fn();
    render(
      <TrackSelector
        audioTracks={audioTracks}
        subtitleTracks={subtitleTracks}
        selectedAudioIndices={[1]}
        onChangeAudioIndices={onChange}
        keepSubtitles={false}
        onChangeKeepSubtitles={vi.fn()}
      />
    );
    fireEvent.click(screen.getByLabelText(/eng/i));
    expect(onChange).toHaveBeenCalledWith([1, 2]);
  });

  it("calls onChangeAudioIndices with the toggled track removed", () => {
    const onChange = vi.fn();
    render(
      <TrackSelector
        audioTracks={audioTracks}
        subtitleTracks={subtitleTracks}
        selectedAudioIndices={[1, 2]}
        onChangeAudioIndices={onChange}
        keepSubtitles={false}
        onChangeKeepSubtitles={vi.fn()}
      />
    );
    fireEvent.click(screen.getByLabelText(/jpn/i));
    expect(onChange).toHaveBeenCalledWith([2]);
  });

  it("renders nothing for subtitles when there are no subtitle tracks", () => {
    render(
      <TrackSelector
        audioTracks={audioTracks}
        subtitleTracks={[]}
        selectedAudioIndices={[1]}
        onChangeAudioIndices={vi.fn()}
        keepSubtitles={false}
        onChangeKeepSubtitles={vi.fn()}
      />
    );
    expect(screen.queryByLabelText(/subtitle/i)).not.toBeInTheDocument();
  });

  it("toggles keepSubtitles when the subtitle checkbox is present and clicked", () => {
    const onChange = vi.fn();
    render(
      <TrackSelector
        audioTracks={audioTracks}
        subtitleTracks={subtitleTracks}
        selectedAudioIndices={[1]}
        onChangeAudioIndices={vi.fn()}
        keepSubtitles={false}
        onChangeKeepSubtitles={onChange}
      />
    );
    fireEvent.click(screen.getByLabelText(/subtitle/i));
    expect(onChange).toHaveBeenCalledWith(true);
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd frontend && npm test -- --run TrackSelector`
Expected: FAIL (module doesn't exist)

- [ ] **Step 4: Implement `TrackSelector`**

Revisar primero `frontend/src/modules/audio/AudioPanel.tsx` para copiar los patrones de estilo/clases Tailwind ya usados en ese módulo (checkboxes, labels, badges) — no inventar clases nuevas.

```tsx
// frontend/src/modules/enhance/TrackSelector.tsx
import type { AudioTrackInfo, SubtitleTrackInfo } from "../../lib/apiTypes";

interface TrackSelectorProps {
  audioTracks: AudioTrackInfo[];
  subtitleTracks: SubtitleTrackInfo[];
  selectedAudioIndices: number[];
  onChangeAudioIndices: (indices: number[]) => void;
  keepSubtitles: boolean;
  onChangeKeepSubtitles: (value: boolean) => void;
}

function trackLabel(track: AudioTrackInfo): string {
  const lang = track.language ?? "unknown";
  const channelLabel = track.channels === 1 ? "mono" : track.channels === 2 ? "stereo" : `${track.channels}ch`;
  return `${lang} (${track.codec}, ${channelLabel})`;
}

export function TrackSelector({
  audioTracks,
  subtitleTracks,
  selectedAudioIndices,
  onChangeAudioIndices,
  keepSubtitles,
  onChangeKeepSubtitles,
}: TrackSelectorProps) {
  function toggleAudio(index: number) {
    if (selectedAudioIndices.includes(index)) {
      onChangeAudioIndices(selectedAudioIndices.filter((i) => i !== index));
    } else {
      onChangeAudioIndices([...selectedAudioIndices, index].sort((a, b) => a - b));
    }
  }

  return (
    <div className="space-y-2">
      {audioTracks.length > 0 && (
        <fieldset className="space-y-1">
          <legend className="text-sm font-medium text-text-secondary">Audio tracks</legend>
          {audioTracks.map((track) => (
            <label key={track.index} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={selectedAudioIndices.includes(track.index)}
                onChange={() => toggleAudio(track.index)}
                aria-label={trackLabel(track)}
              />
              <span>{trackLabel(track)}</span>
              {track.isDefault && (
                <span className="rounded bg-surface-2 px-1.5 py-0.5 text-xs text-text-faint">default</span>
              )}
            </label>
          ))}
        </fieldset>
      )}
      {subtitleTracks.length > 0 && (
        <label className="flex items-center gap-2 text-sm" aria-label="Keep embedded subtitles">
          <input
            type="checkbox"
            checked={keepSubtitles}
            onChange={(e) => onChangeKeepSubtitles(e.target.checked)}
            aria-label="Keep embedded subtitles"
          />
          <span>Keep embedded subtitles ({subtitleTracks.length})</span>
        </label>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd frontend && npm test -- --run TrackSelector`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/lib/apiTypes.ts frontend/src/modules/enhance/TrackSelector.tsx frontend/src/modules/enhance/TrackSelector.test.tsx
git commit -m "Infraestructura: componente TrackSelector + cliente de /video/analyze (Fase A Task 4, parte 1)"
```

- [ ] **Step 7: wire into `VideoPanel.tsx` — test primero**

Leer `VideoPanel.tsx` completo primero (ya lo hicimos durante Task 4.2 de SP14 — buscar dónde se llama `createVideoJob` y dónde vive el estado de `interpEngine` para copiar el mismo patrón de `useState`/reconciliación). Agregar al archivo de test existente `VideoPanel.test.tsx`:

```tsx
// agregar a frontend/src/modules/enhance/VideoPanel.test.tsx
it("calls analyzeVideo when a file is selected, then shows the track selector", async () => {
  vi.mocked(api.analyzeVideo).mockResolvedValueOnce({
    uploadToken: "tok123",
    audioTracks: [{ index: 1, codec: "aac", channels: 2, isDefault: true, language: "jpn" }],
    subtitleTracks: [{ index: 2, codec: "ass", language: "eng" }],
  });
  renderVideoPanel();
  const file = new File(["data"], "clip.mp4", { type: "video/mp4" });
  fireEvent.change(screen.getByLabelText(/upload/i), { target: { files: [file] } });
  await screen.findByText(/jpn/i);
  expect(api.analyzeVideo).toHaveBeenCalledWith(file);
});

it("submits upload_token and selected audio/subtitle choices instead of re-uploading the file", async () => {
  vi.mocked(api.analyzeVideo).mockResolvedValueOnce({
    uploadToken: "tok123",
    audioTracks: [{ index: 1, codec: "aac", channels: 2, isDefault: true, language: "jpn" }],
    subtitleTracks: [{ index: 2, codec: "ass", language: "eng" }],
  });
  vi.mocked(api.createVideoJob).mockResolvedValueOnce({ jobId: "job1" });
  renderVideoPanel();
  const file = new File(["data"], "clip.mp4", { type: "video/mp4" });
  fireEvent.change(screen.getByLabelText(/upload/i), { target: { files: [file] } });
  await screen.findByText(/jpn/i);
  fireEvent.click(screen.getByLabelText(/subtitle/i));
  fireEvent.click(screen.getByRole("button", { name: /submit|enhance|upscale/i }));
  await waitFor(() => expect(api.createVideoJob).toHaveBeenCalled());
  const params = vi.mocked(api.createVideoJob).mock.calls[0][0];
  expect(params.uploadToken).toBe("tok123");
  expect(params.keepSubtitles).toBe(true);
  expect(params.file).toBeUndefined();
});
```

(Ajustar los selectores exactos — `getByLabelText(/upload/i)`, nombre del botón de submit — a lo que el archivo real ya usa; revisar los tests existentes del mismo archivo para copiar los queries reales en vez de adivinar.)

- [ ] **Step 8: Run test to verify it fails**

Run: `cd frontend && npm test -- --run VideoPanel`
Expected: FAIL (no llama a `analyzeVideo` todavía)

- [ ] **Step 9: Implement**

En `VideoPanel.tsx`: agregar estado `analyzeResult: AnalyzeVideoResponse | null`, `selectedAudioIndices: number[]`, `keepSubtitles: boolean`. En el handler de selección de archivo (`onChange` del input de upload), en vez de solo guardar el `File` en estado, llamar a `analyzeVideo(file)`, guardar el resultado, e inicializar `selectedAudioIndices` con el índice de la pista `isDefault` (o `[]` si no hay ninguna marcada default — el backend ya maneja `None`/lista vacía como "comportamiento actual" en Task 2, así que lista vacía en el submit debe traducirse a NO mandar `audio_track_indices` en absoluto, mismo criterio que ya usa `buildVideoJobFormData` para otros campos opcionales). Renderizar `<TrackSelector .../>` cuando `analyzeResult` no es null. En el submit, pasar `uploadToken: analyzeResult?.uploadToken`, `audioTrackIndices: selectedAudioIndices`, `keepSubtitles`, y OMITIR `file` cuando hay `uploadToken` (el llamado a `createVideoJob` debe construir params condicionalmente).

- [ ] **Step 10: Run test to verify it passes**

Run: `cd frontend && npm test -- --run VideoPanel`
Expected: pasan los 2 tests nuevos + los existentes siguen verdes.

- [ ] **Step 11: Run full frontend suite + tsc**

Run: `cd frontend && npm test -- --run && npx tsc --noEmit`
Expected: 0 failures, tsc limpio.

- [ ] **Step 12: Commit**

```bash
git add frontend/src/modules/enhance/VideoPanel.tsx frontend/src/modules/enhance/VideoPanel.test.tsx
git commit -m "Infraestructura: VideoPanel analiza el archivo antes de enviar, deja elegir pistas/subs (Fase A Task 4, parte 2)"
```

---

## Fase B — M/S + RMS-match (Apollo y AudioSR)

### Task 5: `multichannel_restore.py` — mono/estéreo M/S + RMS-match (unidades puras)

**Files:**
- Create: `app/services/engines/multichannel_restore.py`
- Test: `tests/test_multichannel_restore.py`

**Interfaces:**
- Produces: `restore_multichannel(audio: np.ndarray, restore_mono: Callable[[np.ndarray], np.ndarray]) -> np.ndarray` — `audio` shape `(samples, channels)` float32, `restore_mono` firma `(samples,) -> (samples,)` (el contrato exacto de `ApolloRestorer._restore_chunked`/`AudioSrDriver.restore`). Consumido por Task 6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multichannel_restore.py
import numpy as np
import pytest

from app.services.engines.multichannel_restore import restore_multichannel


def _identity(mono: np.ndarray) -> np.ndarray:
    return mono


def test_mono_input_calls_restore_mono_directly():
    audio = np.array([[0.1], [0.2], [-0.3]], dtype=np.float32)
    calls = []

    def spy(mono):
        calls.append(mono.copy())
        return mono * 2

    result = restore_multichannel(audio, spy)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], audio[:, 0])
    np.testing.assert_allclose(result[:, 0], audio[:, 0] * 2)
    assert result.shape == audio.shape


def test_stereo_round_trip_with_identity_restore_reproduces_original():
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.5, 0.5, size=(2000, 2)).astype(np.float32)
    result = restore_multichannel(audio, _identity)
    np.testing.assert_allclose(result, audio, atol=1e-6)


def test_stereo_side_channel_never_passed_to_restore_mono():
    audio = np.zeros((4, 2), dtype=np.float32)
    audio[:, 0] = [1.0, 1.0, 1.0, 1.0]  # L
    audio[:, 1] = [-1.0, -1.0, -1.0, -1.0]  # R -> mid=0, side=1
    calls = []

    def spy(mono):
        calls.append(mono.copy())
        return mono

    restore_multichannel(audio, spy)
    np.testing.assert_allclose(calls[0], np.zeros(4, dtype=np.float32))  # solo el mid (0) llega al modelo


def test_stereo_restoring_mid_changes_only_shared_content():
    audio = np.zeros((4, 2), dtype=np.float32)
    audio[:, 0] = [1.0, 1.0, 1.0, 1.0]
    audio[:, 1] = [1.0, 1.0, 1.0, 1.0]  # side=0, mid=1 (mono content)

    def double_mid(mono):
        return mono * 2

    result = restore_multichannel(audio, double_mid)
    np.testing.assert_allclose(result[:, 0], [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(result[:, 1], [2.0, 2.0, 2.0, 2.0])


def test_rms_matches_original_within_tolerance():
    rng = np.random.default_rng(1)
    audio = (rng.uniform(-0.5, 0.5, size=(4000, 2)) * 0.3).astype(np.float32)

    def amplify(mono):
        return mono * 5.0  # simula un modelo que altera el nivel

    result = restore_multichannel(audio, amplify)
    original_rms = np.sqrt(np.mean(audio**2))
    result_rms = np.sqrt(np.mean(result**2))
    assert result_rms == pytest.approx(original_rms, rel=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_multichannel_restore.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/engines/multichannel_restore.py
from __future__ import annotations

from typing import Callable

import numpy as np

RestoreMonoFn = Callable[[np.ndarray], np.ndarray]


def restore_multichannel(audio: np.ndarray, restore_mono: RestoreMonoFn) -> np.ndarray:
    """audio: (samples, channels) float32. Restaura preservando imagen espacial.

    1 canal: pasa directo por restore_mono.
    2 canales: decodifica Mid/Side, restaura solo Mid, side queda intacto.
    Otros layouts: ver multichannel_layouts.py (Task 7).
    """
    channels = audio.shape[1]
    if channels == 1:
        restored = restore_mono(audio[:, 0])
        return _rms_match(restored, audio[:, 0]).reshape(-1, 1)
    if channels == 2:
        return _restore_stereo_mid_side(audio, restore_mono)
    raise NotImplementedError(f"{channels}-channel audio requires multichannel_layouts (Task 7)")


def _restore_stereo_mid_side(audio: np.ndarray, restore_mono: RestoreMonoFn) -> np.ndarray:
    left, right = audio[:, 0], audio[:, 1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    restored_mid = restore_mono(mid)
    restored_mid = _rms_match(restored_mid, mid)
    left_out = restored_mid + side
    right_out = restored_mid - side
    return np.stack([left_out, right_out], axis=1).astype(np.float32)


def _rms_match(restored: np.ndarray, original: np.ndarray) -> np.ndarray:
    original_rms = np.sqrt(np.mean(original.astype(np.float64) ** 2))
    restored_rms = np.sqrt(np.mean(restored.astype(np.float64) ** 2))
    if restored_rms < 1e-9 or original_rms < 1e-9:
        return restored
    return (restored * (original_rms / restored_rms)).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_multichannel_restore.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/engines/multichannel_restore.py tests/test_multichannel_restore.py
git commit -m "Dominio: restore_multichannel M/S + RMS-match para audio estereo/mono (Fase B Task 5)"
```

---

### Task 6: wire M/S en AudioSR y Apollo

**Files:**
- Modify: `app/services/engines/audiosr_restore.py`
- Modify: `app/services/engines/apollo_restore.py`
- Test: `tests/test_audiosr_restore.py`, `tests/test_apollo_restore.py`

**Interfaces:**
- Consumes: `restore_multichannel` (Task 5).

- [ ] **Step 1: Write the failing test (AudioSR)**

Ubicar el test existente que cubre `_load_mono_48k`/`_run_and_save` (probablemente `tests/test_audiosr_restore.py`) y revisar su fixture de sesión fake antes de escribir. Agregar:

```python
# agregar a tests/test_audiosr_restore.py
def test_run_and_save_preserves_stereo_via_multichannel_restore(tmp_path, monkeypatch, restorer_with_fake_driver):
    import soundfile as sf
    import numpy as np

    input_wav = tmp_path / "in.wav"
    stereo = np.stack([np.full(4800, 0.1), np.full(4800, -0.1)], axis=1).astype(np.float32)
    sf.write(str(input_wav), stereo, 48000)
    output_wav = tmp_path / "out.wav"

    restorer_with_fake_driver._run_and_save(input_wav, output_wav, "cpu")

    result, sr = sf.read(str(output_wav), always_2d=True)
    assert result.shape[1] == 2  # sigue estereo, no colapsa a mono
```

(`restorer_with_fake_driver` debe ser el fixture ya existente que inyecta un driver fake — revisar el archivo de test real para el nombre exacto; si no existe como fixture reusable, replicar el patrón inline que ya usa el test actual de `_run_and_save`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audiosr_restore.py -k stereo -v`
Expected: FAIL (output sigue siendo mono, 1 canal)

- [ ] **Step 3: Implement — AudioSR**

En `app/services/engines/audiosr_restore.py`, reemplazar la llamada a `_load_mono_48k(input_wav)` en `_run_and_save` por una carga multicanal + `restore_multichannel`. Modificar `_load_mono_48k` (renombrar a `_load_audio_48k`, quitar el downmix):

```python
def _load_audio_48k(input_wav: Path) -> np.ndarray:
    import soundfile as sf

    data, sample_rate = sf.read(str(input_wav), dtype="float32", always_2d=True)
    channels = [_resample(data[:, c], sample_rate, AUDIOSR_SAMPLE_RATE) for c in range(data.shape[1])]
    return np.stack(channels, axis=1)
```

En `_run_and_save`, reemplazar:

```python
        audio = _load_mono_48k(input_wav)
```

por:

```python
        from app.services.engines.multichannel_restore import restore_multichannel

        audio = _load_audio_48k(input_wav)
```

Y donde antes se llamaba `driver.restore(audio, ...)` directo sobre el array mono, envolver:

```python
        def restore_mono(mono: np.ndarray) -> np.ndarray:
            return driver.restore(
                mono,
                ddim_steps=self.settings.audiosr_ddim_steps,
                cancel_event=cancel_event,
                step_throttle=(lambda: time.sleep(throttle)) if throttle > 0 else None,
            )

        restored = restore_multichannel(audio, restore_mono)
        _save_wav(output_wav, restored)
```

(Quitar la llamada anterior a `driver.restore(audio, ...)` que operaba sobre el array completo — ahora `restore_multichannel` decide cuántas veces y con qué sub-señal llamar a `restore_mono`.)

`_save_wav` ya usa `sf.write(str(output_wav), audio, AUDIOSR_SAMPLE_RATE)` — `soundfile` acepta arrays 2D `(samples, channels)` sin cambios, no requiere modificación.

- [ ] **Step 4: Run test to verify it passes (AudioSR)**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audiosr_restore.py -v`
Expected: todos verdes, incluyendo el nuevo.

- [ ] **Step 5: Repeat Steps 1-4 for Apollo**

Mismo patrón en `app/services/engines/apollo_restore.py`: renombrar `_load_mono_44k` → `_load_audio_44k` (mismo cambio, sin downmix), envolver `self._restore_chunked(session, mono, throttle, chunk_seconds, overlap_seconds)` como el `restore_mono` pasado a `restore_multichannel` dentro de `_run_and_save`. Agregar el test equivalente en `tests/test_apollo_restore.py`.

- [ ] **Step 6: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add app/services/engines/audiosr_restore.py app/services/engines/apollo_restore.py tests/test_audiosr_restore.py tests/test_apollo_restore.py
git commit -m "Dominio: Apollo y AudioSR restauran M/S en estereo en vez de downmixear a mono (Fase B Task 6)"
```

---

### Task 7: 5.1/7.1 — layouts multicanal + fallback documentado

**Files:**
- Create: `app/services/engines/multichannel_layouts.py`
- Modify: `app/services/engines/multichannel_restore.py`
- Test: `tests/test_multichannel_layouts.py`

**Interfaces:**
- Produces: `restore_surround(audio: np.ndarray, layout: str, restore_mono: RestoreMonoFn) -> np.ndarray` para `layout` en `{"5.1", "7.1"}` (canales en orden ffmpeg estándar: `FL FR FC LFE BL BR [SL SR]`).
- Modifica `restore_multichannel` (Task 5) para despachar a esto cuando `channels in (6, 8)`, y a fallback-mono-con-warning para cualquier otro conteo.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multichannel_layouts.py
import numpy as np
import pytest

from app.services.engines.multichannel_layouts import restore_surround


def _identity(mono: np.ndarray) -> np.ndarray:
    return mono


def test_51_round_trip_identity_reproduces_original():
    rng = np.random.default_rng(2)
    audio = rng.uniform(-0.3, 0.3, size=(1000, 6)).astype(np.float32)
    result = restore_surround(audio, "5.1", _identity)
    np.testing.assert_allclose(result, audio, atol=1e-6)


def test_51_lfe_channel_untouched():
    audio = np.zeros((100, 6), dtype=np.float32)
    audio[:, 3] = 0.42  # LFE

    def fail_if_called(mono):
        raise AssertionError("LFE must never reach restore_mono")

    result = restore_surround(audio, "5.1", lambda mono: mono if np.allclose(mono, 0.42) is False else fail_if_called(mono))
    np.testing.assert_allclose(result[:, 3], audio[:, 3])


def test_51_center_channel_goes_through_restore_mono_directly():
    audio = np.zeros((100, 6), dtype=np.float32)
    audio[:, 2] = 0.5  # FC

    def double(mono):
        return mono * 2

    result = restore_surround(audio, "5.1", double)
    np.testing.assert_allclose(result[:, 2], np.full(100, 1.0))


def test_71_has_eight_channels_round_trip():
    rng = np.random.default_rng(3)
    audio = rng.uniform(-0.3, 0.3, size=(500, 8)).astype(np.float32)
    result = restore_surround(audio, "7.1", _identity)
    np.testing.assert_allclose(result, audio, atol=1e-6)


def test_unknown_layout_raises():
    with pytest.raises(ValueError, match="layout"):
        restore_surround(np.zeros((10, 6), dtype=np.float32), "9.1", _identity)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_multichannel_layouts.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/engines/multichannel_layouts.py
from __future__ import annotations

import numpy as np

from app.services.engines.multichannel_restore import RestoreMonoFn, _rms_match

# Orden estandar ffmpeg: FL FR FC LFE BL BR [SL SR]
_LAYOUT_PAIRS = {
    "5.1": {"front": (0, 1), "center": 2, "lfe": 3, "rear": (4, 5)},
    "7.1": {"front": (0, 1), "center": 2, "lfe": 3, "rear": (4, 5), "side": (6, 7)},
}


def _restore_pair_mid_side(left: np.ndarray, right: np.ndarray, restore_mono: RestoreMonoFn) -> tuple[np.ndarray, np.ndarray]:
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    restored_mid = _rms_match(restore_mono(mid), mid)
    return restored_mid + side, restored_mid - side


def restore_surround(audio: np.ndarray, layout: str, restore_mono: RestoreMonoFn) -> np.ndarray:
    spec = _LAYOUT_PAIRS.get(layout)
    if spec is None:
        raise ValueError(f"Unsupported surround layout: {layout!r}")

    out = audio.copy()
    fl, fr = spec["front"]
    out[:, fl], out[:, fr] = _restore_pair_mid_side(audio[:, fl], audio[:, fr], restore_mono)

    center = spec["center"]
    out[:, center] = _rms_match(restore_mono(audio[:, center]), audio[:, center])

    # LFE (spec["lfe"]) se deja intacto: out ya es una copia de audio, no se toca.

    rl, rr = spec["rear"]
    out[:, rl], out[:, rr] = _restore_pair_mid_side(audio[:, rl], audio[:, rr], restore_mono)

    if "side" in spec:
        sl, sr = spec["side"]
        out[:, sl], out[:, sr] = _restore_pair_mid_side(audio[:, sl], audio[:, sr], restore_mono)

    return out.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_multichannel_layouts.py -v`
Expected: 5 passed. (Exponer `_rms_match` desde `multichannel_restore.py` sin el prefijo de "privado" hacia este módulo hermano es aceptable — mismo paquete, incluso convención ya usada en `validate_driver.py` importando internals de `driver/pipeline.py` en el proyecto hermano GMFSS.)

- [ ] **Step 5: Wire dispatch en `restore_multichannel`**

En `app/services/engines/multichannel_restore.py`, modificar `restore_multichannel`:

```python
import logging

logger = logging.getLogger(__name__)


def restore_multichannel(audio: np.ndarray, restore_mono: RestoreMonoFn) -> np.ndarray:
    channels = audio.shape[1]
    if channels == 1:
        restored = restore_mono(audio[:, 0])
        return _rms_match(restored, audio[:, 0]).reshape(-1, 1)
    if channels == 2:
        return _restore_stereo_mid_side(audio, restore_mono)
    if channels == 6:
        from app.services.engines.multichannel_layouts import restore_surround

        return restore_surround(audio, "5.1", restore_mono)
    if channels == 8:
        from app.services.engines.multichannel_layouts import restore_surround

        return restore_surround(audio, "7.1", restore_mono)
    logger.warning(
        "Unrecognized channel layout (%d channels); falling back to mono restoration", channels
    )
    mono = audio.mean(axis=1)
    restored = _rms_match(restore_mono(mono), mono)
    return np.tile(restored.reshape(-1, 1), (1, channels)).astype(np.float32)
```

Agregar test para el fallback en `tests/test_multichannel_restore.py`:

```python
def test_unknown_channel_count_falls_back_to_mono_with_warning(caplog):
    audio = np.random.default_rng(4).uniform(-0.3, 0.3, size=(200, 4)).astype(np.float32)
    with caplog.at_level("WARNING"):
        result = restore_multichannel(audio, lambda mono: mono)
    assert "channel" in caplog.text.lower()
    assert result.shape == audio.shape
    # las 4 columnas son la misma señal mono repetida
    for c in range(1, 4):
        np.testing.assert_allclose(result[:, c], result[:, 0])
```

- [ ] **Step 6: Run tests, then full backend suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_multichannel_restore.py tests/test_multichannel_layouts.py -v`
Expected: todos verdes.

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add app/services/engines/multichannel_layouts.py app/services/engines/multichannel_restore.py tests/test_multichannel_layouts.py tests/test_multichannel_restore.py
git commit -m "Dominio: restauracion M/S para 5.1/7.1 (frente+rears por par, centro directo, LFE intacto) + fallback a mono con warning (Fase B Task 7)"
```

---

## Fase C — Formato de salida elegible

### Task 8: `audio_output_format` en jobs de video (FLAC + auto-upgrade)

**Files:**
- Modify: `app/models.py` (`VideoUpscaleJob.audio_output_format`)
- Modify: `app/services/video_job_manager.py` (`create_job`, `_resolve_output_container` considera también `audio_output_format`)
- Modify: `app/services/video_upscaler.py` (`_prepare_processed_audio` retorna codec FLAC cuando corresponde)
- Modify: `app/api/routes.py`, `app/schemas.py`
- Test: `tests/test_video_job_manager.py`, `tests/test_video_upscaler.py`

**Interfaces:**
- Consumes: `job.audio_restore` (ya existe), `restore_multichannel` (Fase B, indirectamente vía los motores).
- Produces: `job.audio_output_format: str` (`"auto" | "flac" | "aac"`).

- [ ] **Step 1: Write the failing test**

```python
# agregar a tests/test_video_job_manager.py
@pytest.mark.asyncio
async def test_auto_format_upgrades_container_when_restore_active(manager):
    job = await manager.create_job(
        source_path=manager.settings.uploads_path / "existing.mp4",
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=True,
        audio_restore="apollo",
        audio_output_format="auto",
    )
    assert job.output_container == "mkv"
    assert job.audio_output_format == "auto"
    assert "flac" in job.metadata["containerUpgradedReason"].lower()


@pytest.mark.asyncio
async def test_explicit_aac_does_not_upgrade_container_even_with_restore(manager):
    job = await manager.create_job(
        source_path=manager.settings.uploads_path / "existing.mp4",
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=True,
        audio_restore="apollo",
        audio_output_format="aac",
    )
    assert job.output_container == "mp4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_job_manager.py -k auto_format or explicit_aac -v`
Expected: FAIL (`create_job() got an unexpected keyword argument 'audio_output_format'`)

- [ ] **Step 3: Implement**

`app/models.py` — agregar tras `keep_subtitles: bool = False`:

```python
    audio_output_format: str = "auto"
```

`app/services/video_job_manager.py` — agregar parámetro `audio_output_format: str = "auto"` a `create_job`, pasar a `VideoUpscaleJob(...)`. Reemplazar `_resolve_output_container` por una versión que también considere el formato de audio:

```python
    @staticmethod
    def _resolve_output_container(
        output_container: str, keep_subtitles: bool, audio_restore: str | None, audio_output_format: str
    ) -> tuple[str, str | None]:
        wants_flac = audio_output_format == "flac" or (audio_output_format == "auto" and audio_restore is not None)
        reasons = []
        if keep_subtitles and output_container != "mkv":
            reasons.append("preserve subtitles")
        if wants_flac and output_container != "mkv":
            reasons.append("keep restored audio lossless (FLAC)")
        if not reasons:
            return output_container, None
        return "mkv", f"Output container upgraded to mkv to {' and '.join(reasons)}"
```

Actualizar la llamada en `create_job`:

```python
        resolved_container, container_upgrade_reason = self._resolve_output_container(
            output_container, keep_subtitles, audio_restore, audio_output_format
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_job_manager.py -k auto_format or explicit_aac -v`
Expected: 2 passed.

- [ ] **Step 5: `_prepare_processed_audio` — codec FLAC**

En `app/services/video_upscaler.py`, `_prepare_processed_audio` retorna hoy `["-c:a", "aac", "-b:a", "192k"]` siempre. Cambiar:

```python
    async def _prepare_processed_audio(self, job: VideoUpscaleJob, audio_path: Path) -> tuple[Path, list[str]]:
        current = audio_path.with_name("audio.wav")
        await self._extract_audio_wav(job, current)

        if job.audio_enhance:
            audio_enhanced_path = audio_path.with_name("audio-enhanced.wav")
            await self._enhance_audio(job, current, audio_enhanced_path)
            current = audio_enhanced_path

        if job.audio_restore:
            audio_restored_path = audio_path.with_name("audio-restored.wav")
            await self._restore_audio(job, current, audio_restored_path)
            current = audio_restored_path

        wants_flac = job.audio_output_format == "flac" or (
            job.audio_output_format == "auto" and job.audio_restore is not None
        )
        if wants_flac:
            return current, ["-c:a", "flac"]
        return current, ["-c:a", "aac", "-b:a", "192k"]
```

(El bloque que faltaba mostrar arriba — `if job.audio_restore: ...` — ya existe en el archivo actual, no se reescribe su lógica interna, solo el `return` final cambia.)

- [ ] **Step 6: agregar `audio_output_format` a la ruta/schema**

En `app/api/routes.py`, agregar `audio_output_format: str = Form(default="auto")` a `create_video_job` y pasarlo a `create_job`. En `app/schemas.py`, agregar el campo espejo en el response de job (mismo patrón que `interp_engine`).

- [ ] **Step 7: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures.

- [ ] **Step 8: Commit**

```bash
git add app/models.py app/services/video_job_manager.py app/services/video_upscaler.py app/api/routes.py app/schemas.py tests/test_video_job_manager.py
git commit -m "Aplicacion: audio_output_format auto|flac|aac, FLAC lossless cuando hay restore activo (Fase C Task 8)"
```

---

### Task 9: módulo Audio standalone — selector de formato

**Files:**
- Modify: `app/services/audio_pipeline.py`
- Modify: `app/api/routes.py`, `app/schemas.py`
- Test: `tests/test_audio_pipeline.py`

**Interfaces:**
- Produces: `AudioJob.output_format: str` (`"wav" | "flac" | "mp3"`, default `"flac"`).

- [ ] **Step 1: Write the failing test**

Revisar `tests/test_audio_pipeline.py` existente para el patrón de test del pipeline completo (fixture de job/pipeline fake) antes de escribir.

```python
# agregar a tests/test_audio_pipeline.py
@pytest.mark.asyncio
async def test_output_format_flac_produces_flac_extension(pipeline, tmp_path):
    job = make_audio_job(output_format="flac")  # usar el helper existente del archivo, agregarle el kwarg
    output_path = await pipeline.run(job)
    assert output_path.suffix == ".flac"


@pytest.mark.asyncio
async def test_output_format_wav_produces_wav_extension(pipeline, tmp_path):
    job = make_audio_job(output_format="wav")
    output_path = await pipeline.run(job)
    assert output_path.suffix == ".wav"


@pytest.mark.asyncio
async def test_output_format_mp3_produces_mp3_extension(pipeline, tmp_path):
    job = make_audio_job(output_format="mp3")
    output_path = await pipeline.run(job)
    assert output_path.suffix == ".mp3"


@pytest.mark.asyncio
async def test_default_output_format_is_flac(pipeline, tmp_path):
    job = make_audio_job()  # sin especificar output_format
    assert job.output_format == "flac"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audio_pipeline.py -k output_format -v`
Expected: FAIL (`AudioJob` no tiene `output_format`, o el output sigue siendo siempre `.wav`)

- [ ] **Step 3: Implement**

`app/models.py::AudioJob` — agregar tras `restore: str | None = None`:

```python
    output_format: str = "flac"
```

`app/services/audio_pipeline.py` — la línea `output_path = self.settings.outputs_path / f"{job.id}.wav"` (línea 66) cambia a:

```python
        output_path = self.settings.outputs_path / f"{job.id}.{job.output_format}"
```

Ubicar el paso final donde se escribe/encodea el archivo de salida (probablemente un `ffmpeg`/`soundfile.write` sobre `current` → `output_path`) y parametrizar el codec por `job.output_format`: `flac`→`-c:a flac`, `wav`→sin re-encode (`-c:a pcm_s16le` o el que ya use hoy para wav), `mp3`→`-c:a libmp3lame -b:a 192k`. Seguir el mismo patrón de construcción de comando ffmpeg que ya usa este archivo para el mux actual (no reinventar el wrapper de subprocess).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audio_pipeline.py -v`
Expected: todos verdes.

- [ ] **Step 5: Wire route + schema**

En `app/api/routes.py`, el endpoint de creación de audio job (buscar `create_audio_job` o equivalente) — agregar `output_format: str = Form(default="flac")`, pasar a la construcción del `AudioJob`. En `app/schemas.py`, agregar el campo al response.

- [ ] **Step 6: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add app/models.py app/services/audio_pipeline.py app/api/routes.py app/schemas.py tests/test_audio_pipeline.py
git commit -m "Aplicacion: modulo Audio permite elegir formato de salida wav|flac|mp3, default flac (Fase C Task 9)"
```

---

### Task 10: frontend — selectores de formato con copy amigable

**Files:**
- Modify: `frontend/src/modules/audio/AudioPanel.tsx`, `AudioPanel.test.tsx`
- Modify: `frontend/src/modules/enhance/VideoPanel.tsx`, `VideoPanel.test.tsx`
- Modify: `frontend/src/lib/api.ts` (`CreateAudioJobParams`, `buildAudioJobFormData`)

**Interfaces:**
- Consumes: `audio_output_format` (Task 8), `output_format` (Task 9).

- [ ] **Step 1: Write the failing test (AudioPanel)**

```tsx
// agregar a frontend/src/modules/audio/AudioPanel.test.tsx
it("shows format options with friendly descriptions and defaults to FLAC", () => {
  renderAudioPanel();
  expect(screen.getByLabelText(/flac/i)).toBeChecked();
  expect(screen.getByText(/lossless.*50%|50%.*lighter|smaller/i)).toBeInTheDocument();
  expect(screen.getByText(/only if.*size/i)).toBeInTheDocument(); // copy de MP3
});

it("submits the selected output format", async () => {
  vi.mocked(api.createAudioJob).mockResolvedValueOnce({ jobId: "job1" });
  renderAudioPanel();
  fireEvent.click(screen.getByLabelText(/wav/i));
  fireEvent.click(screen.getByRole("button", { name: /submit|restore|enhance/i }));
  await waitFor(() => expect(api.createAudioJob).toHaveBeenCalled());
  expect(vi.mocked(api.createAudioJob).mock.calls[0][0].outputFormat).toBe("wav");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run AudioPanel`
Expected: FAIL (no existe el selector todavía)

- [ ] **Step 3: Implement**

Leer `AudioPanel.tsx` completo primero (mismo patrón de `restoreMode`/`RestoreMode` que ya se usó en la investigación de este mismo diseño) y agregar un `<fieldset>` de radio buttons análogo, con este copy exacto:

```tsx
const FORMAT_OPTIONS = [
  { value: "flac", label: "FLAC (recommended)", description: "Lossless quality, about 50% smaller than WAV." },
  { value: "wav", label: "WAV", description: "Lossless, uncompressed. Universal compatibility." },
  { value: "mp3", label: "MP3", description: "Lossy, smallest file — only if size matters more than quality." },
] as const;
```

Estado `outputFormat` con default `"flac"` (`useState<"flac" | "wav" | "mp3">("flac")`), radio inputs con `aria-label` = `option.label` (para que `getByLabelText(/flac/i)` matchee), texto de `option.description` renderizado debajo de cada opción. Pasar `outputFormat` al llamado de `createAudioJob`.

En `frontend/src/lib/api.ts`: agregar `outputFormat: string` a `CreateAudioJobParams`, `formData.append("output_format", params.outputFormat)` en `buildAudioJobFormData`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- --run AudioPanel`
Expected: 2 passed + existentes verdes.

- [ ] **Step 5: Repeat for VideoPanel's `audio_output_format`**

Mismo patrón (radio de 2 opciones ahora: `auto` con copy "Recommended — lossless FLAC automatically when audio restoration is on" y `aac` con copy "Standard, smaller files, slight quality loss if restoration is active") solo visible cuando `audioRestore` está activo en `VideoPanel.tsx` (si no hay restore, el formato no importa, no mostrar el selector — evita ruido en la UI para el caso común). Test equivalente en `VideoPanel.test.tsx`.

- [ ] **Step 6: Run full frontend suite + tsc**

Run: `cd frontend && npm test -- --run && npx tsc --noEmit`
Expected: 0 failures, tsc limpio.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/modules/audio/AudioPanel.tsx frontend/src/modules/audio/AudioPanel.test.tsx frontend/src/modules/enhance/VideoPanel.tsx frontend/src/modules/enhance/VideoPanel.test.tsx frontend/src/lib/api.ts
git commit -m "Infraestructura: selector de formato de salida con copy amigable en Audio y Video (Fase C Task 10)"
```

---

### Task 11: docs + verificación final

**Files:**
- Modify: `.env.example`, `README.md`, `CLAUDE.md`

- [ ] **Step 1: `.env.example`** — no requiere variables nuevas (todo es por-job), pero agregar un comentario cerca de `ENABLE_AUDIO_RESTORE`/`ENABLE_AUDIOSR` notando que ahora preservan estéreo/surround (M/S) en vez de mono.
- [ ] **Step 2: `README.md`** — agregar una fila o nota en la sección de video sobre selección de pistas de audio/subtítulos, y actualizar la mención de Apollo/AudioSR para reflejar que preservan estéreo/surround.
- [ ] **Step 3: `CLAUDE.md`** — actualizar la descripción del pipeline de video (línea con "interpolar con RIFE/GMFSS (opcional) → re-encode + audio") para mencionar selección de pistas + subtítulos + auto-upgrade a mkv.
- [ ] **Step 4: correr las 2 suites completas una vez más**

Run: `.venv\Scripts\python.exe -m pytest -q && cd frontend && npm test -- --run && npx tsc --noEmit`
Expected: 0 failures en ambas, tsc limpio.

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md CLAUDE.md
git commit -m "docs: pistas de audio/subtitulos y restauracion M/S en README/CLAUDE.md"
```

---

## Self-Review

- **Cobertura del spec**: Fase A (análisis/selección/subs) → Tasks 1-4. Fase B (M/S+RMS, incl. 5.1/7.1 y fallback documentado) → Tasks 5-7. Fase C (formato elegible en video y audio) → Tasks 8-10. Docs → Task 11. Sin gaps contra `docs/superpowers/specs/2026-07-19-audio-tracks-subs-quality-design.md`.
- **Placeholders**: ninguno — cada step tiene código completo o una instrucción de "leer X primero y seguir su patrón exacto" (nunca "manejar errores apropiadamente" sin decir cómo).
- **Consistencia de tipos**: `restore_multichannel(audio, restore_mono)` firma igual en Tasks 5, 6, 7. `job.audio_track_indices`/`keep_subtitles`/`audio_output_format` mismos nombres en models.py, video_job_manager.py, routes.py, schemas.py, frontend api.ts en todas las tasks que los tocan.
- **Alcance**: 11 tasks, cada una con su propio ciclo de test y commit — mismo grano que SP14. No requiere descomponerse en specs separados (las 3 fases comparten archivos y se benefician de una sola revisión final de rama).
