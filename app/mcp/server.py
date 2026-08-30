"""Servidor MCP de Upflow.

Expone la funcionalidad del servidor local (upscale de imagen/video, audio,
transcripción, generación, descargas, 3D) como tools MCP para agentes de IA.

Transporte: stdio. Config típica de cliente MCP:

    {"command": "<venv>/Scripts/python", "args": ["-m", "app.mcp"],
     "env": {"UPFLOW_URL": "http://127.0.0.1:8090"}}

El server tiene que estar corriendo aparte; estas tools son un cliente HTTP.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.mcp import client
from app.mcp.errors import format_tool_error
from app.mcp.jobs import (
    FAMILIES,
    FAMILY_NAMES,
    cancel_job,
    family_or_raise,
    fetch_job,
    is_terminal,
    normalize_job,
)

mcp = FastMCP("upflow_mcp")

WAIT_POLL_SECONDS = 2.0
READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
CREATES_JOB = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- sistema


@mcp.tool(name="upflow_status", annotations={"title": "Estado del servidor Upflow", **READ_ONLY})
async def upflow_status() -> str:
    """Estado general: salud del servidor, motor activo, colas, dispositivos
    (CPU/GPU con su execution provider) e identidad/permisos del usuario actual.

    Usala primero para verificar que Upflow está corriendo y qué hay disponible.
    Devuelve JSON {health, engine, devices, me}.
    """
    try:
        health = await client.api_get("/api/v1/health")
        engine = await client.api_get("/api/v1/engine")
        devices = await client.api_get("/api/v1/devices")
        me = await client.api_get("/api/v1/auth/me")
        return _dump({"health": health, "engine": engine, "devices": devices, "me": me})
    except Exception as exc:
        return format_tool_error(exc)


CAPABILITY_ENDPOINTS = {
    "overview": "/api/v1/capabilities/tree",
    "video": "/api/v1/video/capabilities",
    "audio": "/api/v1/audio/capabilities",
    "voice_catalog": "/api/v1/audio/voice-catalog",
    "generation": "/api/v1/generation/capabilities",
    "generation_video": "/api/v1/generation/video/capabilities",
    "tts": "/api/v1/tts/capabilities",
    "voice_conversion": "/api/v1/voice/conversion/capabilities",
    "editor": "/api/v1/editor/capabilities",
    "realtime": "/api/v1/realtime/capabilities",
    "translation": "/api/v1/translation/pairs",
    "hardware": "/api/v1/capabilities",
    "part_kinds": "/api/v1/print/parts",
    "printers": "/api/v1/print/printers",
    "model3d": "/api/v1/model3d/capabilities",
}


@mcp.tool(name="upflow_capabilities", annotations={"title": "Capacidades disponibles", **READ_ONLY})
async def upflow_capabilities(domain: str = "overview") -> str:
    """Qué features están disponibles/instaladas.

    domain:
      overview (default) — árbol completo de features por dominio
      video — motores de interpolación instalados (rife/gmfss)
      audio — modos de denoise/restore y presets de mastering
      voice_catalog — pasos de cadena de voz y targets de loudness
      generation — modelos de difusión instalados y devices
      generation_video — modelos de generación de video
      tts — voces Kokoro disponibles
      voice_conversion / editor / realtime / translation / hardware
      part_kinds — piezas paramétricas 3D disponibles y sus parámetros
      printers — impresoras 3D conocidas y tamaño de cama
      model3d — si hay Blender y qué operaciones de modelado desbloquea
    """
    try:
        path = CAPABILITY_ENDPOINTS.get(domain)
        if path is None:
            return (
                f"Error: dominio '{domain}' desconocido. "
                f"Válidos: {', '.join(sorted(CAPABILITY_ENDPOINTS))}"
            )
        return _dump(await client.api_get(path))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_list_models", annotations={"title": "Modelos instalados", **READ_ONLY})
async def upflow_list_models() -> str:
    """Lista todos los modelos registrados (upscalers, ASR, generación, etc.)
    con id, tipo, escala, tamaño y estado. Los `id` sirven como `model_id` en
    las tools de upscale/transcribe/generación."""
    try:
        return _dump(await client.api_get("/api/v1/models"))
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------------- jobs genéricos


@mcp.tool(name="upflow_job_status", annotations={"title": "Estado de un job", **READ_ONLY})
async def upflow_job_status(family: str, job_id: str) -> str:
    """Estado normalizado de un job de cualquier familia.

    family: image | video | audio | generation | transcribe | download | shape3d
    Devuelve {family, jobId, status, progressPct, error, downloadUrl, ...extras}.
    status terminal: completed | failed | cancelled.
    """
    try:
        return _dump(await fetch_job(family_or_raise(family), job_id))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_wait_job", annotations={"title": "Esperar un job", **READ_ONLY})
async def upflow_wait_job(family: str, job_id: str, timeout_seconds: int = 300) -> str:
    """Espera (polling) hasta que el job termine o venza el timeout (5-1800s).

    Devuelve el último estado visto; si `status` no es terminal al volver,
    el job sigue corriendo — volvé a llamar o usá upflow_job_status.
    Para videos largos conviene timeout alto o polling manual espaciado.
    """
    try:
        fam = family_or_raise(family)
        timeout = max(5, min(int(timeout_seconds), 1800))
        deadline = asyncio.get_event_loop().time() + timeout
        job = await fetch_job(fam, job_id)
        while not is_terminal(job) and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(WAIT_POLL_SECONDS)
            job = await fetch_job(fam, job_id)
        if not is_terminal(job):
            job["waitTimedOut"] = True
        return _dump(job)
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_cancel_job", annotations={"title": "Cancelar un job", "readOnlyHint": False, "destructiveHint": True, "openWorldHint": False, "idempotentHint": True})
async def upflow_cancel_job(family: str, job_id: str) -> str:
    """Cancela un job en cola o corriendo. Familias: image | video | audio |
    generation | transcribe | download | shape3d."""
    try:
        return _dump(await cancel_job(family_or_raise(family), job_id))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_download_result", annotations={"title": "Descargar resultado de un job", **READ_ONLY})
async def upflow_download_result(
    family: str,
    job_id: str,
    destination_path: str,
    transcript_format: str = "txt",
    translate_to: str = "",
    file_index: int = 0,
    stem: str = "",
) -> str:
    """Guarda el resultado de un job completado en un archivo local.

    destination_path: ruta destino (si es carpeta, se usa un nombre default).
    transcript_format (solo transcribe): txt | srt | vtt | video.
    translate_to (solo transcribe): código de idioma para traducir subtítulos.
    file_index (solo download): índice del archivo cuando la descarga produjo varios.
    stem (solo audio con separación): id de stem del modelo usado — karaoke:
    instrumental | vocals; de-reverb (reverb_hq): dry | wet. Los stems válidos
    del job vienen en upflow_job_status (campo stems). Un stem que no aplica
    devuelve el 400 de la API listando los válidos.
    Devuelve {outputPath}. Falla con 409 si el job no está completed.
    """
    try:
        fam = family_or_raise(family)
        params: dict[str, Any] = {}
        default_name = fam.default_output_name
        if fam.name == "audio" and stem:
            # Sin whitelist local: los stems dependen del modelo del job y la
            # API es la autoridad — un id inválido vuelve como 400 accionable.
            params["stem"] = stem
            default_name = f"{stem}.flac"
        if fam.name == "transcribe":
            params["fmt"] = transcript_format
            if translate_to:
                params["translate_to"] = translate_to
            if transcript_format not in ("txt", "video"):
                default_name = f"transcript.{transcript_format}"
        if fam.name == "download":
            params["index"] = file_index
            job = await fetch_job(fam, job_id)
            outputs = job.get("outputFiles") or []
            if 0 <= file_index < len(outputs):
                default_name = outputs[file_index]
        destination = client.resolve_output_path(destination_path, default_name)
        await client.api_download(
            f"{fam.base_path}/{job_id}/download", destination, params=params or None
        )
        return _dump({"outputPath": str(destination)})
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_list_jobs", annotations={"title": "Listar jobs", **READ_ONLY})
async def upflow_list_jobs(family: str = "") -> str:
    """Lista jobs propios. family vacío = las 7 familias (image, video, audio,
    generation, transcribe, download, shape3d)."""
    try:
        results: dict[str, Any] = {}
        wanted = [family] if family else list(FAMILY_NAMES)
        for name in wanted:
            fam = family_or_raise(name)
            payload = await client.api_get(fam.base_path)
            raw_jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
            results[name] = [normalize_job(fam, job) for job in raw_jobs]
        return _dump(results)
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------------- imagen


@mcp.tool(name="upflow_upscale_image", annotations={"title": "Reescalar imagen", **CREATES_JOB})
async def upflow_upscale_image(
    file_path: str,
    model_name: str = "realesrgan-x4plus",
    scale: int = 4,
    output_format: str = "png",
    device: str = "",
    model_id: str = "",
    wait: bool = True,
    destination_path: str = "",
) -> str:
    """Reescala una imagen local con Real-ESRGAN u otro modelo instalado.

    file_path: ruta local de la imagen. scale: 2-4 según modelo.
    model_name: del catálogo builtin (ver upflow_status → engine.supportedModels);
    model_id: alternativo, un modelo instalado por el usuario (upflow_list_models).
    wait=True espera el resultado (segundos típicamente). Si además pasás
    destination_path, guarda el resultado y devuelve outputPath directo.
    """
    try:
        name, content = client.read_upload(file_path)
        data: dict[str, Any] = {
            "model_name": model_name,
            "scale": str(scale),
            "output_format": output_format,
        }
        if device:
            data["device"] = device
        if model_id:
            data["model_id"] = model_id
        created = await client.api_post(
            "/api/v1/jobs",
            data=data,
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        fam = FAMILIES["image"]
        job_id = created.get("jobId")
        if not wait:
            return _dump(normalize_job(fam, created))
        job = await fetch_job(fam, job_id)
        while not is_terminal(job):
            await asyncio.sleep(WAIT_POLL_SECONDS)
            job = await fetch_job(fam, job_id)
        if destination_path and job.get("status") == "completed":
            destination = client.resolve_output_path(
                destination_path, f"upscaled-{name}.{output_format}"
            )
            await client.api_download(f"{fam.base_path}/{job_id}/download", destination)
            job["outputPath"] = str(destination)
        return _dump(job)
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------------- video


@mcp.tool(name="upflow_analyze_video", annotations={"title": "Analizar video", **CREATES_JOB})
async def upflow_analyze_video(file_path: str) -> str:
    """Sube un video y devuelve sus pistas de audio/subtítulos + un uploadToken
    reutilizable en upflow_upscale_video (evita subir el archivo dos veces).
    Usala antes de reescalar si necesitás elegir pistas de audio o subtítulos."""
    try:
        name, content = client.read_upload(file_path)
        payload = await client.api_post(
            "/api/v1/video/analyze",
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        return _dump(payload)
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_upscale_video", annotations={"title": "Reescalar/interpolar video", **CREATES_JOB})
async def upflow_upscale_video(
    file_path: str = "",
    upload_token: str = "",
    profile_key: str = "anime-balanced-2x",
    model_name: str = "",
    scale: int = 0,
    fps_multiplier: int = 0,
    target_fps: int = 0,
    target_height: int = 0,
    interp_engine: str = "",
    keep_audio: bool = True,
    audio_enhance: str = "",
    audio_restore: str = "",
    audio_track_indices: str = "",
    keep_subtitles: bool = False,
    audio_output_format: str = "auto",
    output_container: str = "",
    video_codec: str = "",
    video_encoder: str = "",
    crf: int = 0,
    device: str = "",
    model_id: str = "",
    backend: str = "",
) -> str:
    """Crea un job de reescalado/interpolación de video (proceso largo — devuelve
    jobId para seguir con upflow_job_status / upflow_wait_job y descargar con
    upflow_download_result).

    Pasá exactamente uno: file_path (ruta local) o upload_token (de
    upflow_analyze_video). profile_key define modelo+escala+codec (ver
    upflow_status → engine.videoProfiles); los parámetros explícitos pisan el
    perfil (0 o "" = usar el default del perfil). fps_multiplier 2-4 interpola
    frames (interp_engine: rife rápido | gmfss máxima calidad, muy lento).
    audio_enhance/audio_restore: modos de upflow_capabilities(audio).
    audio_track_indices: CSV de índices de upflow_analyze_video; la primera
    pista es la que pasa por enhance/restore.
    """
    try:
        if bool(file_path) == bool(upload_token):
            return "Error: pasá exactamente uno de file_path o upload_token."
        data: dict[str, Any] = {"profile_key": profile_key}
        optional_form = {
            "model_name": model_name,
            "interp_engine": interp_engine,
            "audio_enhance": audio_enhance,
            "audio_restore": audio_restore,
            "audio_track_indices": audio_track_indices,
            "audio_output_format": audio_output_format,
            "output_container": output_container,
            "video_codec": video_codec,
            "video_encoder": video_encoder,
            "device": device,
            "model_id": model_id,
            "backend": backend,
        }
        data.update({key: value for key, value in optional_form.items() if value})
        optional_numeric = {
            "scale": scale,
            "fps_multiplier": fps_multiplier,
            "target_fps": target_fps,
            "target_height": target_height,
            "crf": crf,
        }
        data.update({key: str(value) for key, value in optional_numeric.items() if value})
        data["keep_audio"] = "true" if keep_audio else "false"
        data["keep_subtitles"] = "true" if keep_subtitles else "false"
        files = None
        if file_path:
            name, content = client.read_upload(file_path)
            files = {"file": (name, content, "application/octet-stream")}
        else:
            data["upload_token"] = upload_token
        created = await client.api_post(
            "/api/v1/video/jobs", data=data, files=files, timeout=client.UPLOAD_TIMEOUT
        )
        return _dump(normalize_job(FAMILIES["video"], created))
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------------- audio


@mcp.tool(name="upflow_process_audio", annotations={"title": "Procesar audio", **CREATES_JOB})
async def upflow_process_audio(
    file_path: str,
    denoise: str = "",
    restore: str = "",
    master: str = "",
    voice_steps: str = "",
    voice_delivery: str = "",
    voice_presence_db: float = 0.0,
    output_format: str = "flac",
    lossy_quality: str = "maximum",
    device: str = "",
    cleanup_steps: str = "",
    separate: bool = False,
    separation_model: str = "",
    practice_stems: list[str] | None = None,
    practice_guide_percent: int = 0,
    transcribe_stems: list[str] | None = None,
) -> str:
    """Crea un job de procesamiento de audio: limpieza (quitar ruido/eco/reverb
    de música), denoise, restauración (Apollo/AudioSR), cadena de voz,
    mastering, separación voz/instrumental (karaoke), o SOLO conversión de
    formato. Devuelve jobId (seguir con upflow_wait_job, descargar con
    upflow_download_result).

    CONVERSIÓN PURA: sin ningún paso (sin denoise/restore/master/voice_steps/
    cleanup_steps/separate), con output_format distinto al del archivo, el job
    es válido y hace UNA sola pasada de ffmpeg del original al formato destino.
    No pasa por el decode a 48 kHz / 16 bits que necesitan los motores: un FLAC
    de 44.1 kHz y 24 bits convertido a WAV sale a 44.1 kHz y 24 bits. A un
    destino con pérdida se conserva la tasa si el codec la admite; si no (96 kHz
    a MP3), se resamplea a la más cercana soportada y queda dicho en
    metadata.conversionResampled — nunca en silencio. Lo mismo con un downmix
    forzado (metadata.conversionDownmixed). Pedir el MISMO formato que el
    origen sin ningún paso devuelve 400: no hay nada que hacer.

    cleanup_steps: CSV de ids de la CADENA DE LIMPIEZA (upflow_capabilities(audio)
    → cleanupSteps). Encadena una pasada por id sobre CUALQUIER audio y devuelve
    UN archivo limpio (los stems removidos no se guardan). Se combina libremente
    con denoise/restore/voice_steps/master en el mismo job.
      * El ORDEN es FIJO y lo pone el catálogo, no el CSV: quitar ruido primero
        (el ruido de banda ancha confunde a todo lo que venga después), quitar
        eco después (reflejos discretos), quitar reverb al final (cola difusa).
        Mandarlos en otro orden da exactamente la misma cadena.
      * EXCLUSIVIDAD por familia: deecho_normal y deecho_aggressive son el mismo
        modelo en dos intensidades (elegí uno), y deecho_dereverb hace eco y
        reverb en una sola pasada, así que excluye a los dos de-echo y también a
        reverb_hq. Una combinación redundante devuelve 400 nombrando el par.
      * Cada pasada es CON PÉRDIDA (son máscaras: descartan señal). Desde la
        tercera, el job marca metadata.cleanupOverprocessed.

    separate=True (karaoke): separa el audio en instrumental + voces (dos
    salidas; bajar cada una con upflow_download_result y su stem). Es
    EXCLUSIVO: no se combina con denoise/restore/voice/master ni con
    cleanup_steps en el mismo job (la separación entrega dos archivos y la
    cadena uno) — encadená un segundo job sobre el stem que quieras.
    separation_model: id del catálogo (upflow_capabilities(audio) →
    separationModels); vacío = el default instalado. Los modelos de limpieza
    también están ahí: correrlos por separate=True es la forma de escuchar QUÉ
    sacó una pasada, porque devuelve los dos stems.
    practice_stems (solo con separate=True y un modelo de 3+ stems): lista de
    stem ids a los que hornear una pista de práctica "minus-one" (la canción
    SIN ese instrumento); cada derivado se baja con stem=minus_<id>.
    practice_guide_percent (0-30): porcentaje del instrumento removido que
    queda sonando de guía; 0 = quitarlo del todo.
    transcribe_stems (solo con separate=True): lista de stem ids CON altura a
    transcribir a MIDI+MusicXML (+tab para guitar/bass); nunca "drums" ni un
    id derivado minus_*. Cada archivo se baja con stem=<id>&fmt=midi|
    musicxml|tab. Requiere el pack music-transcription (upflow_capabilities
    (audio) -> "audio.stemTranscription").
    Modos válidos: upflow_capabilities(audio) y upflow_capabilities(voice_catalog).
    voice_steps: CSV de pasos de la cadena de voz en orden.
    output_format: wav | flac | mp3 | m4a. Los dos primeros son sin pérdida
    (conservan tasa y profundidad del origen); m4a es AAC en contenedor MP4, el
    más compatible con teléfonos y con el ecosistema Apple.
    lossy_quality: maximum (MP3 320k / AAC 256k, default) | balanced (192k) |
    compact (128k). Solo aplica a mp3 y m4a; en wav/flac se ignora.
    """
    try:
        name, content = client.read_upload(file_path)
        data: dict[str, Any] = {
            "output_format": output_format,
            "lossy_quality": lossy_quality,
        }
        if separate:
            data["separate"] = "true"
        if separation_model:
            # Siempre viaja: si falta separate, la API responde su 400 explícito
            # ("separation_model solo aplica cuando separate=true") en vez de
            # descartar el pedido en silencio y crear otro job del que se pidió.
            data["separation_model"] = separation_model
        optional = {
            "denoise": denoise,
            "restore": restore,
            "master": master,
            "voice_steps": voice_steps,
            "voice_delivery": voice_delivery,
            "cleanup_steps": cleanup_steps,
            "device": device,
        }
        data.update({key: value for key, value in optional.items() if value})
        if voice_presence_db:
            data["voice_presence_db"] = str(voice_presence_db)
        # Solo si son truthy: la API valida el combo (separate, modelo 3+ stems)
        # y su 400 explica mejor que un default inventado acá.
        if practice_stems:
            data["practice_stems"] = ",".join(practice_stems)
        if practice_guide_percent:
            data["practice_guide_percent"] = str(practice_guide_percent)
        if transcribe_stems:
            data["transcribe_stems"] = ",".join(transcribe_stems)
        created = await client.api_post(
            "/api/v1/audio/jobs",
            data=data,
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        return _dump(normalize_job(FAMILIES["audio"], created))
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------------- transcripción


@mcp.tool(name="upflow_transcribe", annotations={"title": "Transcribir / doblar", **CREATES_JOB})
async def upflow_transcribe(
    file_path: str,
    model_id: str,
    language: str = "",
    output_mode: str = "text",
    target_language: str = "",
    device: str = "",
) -> str:
    """Transcribe audio/video con un modelo ASR instalado; también subtitula o
    dobla video. Devuelve jobId — el texto llega inline en upflow_job_status
    (campo text); subtítulos srt/vtt se bajan con upflow_download_result.

    model_id: modelo ASR instalado (upflow_list_models, kind asr; instalar con
    upflow_install_model kind=asr). output_mode: text | video (subs muxeados) |
    video_burned (subs quemados) | dubbed_video (doblaje — requiere
    target_language). language: código ISO del audio origen (auto si vacío).
    """
    try:
        name, content = client.read_upload(file_path)
        data: dict[str, Any] = {"model_id": model_id, "output_mode": output_mode}
        optional = {
            "language": language,
            "target_language": target_language,
            "device": device,
        }
        data.update({key: value for key, value in optional.items() if value})
        created = await client.api_post(
            "/api/v1/transcribe/jobs",
            data=data,
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        return _dump(normalize_job(FAMILIES["transcribe"], created))
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------------- descargas de medios


@mcp.tool(name="upflow_probe_media_url", annotations={"title": "Inspeccionar URL de video", "readOnlyHint": True, "destructiveHint": False, "openWorldHint": True})
async def upflow_probe_media_url(url: str) -> str:
    """Inspecciona una URL (YouTube, etc.) sin descargar: título, duración,
    alturas disponibles, si es playlist y cuántas entradas tiene."""
    try:
        return _dump(await client.api_post("/api/v1/download/probe", json_body={"url": url}))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_download_media", annotations={"title": "Descargar video/audio de una URL", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
async def upflow_download_media(
    url: str,
    max_height: int = 1080,
    audio_only: bool = False,
    audio_format: str = "mp3",
    video_container: str = "mp4",
    include_playlist: bool = False,
    playlist_limit: int = 10,
    subtitle_languages: str = "",
) -> str:
    """Descarga video/audio de una URL (yt-dlp). Devuelve jobId — seguir con
    upflow_wait_job y bajar archivos con upflow_download_result (file_index
    para playlists). subtitle_languages: CSV de códigos de idioma."""
    try:
        body: dict[str, Any] = {
            "url": url,
            "max_height": max_height,
            "audio_only": audio_only,
            "audio_format": audio_format,
            "video_container": video_container,
            "include_playlist": include_playlist,
            "playlist_limit": playlist_limit,
        }
        if subtitle_languages:
            body["subtitle_languages"] = [
                lang.strip() for lang in subtitle_languages.split(",") if lang.strip()
            ]
        created = await client.api_post("/api/v1/download/jobs", json_body=body)
        return _dump(normalize_job(FAMILIES["download"], created))
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------------- generación


async def _stage_init_image(file_path: str) -> str:
    name, content = client.read_upload(file_path)
    payload = await client.api_post(
        "/api/v1/generation/init-image",
        files={"file": (name, content, "application/octet-stream")},
        timeout=client.UPLOAD_TIMEOUT,
    )
    return payload["initImageToken"]


@mcp.tool(name="upflow_generate_image", annotations={"title": "Generar imagen/video (difusión)", **CREATES_JOB})
async def upflow_generate_image(
    prompt: str,
    model_id: str,
    negative_prompt: str = "",
    steps: int = 0,
    guidance: float = 0.0,
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    scheduler: str = "",
    device: str = "",
    init_image_path: str = "",
    mask_image_path: str = "",
    strength: float = 0.0,
    auto_upscale: bool = False,
    frames: int = 0,
    fps: int = 0,
) -> str:
    """Genera una imagen (txt2img/img2img/inpaint) o video corto con un modelo
    de difusión instalado. Devuelve jobId (upflow_wait_job + upflow_download_result).

    model_id: de upflow_capabilities(generation). init_image_path: imagen base
    para img2img; mask_image_path: máscara (blanco = regenerar) para inpaint.
    strength 0-1 (cuánto cambia la imagen base). width/height 64-1024 múltiplos
    de 32. seed -1 = aleatorio. frames/fps solo para modelos de video
    (upflow_capabilities(generation_video)). steps/guidance 0 = default del modelo.
    """
    try:
        body: dict[str, Any] = {
            "prompt": prompt,
            "model_id": model_id,
            "width": width,
            "height": height,
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if steps:
            body["steps"] = steps
        if guidance:
            body["guidance"] = guidance
        if seed >= 0:
            body["seed"] = seed
        if scheduler:
            body["scheduler"] = scheduler
        if device:
            body["device"] = device
        if strength:
            body["strength"] = strength
        if auto_upscale:
            body["auto_upscale"] = True
        if frames:
            body["frames"] = frames
        if fps:
            body["fps"] = fps
        if init_image_path:
            body["init_image_token"] = await _stage_init_image(init_image_path)
        if mask_image_path:
            body["mask_image_token"] = await _stage_init_image(mask_image_path)
        created = await client.api_post("/api/v1/generation/jobs", json_body=body)
        return _dump(normalize_job(FAMILIES["generation"], created))
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------------- voz


@mcp.tool(name="upflow_text_to_speech", annotations={"title": "Texto a voz", **CREATES_JOB})
async def upflow_text_to_speech(text: str, voice: str, destination_path: str) -> str:
    """Sintetiza voz (Kokoro TTS) y guarda un WAV local. Sincrónico.

    text: máx 2000 caracteres. voice: de upflow_capabilities(tts).
    Devuelve {outputPath}. 409 si el pack de TTS no está instalado
    (instalable vía upflow_provision_pack).
    """
    try:
        wav = await client.api_post(
            "/api/v1/tts/synthesize",
            json_body={"text": text, "voice": voice},
            timeout=300.0,
        )
        if not isinstance(wav, (bytes, bytearray)):
            return _dump(wav)
        destination = client.resolve_output_path(destination_path, "voz.wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(wav)
        return _dump({"outputPath": str(destination)})
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------------- 3D


@mcp.tool(name="upflow_init_image", annotations={"title": "Subir imagen de partida", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def upflow_init_image(file_path: str) -> str:
    """Sube una imagen local y devuelve su imageToken, reutilizable.

    Es la puerta de entrada de todo lo que parte de una imagen: img2img,
    inpaint, selección por clic, insertar objeto y foto a malla. Sin esto,
    esos carriles solo existen para quien usa la pantalla.
    """
    try:
        name, content = client.read_upload(file_path)
        payload = await client.api_post(
            "/api/v1/generation/init-image",
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        return _dump(payload)
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_generate_3d", annotations={"title": "Generar modelo 3D", **CREATES_JOB})
async def upflow_generate_3d(
    prompt: str = "",
    source: str = "mesh",
    printer: str = "ender-3",
    target_mm: float = 0.0,
    image_path: str = "",
) -> str:
    """Genera un modelo 3D imprimible desde texto o desde una foto (~2 min).
    Devuelve jobId (family shape3d) — el estado incluye canPrint/sizeMm/
    blockers, descargar STL con upflow_download_result.

    source: mesh (Shap-E desde texto) | photo (Shap-E img2img) | cad (OpenSCAD
    vía LLM, única vía con cotas exactas).
    image_path: obligatorio con source=photo; se sube solo. Un DIBUJO PLANO
    sale mal — las zonas sin pistas de volumen salen como losas. Para modelado
    de personaje usá upflow_reference_scene, no esto.
    target_mm: tamaño objetivo del eje mayor (mesh y photo).
    printer: upflow_capabilities(printers).
    """
    try:
        body: dict[str, Any] = {"prompt": prompt, "printer": printer, "source": source}
        if target_mm:
            body["target_mm"] = target_mm
        if image_path:
            subida = await client.api_post(
                "/api/v1/generation/init-image",
                files={"file": (*client.read_upload(image_path), "application/octet-stream")},
                timeout=client.UPLOAD_TIMEOUT,
            )
            body["imageToken"] = subida.get("initImageToken") or subida.get("token")
            body["source"] = "photo"
        created = await client.api_post("/api/v1/print/generate", json_body=body)
        return _dump(normalize_job(FAMILIES["shape3d"], created))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_make_part", annotations={"title": "Pieza paramétrica 3D", **CREATES_JOB})
async def upflow_make_part(
    kind: str,
    params_json: str,
    printer: str = "ender-3",
    destination_path: str = "",
) -> str:
    """Construye una pieza paramétrica con dimensiones exactas (sincrónico) y
    la verifica contra la cama de la impresora.

    kind y parámetros válidos: upflow_capabilities(part_kinds).
    params_json: JSON de {param: valor_mm}, ej '{"width": 40, "height": 20}'.
    Si pasás destination_path, guarda el STL y devuelve outputPath.
    """
    try:
        params = json.loads(params_json)
        payload = await client.api_post(
            "/api/v1/print/parts",
            json_body={"kind": kind, "params": params, "printer": printer},
        )
        if destination_path and payload.get("downloadUrl"):
            destination = client.resolve_output_path(destination_path, f"{kind}.stl")
            await client.api_download(payload["downloadUrl"], destination)
            payload["outputPath"] = str(destination)
        return _dump(payload)
    except json.JSONDecodeError:
        return "Error: params_json no es JSON válido. Ejemplo: '{\"width\": 40}'"
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_check_stl", annotations={"title": "Verificar STL imprimible", **READ_ONLY})
async def upflow_check_stl(file_path: str, printer: str = "ender-3") -> str:
    """Verifica un STL local contra una impresora: cabe en la cama, watertight,
    manifold, voladizos. Devuelve verdict con blockers y consejos. Sincrónico."""
    try:
        name, content = client.read_upload(file_path)
        payload = await client.api_post(
            "/api/v1/print/check",
            data={"printer": printer},
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        return _dump(payload)
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------- modelado 3D
#
# Este carril NO genera el personaje: prepara el andamiaje para modelarlo. Es
# deterministico —lo hace Blender, no un modelo— y por eso sale igual las mil
# veces. Las piezas son atomicas para poder medir entre paso y paso: encadenar
# operaciones de malla sin auditar en el medio es aplicar parches sin compilar.


@mcp.tool(name="upflow_model3d_capabilities", annotations={"title": "Que hay para modelado 3D", **READ_ONLY})
async def upflow_model3d_capabilities() -> str:
    """Dice si hay Blender, cuál, y qué operaciones desbloquea.

    Blender no se baja desde la app: lo instala el usuario. Preguntá esto
    antes de encadenar nada del carril, porque sin Blender no hay carril.
    """
    try:
        return _dump(await client.api_get("/api/v1/model3d/capabilities"))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_audit_mesh", annotations={"title": "Auditar malla", **READ_ONLY})
async def upflow_audit_mesh(file_path: str) -> str:
    """Mide una malla sin tocarla: STL, OBJ, PLY, GLB, GLTF o FBX.

    Devuelve caras, cuádruples vs triángulos, n-gons, aristas no-manifold,
    islas sueltas, UVs y medidas, más blockers (impiden seguir) y warnings
    (saldría mejor de otra forma). Sincrónico.
    """
    try:
        name, content = client.read_upload(file_path)
        payload = await client.api_post(
            "/api/v1/model3d/audit",
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        return _dump(payload)
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_remesh", annotations={"title": "Rehacer la topología de una malla", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def upflow_remesh(
    file_path: str,
    voxel_meters: float = 0.01,
    destination_path: str = "",
) -> str:
    """Rehace la topología de una malla por voxeles y devuelve el ANTES y el DESPUÉS.

    Es lo que convierte primitivas que se solapan —o una malla con agujeros— en
    una sola superficie cerrada. Uniformiza: no respeta aristas vivas, así que
    para una pieza con cantos rectos NO es lo que querés.

    voxel_meters: el tamaño del voxel. Más chico = más caras y más detalle;
    mínimo 0.002 m. Medido sobre un personaje de 1,70 m: 0.05 m deja 8k caras y
    0.02 m deja 51k, partiendo de 288k.

    Las dos auditorías viajan juntas porque un remesh gana topología y pierde
    detalle, y cuánto perdió solo se ve comparando. No hay modo de cuádruples:
    QuadriFlow cancela en este entorno y devolver "listo" con la malla intacta
    sería mentir.
    """
    try:
        name, content = client.read_upload(file_path)
        payload = await client.api_post(
            "/api/v1/model3d/remesh",
            data={"voxelMeters": voxel_meters},
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        if destination_path and payload.get("downloadUrl"):
            destino = client.resolve_output_path(destination_path, "remallada.glb")
            await client.api_download(payload["downloadUrl"], destino)
            payload["outputPath"] = str(destino)
        return _dump(payload)
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_generate_mesh", annotations={"title": "Generar una malla desde una imagen", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def upflow_generate_mesh(
    file_path: str,
    engine: str = "triposg",
    destination_path: str = "",
    steps: int = 50,
    guidance: float = 7.0,
) -> str:
    """Genera una malla 3D desde UNA imagen con un motor generativo LOCAL.

    Nada sale de esta máquina. Mirá `upflow_model3d_capabilities` primero: dice
    qué motores hay, con qué licencia y qué le falta a cada uno; un motor
    ausente no es un error, es la respuesta.

    Lo que devuelve NO está aprobado por haber salido —`audited` viene en
    false—. Un generador puede devolver una superficie preciosa con doscientas
    islas sueltas. El paso siguiente es `upflow_score_fit` (cuánto calza con el
    dibujo) o `upflow_audit_mesh` (si la malla sirve), y recién ahí se decide.

    Sobre los motores, medido el 2026-08-28: el que es libre no corre en AMD y
    el que corre en AMD no es libre. `triposg` es MIT y anda en CPU, que es por
    lo que está primero — no por ser el mejor de la comparativa.
    """
    try:
        name, content = client.read_upload(file_path)
        payload = await client.api_post(
            "/api/v1/model3d/generate",
            data={"engine": engine, "steps": steps, "guidance": guidance},
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        if destination_path and payload.get("downloadUrl"):
            destino = client.resolve_output_path(destination_path, "generada.glb")
            await client.api_download(payload["downloadUrl"], destino)
            payload["outputPath"] = str(destino)
        return _dump(payload)
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_score_fit", annotations={"title": "Medir cuánto calza una malla con el dibujo", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def upflow_score_fit(
    token: str,
    file_path: str,
    height_meters: float,
    scale_view: str = "",
    resolution: int = 512,
) -> str:
    """Mide cuánto se parece una malla al dibujo, y de QUÉ TIPO es la diferencia.

    Es la balanza: la misma medida para una malla generada por un modelo,
    esculpida a mano o armada con primitivas, así que comparar dos formas de
    llegar deja de ser cuestión de opinión. Usá `upflow_sheet_views` primero
    para obtener el token.

    Por vista devuelve `blame`, que es lo accionable:
      - "escala": las dos medidas se van para el mismo lado. Reescalá; modelar
        no lo arregla.
      - "partes": el contorno está bien pero las partes no caen en el mismo
        lugar adentro. NO significa mover la malla en la escena —la comparación
        centra las dos siluetas, así que una traslación global no cambia el
        número, probado— sino mover una PARTE respecto del resto.
      - "forma": proporción y centrado correctos y aun así no calza. Recién acá
        hay que modelar.

    height_meters: la altura REAL de la vista que fija la escala (`scale_view`,
    por defecto la más alta). Es UNA sola escala para toda la hoja a propósito:
    escalar cada vista por su propia tinta las deja a escalas distintas y
    entonces ninguna malla puede calzar las dos —medido sobre una gorra, donde
    de frente el punto más bajo era la banda y de perfil la punta de la visera.

    La auditoría de topología viaja junto al calce: una malla puede calzar la
    silueta y ser inservible por estar rota.
    """
    try:
        name, content = client.read_upload(file_path)
        datos: dict[str, object] = {"heightMeters": height_meters, "resolution": resolution}
        if scale_view:
            datos["scaleView"] = scale_view
        return _dump(
            await client.api_post(
                f"/api/v1/model3d/fit/{token}",
                data=datos,
                files={"file": (name, content, "application/octet-stream")},
                timeout=client.UPLOAD_TIMEOUT,
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_sheet_views", annotations={"title": "Partir hoja de turnaround", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def upflow_sheet_views(file_path: str, expected_views: int = 4) -> str:
    """Parte una hoja de turnaround en sus vistas y devuelve un token.

    Descarta sola la barra de altura y la paleta de color. Nombra las vistas
    por POSICIÓN (front, side, back, side_left): es una convención, no una
    deducción — mirá el resultado y corregí con upflow_rename_views si la
    hoja viene en otro orden.
    Revisá `warnings`: si detectó menos vistas de las esperadas puede haber
    dos dibujadas superpuestas, y esas no se separan solas.
    El token alimenta upflow_reference_scene.
    """
    try:
        name, content = client.read_upload(file_path)
        payload = await client.api_post(
            "/api/v1/model3d/sheet/views",
            data={"expectedViews": expected_views},
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        return _dump(payload)
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_rename_views", annotations={"title": "Corregir qué vista es cada recorte", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def upflow_rename_views(token: str, names: str) -> str:
    """Reasigna qué vista es cada recorte, de izquierda a derecha.

    names: separados por coma, uno por vista detectada. Válidos: front, side,
    back, side_left.

    Nombrarlas por posición es una convención, no una deducción: mirando los
    píxeles no hay forma de saber si el tercer panel es la espalda o un tres
    cuartos. Si la hoja viene en otro orden, la escena de referencia sale con
    el dibujo equivocado en cada plano y NADA lo delata — por eso conviene
    mirar las vistas antes de armarla.
    """
    try:
        pedidos = [n.strip() for n in names.split(",") if n.strip()]
        return _dump(
            await client.api_post(
                f"/api/v1/model3d/sheet/{token}/names", json_body={"names": pedidos}
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_sheet_proportions", annotations={"title": "Proporciones del personaje", **READ_ONLY})
async def upflow_sheet_proportions(token: str, height_meters: float = 1.70) -> str:
    """Cuántas cabezas mide el personaje y cuánto de ancho tiene a cada altura.

    token: el que devolvió upflow_sheet_views.
    Es lo que un modelador mira antes de empezar. Cada altura viaja con lo que
    opinan las DOS vistas: `agrees` en false significa que cada una la ubica en
    otro lado y no hay que fiarse — medido sobre una hoja real, un personaje con
    los brazos colgando no tiene cintura en su silueta, y forzar la medición
    devuelve un número inventado con cara de dato.
    """
    try:
        return _dump(
            await client.api_get(
                f"/api/v1/model3d/proportions/{token}",
                params={"heightMeters": height_meters},
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_reference_scene", annotations={"title": "Escena de referencia en Blender", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def upflow_reference_scene(
    token: str,
    height_meters: float = 1.70,
    destination_path: str = "",
) -> str:
    """Arma el .blend con las vistas alineadas a escala real, listo para modelar.

    token: el que devolvió upflow_sheet_views.
    height_meters: cuánto mide el personaje de verdad. Escala por la TINTA del
    dibujo, no por el tamaño del archivo, así que el margen del recorte no
    miente sobre la altura.
    Deja los pies en el origen, cada vista detrás de su cámara, la colección
    bloqueada para no agarrarla con el ratón y fuera del render final.
    Si pasás destination_path, baja el .blend ahí y devuelve outputPath.
    """
    try:
        payload = await client.api_post(
            "/api/v1/model3d/reference-scene",
            json_body={"token": token, "heightMeters": height_meters},
            # El servidor se da 900 s para Blender; con el techo default del
            # cliente (120 s) el corte venia del lado equivocado y el mensaje
            # decia "reintenta" sobre algo que estaba andando bien.
            timeout=client.UPLOAD_TIMEOUT,
        )
        if destination_path and payload.get("downloadUrl"):
            destino = client.resolve_output_path(destination_path, "escena.blend")
            await client.api_download(payload["downloadUrl"], destino)
            payload["outputPath"] = str(destino)
        return _dump(payload)
    except Exception as exc:
        return format_tool_error(exc)


# ------------------------------------------------------------ editor de imagen
#
# Todo lo que parte de una imagen empieza por upflow_init_image: el token que
# devuelve es la moneda comun de estas tools.


@mcp.tool(name="upflow_segment_object", annotations={"title": "Seleccionar objeto por clic", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def upflow_segment_object(
    image_token: str,
    x: float,
    y: float,
    destination_path: str = "",
    device: str = "",
) -> str:
    """Recorta el objeto que hay en un punto de la imagen y devuelve su máscara.

    x e y van en píxeles de la imagen ORIGINAL (la que subió
    upflow_init_image, que devuelve su width/height), no de ninguna vista
    escalada.

    La ruta devuelve un PNG en escala de grises, no JSON — inservible para
    encadenar. Así que la máscara se vuelve a subir sola y lo que devuelve
    esta tool es `maskToken`, que es lo que consume upflow_insert_object.
    Con destination_path además queda el PNG en disco para poder mirarlo.

    Requiere el pack de selección por toque (MobileSAM): si no está,
    upflow_capabilities("editor") lo dice antes de intentar.
    """
    try:
        cuerpo: dict[str, Any] = {"imageToken": image_token, "x": x, "y": y}
        if device:
            cuerpo["device"] = device
        mascara = await client.api_post("/api/v1/editor/segment", json_body=cuerpo)
        if not isinstance(mascara, bytes):
            return _dump(mascara)

        if destination_path:
            destino = client.resolve_output_path(destination_path, "mascara.png")
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(mascara)

        subida = await client.api_post(
            "/api/v1/generation/init-image",
            files={"file": ("mascara.png", mascara, "image/png")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        salida = {"maskToken": subida.get("initImageToken"), "bytes": len(mascara)}
        if destination_path:
            salida["outputPath"] = str(client.resolve_output_path(destination_path, "mascara.png"))
        return _dump(salida)
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_insert_object", annotations={"title": "Insertar objeto en otra imagen", **CREATES_JOB})
async def upflow_insert_object(
    target_token: str,
    source_token: str,
    source_mask_token: str,
    x: int = 0,
    y: int = 0,
    width: int = 8,
    height: int = 8,
    target_mask_token: str = "",
    feather_px: int = 6,
    match_color: bool = True,
    harmonize: bool = False,
    harmonize_blend: float = 0.35,
    model_id: str = "",
    prompt: str = "",
) -> str:
    """Pega un objeto de una imagen en otra, con su máscara.

    target_token / source_token: destino y origen, de upflow_init_image.
    source_mask_token: la máscara del objeto en el ORIGEN, de
    upflow_segment_object. Tiene que medir exactamente lo mismo que el origen.

    x, y, width, height: el rectángulo destino en píxeles del destino. El
    objeto se ajusta ADENTRO conservando su proporción y centrado, no se
    estira — el resultado suele ser más chico que la caja. Los valores por
    defecto (8x8 en 0,0) no sirven para nada real: mandá una caja de verdad.

    target_mask_token: modo reemplazo — la máscara de un objeto que ya está en
    el destino. Con esto x/y/width/height se IGNORAN: posición y tamaño salen
    de esa máscara.

    harmonize: segunda pasada de inpaint sobre la costura. Exige model_id de un
    modelo de inpainting real (9 canales); uno normal se rechaza. Crea un job
    de la familia generation.
    """
    try:
        cuerpo: dict[str, Any] = {
            "targetToken": target_token,
            "sourceToken": source_token,
            "sourceMaskToken": source_mask_token,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "featherPx": feather_px,
            "matchColor": match_color,
            "harmonize": harmonize,
            "harmonizeBlend": harmonize_blend,
        }
        if target_mask_token:
            cuerpo["targetMaskToken"] = target_mask_token
        if model_id:
            cuerpo["modelId"] = model_id
        if prompt:
            cuerpo["prompt"] = prompt
        return _dump(await client.api_post("/api/v1/editor/insert-object", json_body=cuerpo))
    except Exception as exc:
        return format_tool_error(exc)


# ------------------------------------------------------------ mallas y piezas


@mcp.tool(name="upflow_repair_mesh", annotations={"title": "Reparar malla", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def upflow_repair_mesh(file_path: str, destination_path: str = "") -> str:
    """Cierra los agujeros de una malla rota y vuelve a MEDIRLA.

    La malla reparada se entrega IGUAL cuando no quedó cerrada, y el reporte
    lo dice: quien pide decide si le sirve. Decir "reparada" sobre algo que
    sigue abierto sería el peor falso positivo.
    Sincrónico. Con destination_path baja el STL reparado.
    """
    try:
        name, content = client.read_upload(file_path)
        payload = await client.api_post(
            "/api/v1/print/repair",
            files={"file": (name, content, "application/octet-stream")},
            timeout=client.UPLOAD_TIMEOUT,
        )
        if destination_path and payload.get("downloadUrl"):
            destino = client.resolve_output_path(destination_path, "pieza-reparada.stl")
            await client.api_download(payload["downloadUrl"], destino)
            payload["outputPath"] = str(destino)
        return _dump(payload)
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_estimate_size", annotations={"title": "Estimar tamaño real de un objeto", **READ_ONLY})
async def upflow_estimate_size(prompt: str) -> str:
    """Cuánto mide de verdad el objeto que describís, en milímetros.

    Sirve para elegir target_mm sin inventarlo. Es una ESTIMACIÓN de un
    modelo de lenguaje contra un objeto de referencia, no una cota: para una
    pieza que tiene que encajar está el carril paramétrico.
    """
    try:
        return _dump(await client.api_post("/api/v1/print/estimate-size", json_body={"prompt": prompt}))
    except Exception as exc:
        return format_tool_error(exc)


# ------------------------------------------------------------ prompts guardados


@mcp.tool(name="upflow_prompt_presets", annotations={"title": "Prompts de ejemplo", **READ_ONLY})
async def upflow_prompt_presets() -> str:
    """Catálogo fijo de prompts de ejemplo por modo, que trae la app."""
    try:
        return _dump(await client.api_get("/api/v1/generation/prompt-presets"))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_saved_prompts", annotations={"title": "Prompts guardados", **READ_ONLY})
async def upflow_saved_prompts() -> str:
    """Los prompts que guardó este usuario. El `id` sirve para borrarlos."""
    try:
        return _dump(await client.api_get("/api/v1/generation/saved-prompts"))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_save_prompt", annotations={"title": "Guardar un prompt", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": False})
async def upflow_save_prompt(
    name: str,
    prompt: str,
    negative_prompt: str = "",
    mode: str = "",
) -> str:
    """Guarda un prompt con nombre. Devuelve su `id`, que es el ÚNICO
    identificador para borrarlo después — anotalo si vas a necesitarlo."""
    try:
        cuerpo: dict[str, Any] = {"name": name, "prompt": prompt}
        if negative_prompt:
            cuerpo["negativePrompt"] = negative_prompt
        if mode:
            cuerpo["mode"] = mode
        return _dump(await client.api_post("/api/v1/generation/saved-prompts", json_body=cuerpo))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_delete_saved_prompt", annotations={"title": "Borrar un prompt guardado", "readOnlyHint": False, "destructiveHint": True, "openWorldHint": False})
async def upflow_delete_saved_prompt(prompt_id: str) -> str:
    """Borra un prompt guardado. Un segundo borrado del mismo id da 404, no
    error real: significa que ya no estaba."""
    try:
        return _dump(await client.api_delete(f"/api/v1/generation/saved-prompts/{prompt_id}"))
    except Exception as exc:
        return format_tool_error(exc)


# ---------------------------------------------------------------- modelos


SEARCH_ENDPOINTS = {
    "upscaler": "/api/v1/models/search",
    "asr": "/api/v1/asr/models/search",
    "generation": "/api/v1/generation/models/search",
}

INSTALL_ENDPOINTS = {
    "upscaler": ("/api/v1/models/install", "/api/v1/models/install/{id}"),
    "asr": ("/api/v1/asr/models/install", "/api/v1/asr/models/install/{id}"),
    "generation": ("/api/v1/generation/models", "/api/v1/generation/models/install/{id}"),
    "generation_vulkan": (
        "/api/v1/generation/models/vulkan",
        "/api/v1/generation/models/vulkan/{id}",
    ),
}


@mcp.tool(name="upflow_search_models", annotations={"title": "Buscar modelos en Hugging Face", "readOnlyHint": True, "destructiveHint": False, "openWorldHint": True})
async def upflow_search_models(kind: str, query: str = "") -> str:
    """Busca modelos instalables en Hugging Face, filtrados por compatibilidad.

    kind: upscaler | asr | generation. query vacío = explorar los más
    descargados (upscaler requiere query). Los repoId del resultado se
    instalan con upflow_install_model."""
    try:
        path = SEARCH_ENDPOINTS.get(kind)
        if path is None:
            return f"Error: kind '{kind}' inválido. Válidos: {', '.join(sorted(SEARCH_ENDPOINTS))}"
        return _dump(await client.api_get(path, params={"q": query}))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_install_model", annotations={"title": "Instalar modelo", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
async def upflow_install_model(
    kind: str,
    repo_id: str,
    precision: str = "",
    checkpoint_path: str = "",
    filename: str = "",
) -> str:
    """Instala un modelo desde Hugging Face (descarga + conversión automática).
    Devuelve installId — seguir con upflow_install_status.

    kind: upscaler | asr | generation | generation_vulkan.
    precision (generation): fp16 default. filename: solo generation_vulkan
    (checkpoint suelto .gguf/.safetensors). Antes de instalar modelos de
    generación conviene mirar el preflight de VRAM en la UI."""
    try:
        endpoints = INSTALL_ENDPOINTS.get(kind)
        if endpoints is None:
            return f"Error: kind '{kind}' inválido. Válidos: {', '.join(sorted(INSTALL_ENDPOINTS))}"
        body: dict[str, Any] = {"repo_id": repo_id}
        if kind == "generation_vulkan":
            if not filename:
                return "Error: generation_vulkan requiere filename (checkpoint dentro del repo)."
            body["filename"] = filename
        else:
            if precision:
                body["precision"] = precision
            if checkpoint_path:
                body["checkpoint_path"] = checkpoint_path
        return _dump(await client.api_post(endpoints[0], json_body=body))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_install_status", annotations={"title": "Estado de instalación de modelo", **READ_ONLY})
async def upflow_install_status(kind: str, install_id: str) -> str:
    """Estado de una instalación de modelo (progressPct, status, modelId al
    terminar). kind: upscaler | asr | generation | generation_vulkan."""
    try:
        endpoints = INSTALL_ENDPOINTS.get(kind)
        if endpoints is None:
            return f"Error: kind '{kind}' inválido. Válidos: {', '.join(sorted(INSTALL_ENDPOINTS))}"
        return _dump(await client.api_get(endpoints[1].format(id=install_id)))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_provision_pack", annotations={"title": "Instalar pack de binarios", "readOnlyHint": False, "destructiveHint": False, "openWorldHint": True})
async def upflow_provision_pack(pack: str = "", capability_id: str = "", job_id: str = "") -> str:
    """Descarga packs de binarios/modelos que le faltan a una feature.

    Tres usos: capability_id (instala lo que le falte a esa feature del árbol
    de upflow_capabilities), pack (por nombre directo), o job_id (consultar
    estado de una provisión en curso). Pasá exactamente uno.
    """
    try:
        provided = [value for value in (pack, capability_id, job_id) if value]
        if len(provided) != 1:
            return "Error: pasá exactamente uno de pack, capability_id o job_id."
        if job_id:
            return _dump(await client.api_get(f"/api/v1/capabilities/provision/{job_id}"))
        if capability_id:
            return _dump(
                await client.api_post(f"/api/v1/capabilities/{capability_id}/provision")
            )
        return _dump(await client.api_post(f"/api/v1/packs/{pack}/provision"))
    except Exception as exc:
        return format_tool_error(exc)


# ------------------------------------------------ MCP REST faltantes: modelos y sistema
#
# Bloque aislado para que estos equivalentes REST no se mezclen con los tools
# de impresión ni con el grupo de prompts guardados que se editan en paralelo.


PREFLIGHT_ENDPOINTS = {
    "upscaler": "/api/v1/models/preflight",
    "generation": "/api/v1/generation/models/preflight",
}


@mcp.tool(name="upflow_convert_model", annotations={"title": "Convertir modelo de generación", **CREATES_JOB})
async def upflow_convert_model(repo_id: str, precision: str = "") -> str:
    """Inicia la conversión de un repo de Hugging Face a un modelo ONNX de
    generación. Devuelve conversionId; consultalo con upflow_conversion_status.

    precision vacío usa fp16, el default de la API. La conversión puede tardar
    varios minutos y sigue corriendo aunque el cliente cambie de sección.
    """
    try:
        body: dict[str, Any] = {"repo_id": repo_id}
        if precision:
            body["precision"] = precision
        return _dump(
            await client.api_post(
                "/api/v1/generation/models/convert", json_body=body
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_conversion_status", annotations={"title": "Estado de conversión de modelo", **READ_ONLY})
async def upflow_conversion_status(conversion_id: str) -> str:
    """Consulta progreso, etapa, resultado o error de una conversión.

    Repetí la consulta mientras status no sea terminal; modelId aparece cuando
    el modelo convertido quedó registrado.
    """
    try:
        return _dump(
            await client.api_get(
                f"/api/v1/generation/models/convert/{conversion_id}"
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(
    name="upflow_cancel_conversion",
    annotations={
        "title": "Cancelar conversión de modelo",
        "readOnlyHint": False,
        "destructiveHint": True,
        "openWorldHint": False,
    },
)
async def upflow_cancel_conversion(conversion_id: str) -> str:
    """Solicita detener una conversión en curso.

    El corte ocurre al terminar el submodelo actual; si la conversión ya acabó,
    la API responde que no queda trabajo por cancelar.
    """
    try:
        return _dump(
            await client.api_post(
                f"/api/v1/generation/models/convert/{conversion_id}/cancel"
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_list_conversions", annotations={"title": "Conversiones de modelos activas", **READ_ONLY})
async def upflow_list_conversions() -> str:
    """Lista las conversiones que siguen activas para recuperar sus ids y
    volver a enganchar el seguimiento después de perder el estado de la UI."""
    try:
        return _dump(
            await client.api_get("/api/v1/generation/models/conversions")
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_create_inpaint_model", annotations={"title": "Crear variante de inpainting", **CREATES_JOB})
async def upflow_create_inpaint_model(model_id: str) -> str:
    """Crea una variante de inpainting a partir de un modelo de generación
    instalado que conserve un origen compatible en Hugging Face.

    Devuelve conversionId para seguirla con upflow_conversion_status; la API
    rechaza modelos ya inpaint o sin los pesos de origen necesarios.
    """
    try:
        return _dump(
            await client.api_post(
                f"/api/v1/generation/models/{model_id}/create-inpaint"
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_optimize_model", annotations={"title": "Optimizar modelo de generación", **CREATES_JOB})
async def upflow_optimize_model(model_id: str) -> str:
    """Crea la variante optimizada por fusión de grafo de un modelo ONNX
    instalado. La API valida origen, arquitectura, RAM libre y pesos antes de
    encolar; devuelve conversionId para upflow_conversion_status."""
    try:
        return _dump(
            await client.api_post(
                f"/api/v1/generation/models/{model_id}/optimize"
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_model_preflight", annotations={"title": "Preflight de modelo instalable", **READ_ONLY})
async def upflow_model_preflight(
    kind: str,
    repo_id: str,
    width: int = 512,
    height: int = 512,
) -> str:
    """Evalúa compatibilidad y recursos antes de instalar un modelo.

    kind: upscaler | generation. El preflight de generación usa width/height
    para estimar el costo; el de upscaler solo necesita repo_id. Devuelve las
    mediciones reales disponibles y deja en null lo que no pudo medir.
    """
    try:
        path = PREFLIGHT_ENDPOINTS.get(kind)
        if path is None:
            return (
                f"Error: kind '{kind}' inválido. "
                f"Válidos: {', '.join(sorted(PREFLIGHT_ENDPOINTS))}"
            )
        params: dict[str, Any] = {"repoId": repo_id}
        if kind == "generation":
            params.update({"width": width, "height": height})
        return _dump(await client.api_get(path, params=params))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_settings", annotations={"title": "Ajustes editables", **READ_ONLY})
async def upflow_settings() -> str:
    """Lista qué ajustes admite cambiar la instalación, si están configurados,
    cuáles exigen reinicio y el valor visible de los flags no secretos."""
    try:
        return _dump(await client.api_get("/api/v1/settings"))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(
    name="upflow_update_setting",
    annotations={
        "title": "Actualizar ajuste global",
        "readOnlyHint": False,
        "destructiveHint": True,
        "openWorldHint": False,
    },
)
async def upflow_update_setting(key: str, value: str) -> str:
    """Cambia la configuración de la app para todos los usuarios de esta
    instalación. Consultá upflow_settings para saber qué claves son editables
    y si el cambio exige reiniciar antes de que tenga efecto."""
    try:
        return _dump(
            await client.api_patch(
                "/api/v1/settings", json_body={"key": key, "value": value}
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(
    name="upflow_rescan",
    annotations={"title": "Reescanear capacidades", **CREATES_JOB},
)
async def upflow_rescan() -> str:
    """Vuelve a detectar binarios, modelos y demás capacidades locales después
    de una instalación o cambio externo, y devuelve el catálogo actualizado."""
    try:
        return _dump(await client.api_post("/api/v1/capabilities/rescan"))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_update_check", annotations={"title": "Buscar actualización de Upflow", **READ_ONLY})
async def upflow_update_check() -> str:
    """Consulta si existe una versión más reciente y devuelve la versión local,
    la última publicada, fecha de comprobación y enlace de la release."""
    try:
        return _dump(await client.api_get("/api/v1/update-check"))
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_onnx_diagnostics", annotations={"title": "Diagnósticos ONNX", **READ_ONLY})
async def upflow_onnx_diagnostics() -> str:
    """Lista las combinaciones modelo/dispositivo diagnosticables y los
    reportes cacheados de operaciones que cayeron a CPU."""
    try:
        return _dump(
            await client.api_get("/api/v1/capabilities/onnx-diagnostics")
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(
    name="upflow_onnx_scan",
    annotations={"title": "Escanear fallback ONNX", **CREATES_JOB},
)
async def upflow_onnx_scan(model_id: str, device_id: str) -> str:
    """Ejecuta el diagnóstico de un modelo ONNX en un dispositivo concreto y
    señala qué operaciones terminaron en CPU. Usá los ids de
    upflow_onnx_diagnostics para elegir una combinación válida."""
    try:
        return _dump(
            await client.api_post(
                "/api/v1/capabilities/onnx-diagnostics/"
                f"{model_id}/{device_id}/scan"
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(
    name="upflow_fix_lever",
    annotations={"title": "Aplicar reparación de capacidad", **CREATES_JOB},
)
async def upflow_fix_lever(lever_id: str) -> str:
    """Aplica la reparación ofrecida por una palanca de capacidad y devuelve su
    estado actualizado. Solo sirve para palancas marcadas como fixable por el
    catálogo de capacidades."""
    try:
        return _dump(
            await client.api_post(f"/api/v1/capabilities/{lever_id}/fix")
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(
    name="upflow_provision_capability",
    annotations={
        "title": "Provisionar capacidad",
        "readOnlyHint": False,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def upflow_provision_capability(capability_id: str) -> str:
    """Descarga el primer pack pendiente de una capacidad resoluble.

    Devuelve jobId; seguí el progreso con upflow_provision_pack(job_id=...).
    Si la capacidad ya está lista o no tiene un pack pendiente, la API explica
    por qué no puede provisionarla.
    """
    try:
        return _dump(
            await client.api_post(
                f"/api/v1/capabilities/{capability_id}/provision"
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(name="upflow_audio_compare", annotations={"title": "Comparar separadores de audio", **CREATES_JOB})
async def upflow_audio_compare(
    file_path: str,
    models: str,
    excerpt_seconds: int = 30,
    offset_seconds: float | None = None,
) -> str:
    """Corre varios modelos de separación sobre el mismo fragmento de audio
    para comparar el material real del usuario, no un ranking genérico.

    models: ids separados por comas. excerpt_seconds define la duración;
    offset_seconds vacío deja que el servidor elija el centro del archivo.
    Devuelve un jobId por modelo y el recorte exacto usado.
    """
    try:
        name, content = client.read_upload(file_path)
        data: dict[str, Any] = {
            "models": models,
            "excerpt_seconds": str(excerpt_seconds),
        }
        if offset_seconds is not None:
            data["offset_seconds"] = str(offset_seconds)
        return _dump(
            await client.api_post(
                "/api/v1/audio/compare",
                data=data,
                files={"file": (name, content, "application/octet-stream")},
                timeout=client.UPLOAD_TIMEOUT,
            )
        )
    except Exception as exc:
        return format_tool_error(exc)


@mcp.tool(
    name="upflow_realtime_start",
    annotations={"title": "Iniciar overlay en tiempo real", **CREATES_JOB},
)
async def upflow_realtime_start(
    preset: str,
    max_frame_rate: int | None = None,
) -> str:
    """Inicia el overlay local de escalado en tiempo real con un preset
    disponible en upflow_capabilities(realtime).

    max_frame_rate es opcional y admite 24-480; la respuesta devuelve el pid
    del proceso lanzado y el preset efectivo.
    """
    try:
        body: dict[str, Any] = {"preset": preset}
        if max_frame_rate is not None:
            body["max_frame_rate"] = max_frame_rate
        return _dump(
            await client.api_post("/api/v1/realtime/start", json_body=body)
        )
    except Exception as exc:
        return format_tool_error(exc)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
