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


@mcp.tool(name="upflow_generate_3d", annotations={"title": "Generar modelo 3D", **CREATES_JOB})
async def upflow_generate_3d(
    prompt: str,
    source: str = "mesh",
    printer: str = "ender-3",
    target_mm: float = 0.0,
) -> str:
    """Genera un modelo 3D imprimible desde texto (~2 min). Devuelve jobId
    (family shape3d) — el estado incluye canPrint/sizeMm/blockers, descargar
    STL con upflow_download_result.

    source: mesh (Shap-E, orgánico) | cad (OpenSCAD vía LLM, piezas exactas).
    target_mm: tamaño objetivo del eje mayor (solo mesh).
    printer: upflow_capabilities(printers).
    """
    try:
        body: dict[str, Any] = {"prompt": prompt, "printer": printer, "source": source}
        if target_mm:
            body["target_mm"] = target_mm
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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
