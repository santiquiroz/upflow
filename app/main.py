from __future__ import annotations

import logging

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.auth_routes import router as auth_router
from app.api.capability_routes import router as capability_router
from app.api.editor_routes import router as editor_router
from app.api.routes import router as api_router
from app.api.users_routes import router as users_router
from app.core.log_file import configure_file_logging
from app.config import AUDIO_ENHANCE_MODES, ensure_auth_secret, get_settings
from app.security import LoopbackGuardMiddleware, OriginGuardMiddleware
from app.services.audio_job_manager import AudioJobManager
from app.services.audio_pipeline import AudioPipeline
from app.services.auth.identity import LocalPasswordProvider
from app.services.auth.quotas import QuotaService
from app.services.auth.user_store import UserStore
from app.services.capability_probe import CapabilityProbe
from app.services.device_router import DeviceRouter
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import DevicesService
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.restorer_registry import build_restorers
from app.services.engines.audio_enhance import AudioEnhancer
from app.services.engines.generation_onnx import GenerationEngine
from app.services.engines.gmfss_engine import GmfssEngine
from app.services.engines.mdx_separator import MdxSeparator
from app.services.engines.roformer_separator import RoformerSeparator
from app.services.engines.umx_separator import UmxSeparator
from app.services.engines.vr_deecho_separator import VrDeEchoSeparator
from app.services.engines.onnx_upscaler import OnnxUpscaler
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.engines.rife_ncnn import RifeNcnnEngine
from app.services.engines.voice_enhance import VoiceEnhancer
from app.services.generation_converter import GenerationModelConverter
from app.services.generation_installer import GenerationModelInstaller
from app.services.generation_job_manager import GenerationJobManager
from app.services.hf_client import HfClient
from app.services.job_manager import JobManager
from app.services.media_tools import MediaTools
from app.services.model_installer import ModelInstaller
from app.services.asr_installer import AsrModelInstaller
from app.services.engines.transcribe_onnx import TranscribeEngine
from app.services.download_chain import start_separation_followups
from app.services.download_job_manager import DownloadJobManager
from app.services.karaoke_job_manager import KaraokeJobManager
from app.services.transcribe_job_manager import TranscribeJobManager
from app.services.engines.shape3d import Shape3dEngine
from app.services.openscad_llm import OpenAiCompatibleClient
from app.services.model_packs import enqueue_pending_model_packs
from app.services.shape3d_job_manager import Shape3dJobManager
from app.services.pack_provisioner import PackProvisioner
from app.services.model_registry import ModelRegistry
from app.services.onnx_cpu_fallback_probe import OnnxCpuFallbackProbe
from app.services.resource_probes import DxgiVramProbe, SystemRamProbe
from app.services.retention_sweeper import RetentionSweeper
from app.services.settings_service import register_live_settings
from app.services.storage import StorageService
from app.services.update_service import UpdateService
from app.services.video_job_manager import VideoJobManager
from app.services.video_upscaler import VideoUpscaler

APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = APP_DIR.parent / "frontend" / "dist"


def _build_migan_eraser(settings):
    from app.services.engines.migan_eraser import MiganEraser

    return MiganEraser(settings)


def _build_sdcpp_engine(settings):
    # Lane experimental Fase 3: solo se construye si el flag está encendido;
    # apagado no importa ni el módulo.
    if not settings.enable_sdcpp:
        return None
    from app.services.engines.sdcpp_engine import SdcppEngine

    return SdcppEngine(settings)


def _build_video_engine(settings):
    if not settings.enable_sdcpp:
        return None
    from app.services.engines.sdcpp_video import SdcppVideoEngine

    return SdcppVideoEngine(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    register_live_settings(settings)
    log_file = configure_file_logging(settings)
    if log_file is not None:
        logging.getLogger(__name__).info("log a archivo activo en %s", log_file)
    if settings.auth_mode == "multi":
        ensure_auth_secret(settings)
    user_store = UserStore(settings)
    identity_provider = LocalPasswordProvider(user_store)
    quota_service = QuotaService(settings)
    storage = StorageService(settings)
    engine = RealEsrganNcnnEngine(settings)
    media_tools = MediaTools(settings)
    rife_engine = RifeNcnnEngine(settings)
    audio_enhancers = {mode: AudioEnhancer(settings, mode) for mode in AUDIO_ENHANCE_MODES}
    # Shared across the ONNX-session-caching engines so a device switch
    # between them evicts only the previous owner's entry for that device
    # (see GpuSessionCoordinator docstring).
    gpu_coordinator = GpuSessionCoordinator()
    gmfss_engine = GmfssEngine(settings, gpu_coordinator)
    restorers = build_restorers(settings, gpu_coordinator)
    devices_service = DevicesService(settings)
    capability_probe = CapabilityProbe(settings)
    onnx_cpu_fallback_probe = OnnxCpuFallbackProbe(settings, devices_service, gpu_coordinator)
    model_registry = ModelRegistry(settings)
    onnx_engine = OnnxUpscaler(settings, model_registry, devices_service, gpu_coordinator)
    onnx_video_engine = OnnxVideoUpscaler(settings, model_registry, devices_service, gpu_coordinator)
    # Subproyecto B: real VRAM/RAM admission on top of the existing
    # job-count gate. "npu" has no real probe yet (no NPU enumeration story
    # in devices_service.py) -- omitting it from this dict is equivalent to
    # registering a NullProbe, both fail-open.
    resource_probes = {"gpu": DxgiVramProbe(), "cpu": SystemRamProbe()}
    device_semaphores = DeviceSemaphores(settings, resource_probes=resource_probes)
    generation_engine = GenerationEngine(settings, gpu_coordinator)
    generation_job_manager = GenerationJobManager(
        settings,
        generation_engine,
        device_semaphores,
        registry=model_registry,
        upscale_engine=engine,
        onnx_upscale_engine=onnx_engine,
        devices=devices_service,
        quota_service=quota_service,
        sdcpp_engine=_build_sdcpp_engine(settings),
        migan_eraser=_build_migan_eraser(settings),
        video_engine=_build_video_engine(settings),
    )
    # Shared across both managers (like device_semaphores) so an auto-routed
    # image job and an auto-routed video job never pick the same free
    # device in the same race window -- see DeviceRouter's docstring.
    device_router = DeviceRouter(device_semaphores)
    job_manager = JobManager(
        settings,
        engine,
        device_semaphores,
        onnx_engine=onnx_engine,
        registry=model_registry,
        devices=devices_service,
        device_router=device_router,
        quota_service=quota_service,
    )
    video_upscaler = VideoUpscaler(
        settings,
        engine,
        media_tools,
        rife_engine,
        gmfss_engine,
        audio_enhancers,
        onnx_engine=onnx_engine,
        model_registry=model_registry,
        restorers=restorers,
        onnx_video_engine=onnx_video_engine,
        devices=devices_service,
    )
    video_job_manager = VideoJobManager(
        settings,
        video_upscaler,
        media_tools,
        device_semaphores,
        registry=model_registry,
        devices=devices_service,
        device_router=device_router,
        quota_service=quota_service,
    )
    voice_enhancer = VoiceEnhancer(settings)
    separators = {
        "mdx": MdxSeparator(settings, gpu_coordinator),
        "vr": VrDeEchoSeparator(settings, gpu_coordinator),
        "roformer": RoformerSeparator(settings, gpu_coordinator),
        "umx": UmxSeparator(settings, gpu_coordinator),
    }
    audio_pipeline = AudioPipeline(
        settings,
        audio_enhancers,
        restorers,
        voice_enhancer=voice_enhancer,
        separators=separators,
    )
    audio_job_manager = AudioJobManager(
        settings,
        audio_pipeline,
        device_semaphores,
        devices=devices_service,
        quota_service=quota_service,
    )
    update_service = UpdateService(settings)
    hf_client = HfClient(settings)
    model_installer = ModelInstaller(settings, model_registry, hf_client)
    generation_installer = GenerationModelInstaller(
        settings, model_registry, hf_client, gpu_coordinator, device_semaphores
    )
    generation_converter = GenerationModelConverter(settings, generation_installer, hf_client)
    pack_provisioner = PackProvisioner(settings)
    asr_installer = AsrModelInstaller(settings, model_registry, hf_client)
    transcribe_engine = TranscribeEngine(settings, gpu_coordinator)
    transcribe_jobs = TranscribeJobManager(
        settings,
        transcribe_engine,
        device_semaphores,
        registry=model_registry,
        devices=devices_service,
        quota_service=quota_service,
        # Los MISMOS motores que el modulo de audio: el karaoke separa, y un
        # segundo juego duplicaria los .onnx en VRAM.
        separators=separators,
    )
    karaoke_jobs = KaraokeJobManager(
        settings,
        transcribe_engine,
        device_semaphores,
        registry=model_registry,
        separators=separators,
        restorers=restorers,
        devices=devices_service,
        quota_service=quota_service,
    )
    download_jobs = DownloadJobManager(
        settings,
        quota_service=quota_service,
        follow_up=lambda job, owner: start_separation_followups(
            job,
            audio_jobs=audio_job_manager,
            uploads_path=settings.uploads_path,
            owner=owner,
        ),
    )
    generation_installer.enqueue_conversion = generation_converter.convert_from_hf
    await job_manager.start()
    await video_job_manager.start()
    await audio_job_manager.start()
    await model_installer.start()
    await generation_job_manager.start()
    await generation_installer.start()
    await generation_converter.start()
    await enqueue_pending_model_packs(model_registry, generation_converter)
    await pack_provisioner.start()
    await asr_installer.start()
    shape3d_jobs = Shape3dJobManager(
        settings,
        Shape3dEngine(
            settings.shape3d_model_path, settings.shape3d_img2img_model_path
        ),
        # Solo si hay un servidor de modelo configurado. Sin el, el carril de
        # malla anda igual y el de CAD dice que falta, que es mejor que fingir.
        cad_client=(
            OpenAiCompatibleClient(
                base_url=settings.cad_llm_base_url, model=settings.cad_llm_model
            )
            if settings.cad_llm_base_url
            else None
        ),
        quota_service=quota_service,
    )

    # Los SIETE managers, no solo los cuatro originales: los que faltaban
    # (transcribe/shape3d/download) dejaban a un usuario no-admin sin limite de
    # concurrencia ni de cola en esas familias.
    quota_service.attach_managers(
        job_manager,
        video_job_manager,
        audio_job_manager,
        generation_job_manager,
        transcribe_jobs,
        shape3d_jobs,
        download_jobs,
    )
    retention_sweeper = RetentionSweeper(
        settings, job_manager, video_job_manager, audio_job_manager,
        generation_job_manager=generation_job_manager,
        transcribe_job_manager=transcribe_jobs,
        shape3d_job_manager=shape3d_jobs,
        download_job_manager=download_jobs,
    )
    await retention_sweeper.start()
    await transcribe_jobs.start()
    await karaoke_jobs.start()
    await shape3d_jobs.start()
    await download_jobs.start()

    app.state.storage = storage
    app.state.engine = engine
    app.state.media_tools = media_tools
    app.state.rife_engine = rife_engine
    app.state.gmfss_engine = gmfss_engine
    app.state.audio_enhancers = audio_enhancers
    app.state.restorers = restorers
    app.state.onnx_engine = onnx_engine
    app.state.onnx_video_engine = onnx_video_engine
    app.state.devices_service = devices_service
    # Las MISMAS instancias que gatean la admision de jobs, reusadas por el
    # pre-flight de modelos de generacion -- no se construyen sondas aparte.
    app.state.resource_probes = resource_probes
    app.state.capability_probe = capability_probe
    app.state.onnx_cpu_fallback_probe = onnx_cpu_fallback_probe
    app.state.job_manager = job_manager
    app.state.video_job_manager = video_job_manager
    app.state.audio_job_manager = audio_job_manager
    app.state.retention_sweeper = retention_sweeper
    app.state.model_registry = model_registry
    app.state.update_service = update_service
    app.state.hf_client = hf_client
    app.state.model_installer = model_installer
    from app.services.editor_segmenter import EditorSegmenter

    from app.services.vulkan_installer import VulkanModelInstaller

    vulkan_installer = VulkanModelInstaller(settings, hf_client)
    await vulkan_installer.start()
    app.state.vulkan_installer = vulkan_installer
    app.state.editor_segmenter = EditorSegmenter(settings)
    app.state.generation_job_manager = generation_job_manager
    app.state.generation_installer = generation_installer
    app.state.generation_converter = generation_converter
    app.state.pack_provisioner = pack_provisioner
    app.state.asr_installer = asr_installer
    app.state.transcribe_engine = transcribe_engine
    app.state.transcribe_jobs = transcribe_jobs
    app.state.karaoke_jobs = karaoke_jobs
    app.state.shape3d_jobs = shape3d_jobs
    app.state.download_jobs = download_jobs
    app.state.user_store = user_store
    app.state.identity_provider = identity_provider
    app.state.quota_service = quota_service
    try:
        yield
    finally:
        await job_manager.stop()
        await video_job_manager.stop()
        await audio_job_manager.stop()
        await retention_sweeper.stop()
        await model_installer.stop()
        await generation_job_manager.stop()
        await generation_installer.stop()
        await generation_converter.stop()
        await pack_provisioner.stop()
        await asr_installer.stop()
        await transcribe_jobs.stop()
        await karaoke_jobs.stop()
        await shape3d_jobs.stop()
        await download_jobs.stop()


def _serve_index(index_path: Path) -> Response:
    if not index_path.exists():
        return PlainTextResponse(
            "Frontend build no encontrado. Ejecuta: cd frontend && npm run build",
            status_code=503,
        )
    return FileResponse(index_path)


def configure_web_routes(app: FastAPI, frontend_dist: Path = FRONTEND_DIST_DIR) -> None:
    """Serves the built React SPA as the only UI: static assets plus an
    index.html fallback for every other non-API path, so client-side routes
    (e.g. /models) resolve on a hard refresh instead of 404ing.

    frontend_dist must already contain a `npm run build` output (index.html
    + assets/) — the release zip and local dev builds both produce this
    before the server starts; there is no legacy fallback anymore.
    """
    assets_dir = frontend_dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="spa-assets")
    index_path = frontend_dist / "index.html"

    @app.get("/", include_in_schema=False)
    async def spa_index() -> Response:
        return _serve_index(index_path)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> Response:
        # An unmatched /api/* path is a real 404 (renamed/removed endpoint),
        # not a client-side route — serving index.html would hand a stale
        # frontend HTML where it expects JSON, masking the error. Wrong
        # methods on existing endpoints still 405 via the api_router.
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        return _serve_index(index_path)


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(OriginGuardMiddleware, allowed_origins=settings.allowed_origin_values)
app.add_middleware(LoopbackGuardMiddleware, auth_mode=settings.auth_mode)
app.include_router(api_router)
app.include_router(capability_router)
app.include_router(editor_router)
app.include_router(auth_router)
app.include_router(users_router)
configure_web_routes(app)
