from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import AUDIO_ENHANCE_MODES, get_settings
from app.security import OriginGuardMiddleware
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import DevicesService
from app.services.engines.audio_enhance import AudioEnhancer
from app.services.engines.onnx_upscaler import OnnxUpscaler
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.engines.rife_ncnn import RifeNcnnEngine
from app.services.hf_client import HfClient
from app.services.job_manager import JobManager
from app.services.media_tools import MediaTools
from app.services.model_installer import ModelInstaller
from app.services.model_registry import ModelRegistry
from app.services.retention_sweeper import RetentionSweeper
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager
from app.services.video_upscaler import VideoUpscaler

APP_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = APP_DIR.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    storage = StorageService(settings)
    engine = RealEsrganNcnnEngine(settings)
    media_tools = MediaTools(settings)
    rife_engine = RifeNcnnEngine(settings)
    audio_enhancers = {mode: AudioEnhancer(settings, mode) for mode in AUDIO_ENHANCE_MODES}
    devices_service = DevicesService(settings)
    model_registry = ModelRegistry(settings)
    onnx_engine = OnnxUpscaler(settings, model_registry, devices_service)
    device_semaphores = DeviceSemaphores(settings)
    job_manager = JobManager(
        settings,
        engine,
        device_semaphores,
        onnx_engine=onnx_engine,
        registry=model_registry,
        devices=devices_service,
    )
    video_upscaler = VideoUpscaler(
        settings,
        engine,
        media_tools,
        rife_engine,
        audio_enhancers,
        onnx_engine=onnx_engine,
        model_registry=model_registry,
    )
    video_job_manager = VideoJobManager(
        settings,
        video_upscaler,
        media_tools,
        device_semaphores,
        registry=model_registry,
        devices=devices_service,
    )
    retention_sweeper = RetentionSweeper(settings, job_manager, video_job_manager)
    hf_client = HfClient(settings)
    model_installer = ModelInstaller(settings, model_registry, hf_client)
    await job_manager.start()
    await video_job_manager.start()
    await retention_sweeper.start()
    await model_installer.start()

    app.state.storage = storage
    app.state.engine = engine
    app.state.media_tools = media_tools
    app.state.rife_engine = rife_engine
    app.state.audio_enhancers = audio_enhancers
    app.state.onnx_engine = onnx_engine
    app.state.devices_service = devices_service
    app.state.job_manager = job_manager
    app.state.video_job_manager = video_job_manager
    app.state.retention_sweeper = retention_sweeper
    app.state.model_registry = model_registry
    app.state.hf_client = hf_client
    app.state.model_installer = model_installer
    try:
        yield
    finally:
        await job_manager.stop()
        await video_job_manager.stop()
        await retention_sweeper.stop()
        await model_installer.stop()


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
app.include_router(api_router)
configure_web_routes(app)
