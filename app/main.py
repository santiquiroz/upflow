from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import AUDIO_ENHANCE_MODES, get_settings
from app.security import OriginGuardMiddleware
from app.services.devices_service import DevicesService
from app.services.engines.audio_enhance import AudioEnhancer
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
from app.web.routes import router as web_router

APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    storage = StorageService(settings)
    engine = RealEsrganNcnnEngine(settings)
    media_tools = MediaTools(settings)
    rife_engine = RifeNcnnEngine(settings)
    audio_enhancers = {mode: AudioEnhancer(settings, mode) for mode in AUDIO_ENHANCE_MODES}
    devices_service = DevicesService(settings)
    gpu_semaphore = asyncio.Semaphore(settings.gpu_concurrency)
    job_manager = JobManager(settings, engine, gpu_semaphore)
    video_upscaler = VideoUpscaler(settings, engine, media_tools, rife_engine, audio_enhancers)
    video_job_manager = VideoJobManager(settings, video_upscaler, media_tools, gpu_semaphore)
    retention_sweeper = RetentionSweeper(settings, job_manager, video_job_manager)
    model_registry = ModelRegistry(settings)
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


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(OriginGuardMiddleware, allowed_origins=settings.allowed_origin_values)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
app.include_router(api_router)
app.include_router(web_router)
