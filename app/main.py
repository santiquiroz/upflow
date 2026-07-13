from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import get_settings
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.job_manager import JobManager
from app.services.media_tools import MediaTools
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager
from app.services.video_upscaler import VideoUpscaler
from app.web.routes import router as web_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    storage = StorageService(settings)
    engine = RealEsrganNcnnEngine(settings)
    media_tools = MediaTools(settings)
    job_manager = JobManager(settings, engine)
    video_upscaler = VideoUpscaler(settings, engine, media_tools)
    video_job_manager = VideoJobManager(settings, video_upscaler, media_tools)
    await job_manager.start()
    await video_job_manager.start()

    app.state.storage = storage
    app.state.engine = engine
    app.state.media_tools = media_tools
    app.state.job_manager = job_manager
    app.state.video_job_manager = video_job_manager
    try:
        yield
    finally:
        await job_manager.stop()
        await video_job_manager.stop()


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(api_router)
app.include_router(web_router)
