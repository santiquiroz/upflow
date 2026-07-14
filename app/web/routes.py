from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import get_settings

APP_DIR = Path(__file__).resolve().parent.parent

router = APIRouter(tags=["web"])
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    settings = get_settings()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": settings.web_title,
            "default_model": settings.default_model,
            "default_video_profile": settings.default_video_profile,
            "allowed_scales": settings.allowed_scale_values,
            "allowed_fps_multipliers": settings.allowed_fps_multiplier_values,
            "supported_models": settings.model_catalog,
            "video_profiles": settings.video_profile_catalog,
            "engine": settings.engine,
        },
    )
