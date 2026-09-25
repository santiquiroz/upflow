from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas_restore import LicensesResponse
from app.services.licenses_view import licenses_view

router = APIRouter(prefix="/api/v1", tags=["licenses"])


@router.get("/licenses", response_model=LicensesResponse)
async def get_licenses(settings: Settings = Depends(get_settings)) -> LicensesResponse:
    # Lee THIRD_PARTY_NOTICES.md y los LICENSE de cada pack desde disco: fuera del event loop.
    view = await asyncio.to_thread(licenses_view, settings)
    return LicensesResponse.model_validate(view)
