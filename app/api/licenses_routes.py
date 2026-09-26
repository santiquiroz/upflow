from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.schemas_restore import LicenseGateResponse, LicensesResponse
from app.services.license_gate import license_gate
from app.services.licenses_view import licenses_view
from app.services.pack_provisioner import UnknownPackError, script_for

router = APIRouter(prefix="/api/v1", tags=["licenses"])


@router.get("/licenses", response_model=LicensesResponse)
async def get_licenses(settings: Settings = Depends(get_settings)) -> LicensesResponse:
    # Lee THIRD_PARTY_NOTICES.md y los LICENSE de cada pack desde disco: fuera del event loop.
    view = await asyncio.to_thread(licenses_view, settings)
    return LicensesResponse.model_validate(view)


@router.get("/packs/{pack}/license", response_model=LicenseGateResponse)
async def get_pack_license(pack: str) -> LicenseGateResponse:
    try:
        script_for(pack)
    except UnknownPackError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    gate = await asyncio.to_thread(license_gate, pack)
    return LicenseGateResponse.model_validate(gate)
