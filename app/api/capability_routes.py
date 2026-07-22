from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import CapabilitiesResponse, FixLeverResponse, LeverResponse
from app.services.capability_probe import CapabilityProbe, Lever

router = APIRouter(prefix="/api/v1/capabilities", tags=["capabilities"])


def get_capability_probe(request: Request) -> CapabilityProbe:
    return request.app.state.capability_probe


def _to_response(lever: Lever) -> LeverResponse:
    return LeverResponse(id=lever.id, label=lever.label, status=lever.status, detail=lever.detail, fixable=lever.fixable)


@router.get("", response_model=CapabilitiesResponse)
async def get_capabilities(probe: CapabilityProbe = Depends(get_capability_probe)) -> CapabilitiesResponse:
    levers = await probe.list_levers()
    return CapabilitiesResponse(levers=[_to_response(lever) for lever in levers])


@router.post("/rescan", response_model=CapabilitiesResponse)
async def rescan_capabilities(probe: CapabilityProbe = Depends(get_capability_probe)) -> CapabilitiesResponse:
    levers = await probe.rescan()
    return CapabilitiesResponse(levers=[_to_response(lever) for lever in levers])


@router.post("/{lever_id}/fix", response_model=FixLeverResponse)
async def fix_lever(lever_id: str, probe: CapabilityProbe = Depends(get_capability_probe)) -> FixLeverResponse:
    try:
        lever = await probe.apply_fix(lever_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FixLeverResponse(lever=_to_response(lever))
