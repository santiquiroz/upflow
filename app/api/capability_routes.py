from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import (
    CapabilitiesResponse,
    CpuFallbackReportResponse,
    FixLeverResponse,
    LeverResponse,
    OnnxDiagnosticEntryResponse,
    OnnxDiagnosticsResponse,
    ScanOnnxDiagnosticResponse,
)
from app.services.capability_probe import CapabilityProbe, Lever
from app.services.onnx_cpu_fallback_probe import CpuFallbackReport, OnnxCpuFallbackProbe

router = APIRouter(prefix="/api/v1/capabilities", tags=["capabilities"])


def get_capability_probe(request: Request) -> CapabilityProbe:
    return request.app.state.capability_probe


def get_onnx_cpu_fallback_probe(request: Request) -> OnnxCpuFallbackProbe:
    return request.app.state.onnx_cpu_fallback_probe


def _to_response(lever: Lever) -> LeverResponse:
    return LeverResponse(id=lever.id, label=lever.label, status=lever.status, detail=lever.detail, fixable=lever.fixable)


def _report_to_response(report: CpuFallbackReport) -> CpuFallbackReportResponse:
    return CpuFallbackReportResponse(
        model_id=report.model_id, device_id=report.device_id, hot_ops=list(report.hot_ops), clean=report.clean
    )


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


@router.get("/onnx-diagnostics", response_model=OnnxDiagnosticsResponse)
async def get_onnx_diagnostics(
    probe: OnnxCpuFallbackProbe = Depends(get_onnx_cpu_fallback_probe),
) -> OnnxDiagnosticsResponse:
    entries = [
        OnnxDiagnosticEntryResponse(
            model_id=model_id,
            device_id=device_id,
            report=_report_to_response(cached) if (cached := probe.cached(model_id, device_id)) else None,
        )
        for model_id, device_id in probe.catalog()
    ]
    return OnnxDiagnosticsResponse(entries=entries)


@router.post("/onnx-diagnostics/{model_id}/{device_id}/scan", response_model=ScanOnnxDiagnosticResponse)
async def scan_onnx_diagnostic(
    model_id: str, device_id: str, probe: OnnxCpuFallbackProbe = Depends(get_onnx_cpu_fallback_probe)
) -> ScanOnnxDiagnosticResponse:
    try:
        report = await probe.scan(model_id, device_id)
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ScanOnnxDiagnosticResponse(report=_report_to_response(report))
