"""Payload de `GET /api/v1/video/cctv/presets` (spec §5.5): presets, pasos por carril con
sus esquemas y la disponibilidad de cada filtro en la build de ffmpeg (§4.2 paso 9)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.services.cctv_chain import LANES, Lane, catalog_schema
from app.services.cctv_presets import presets_schema
from app.services.ffmpeg_capabilities import (
    FfmpegCapabilities,
    UnavailableFilter,
    mode_unavailable_reason,
    unavailable_filters,
    unavailable_steps,
)

MissingByFilter = Mapping[tuple[str, str], UnavailableFilter]


def missing_by_filter(caps: FfmpegCapabilities) -> MissingByFilter:
    return {(item.step_id, item.filter): item for item in unavailable_filters(caps)}


def filter_with_availability(step_id: str, schema: dict[str, Any], missing: MissingByFilter) -> dict[str, Any]:
    item = missing.get((step_id, schema["name"]))
    return {
        **schema,
        "available": item is None,
        "unavailableReasonKey": None if item is None else item.reason_key,
        "unavailableReason": None if item is None else item.message,
    }


def step_with_availability(schema: dict[str, Any], missing: MissingByFilter) -> dict[str, Any]:
    filters = [filter_with_availability(schema["id"], spec, missing) for spec in schema["filters"]]
    return {**schema, "filters": filters, "available": any(spec["available"] for spec in filters)}


def lane_steps(lane: Lane, missing: MissingByFilter) -> list[dict[str, Any]]:
    return [step_with_availability(schema, missing) for schema in catalog_schema(lane)]


def ffmpeg_summary(caps: FfmpegCapabilities) -> dict[str, Any]:
    return {"version": caps.version, "gpl": caps.is_gpl, "binarySha256": caps.binary_sha256}


def presets_payload(caps: FfmpegCapabilities) -> dict[str, Any]:
    missing = missing_by_filter(caps)
    reason = mode_unavailable_reason(caps)
    return {
        "modeAvailable": reason is None,
        "modeUnavailableReason": reason,
        "ffmpeg": ffmpeg_summary(caps),
        "presets": presets_schema(),
        "steps": {lane: lane_steps(lane, missing) for lane in LANES},
        "unavailableSteps": list(unavailable_steps(caps)),
        "unavailableFilters": [item.to_json() for item in unavailable_filters(caps)],
    }
