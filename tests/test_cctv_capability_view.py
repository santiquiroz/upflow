from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.capabilities import HostProbes
from app.services.cctv_capability_view import cctv_capability_view
from app.services.ffmpeg_capabilities import (
    FfmpegCapabilities,
    FfmpegProbeError,
    parse_buildconf,
    parse_encoders,
    parse_filters,
    parse_version,
    unavailable_steps,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ffmpeg"


def fixture_caps(build: str) -> FfmpegCapabilities:
    def read(kind: str) -> str:
        return (FIXTURES / f"{kind}_{build}.txt").read_text(encoding="utf-8")

    return FfmpegCapabilities(
        binary_sha256="0" * 64,
        version=parse_version(read("version")),
        configuration=parse_buildconf(read("buildconf")),
        filters=parse_filters(read("filters")),
        encoders=parse_encoders(read("encoders")),
        cpu_extensions=(),
    )


def without_filters(caps: FfmpegCapabilities) -> FfmpegCapabilities:
    return FfmpegCapabilities(
        caps.binary_sha256, caps.version, caps.configuration, frozenset(), caps.encoders, ()
    )


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None, **overrides)


def with_ffmpeg(tmp_path: Path) -> Settings:
    binary = tmp_path / "ffmpeg.exe"
    binary.write_bytes(b"x")
    return make_settings(tmp_path, FFMPEG_BINARY=str(binary))


def probes(*, build: bool = True, gpu: bool = True) -> HostProbes:
    return HostProbes(ffmpeg_cctv_build=lambda _s: build, dml_gpu=lambda _s: gpu)


def loader(caps: FfmpegCapabilities):
    return lambda _binary: caps


def failing_loader(_binary: Path) -> FfmpegCapabilities:
    raise FfmpegProbeError("ffmpeg not found")


def test_sin_ffmpeg_ningun_carril_esta_disponible(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, FFMPEG_BINARY=str(tmp_path / "no-ffmpeg.exe"))

    view = cctv_capability_view(settings, probes(), failing_loader)

    assert view.available is False
    assert view.reason_key == "capability.setup.missingPack"
    assert view.ai_available is False
    assert view.ai_reason_key == "capability.setup.missingPack"
    assert view.unavailable_steps == ()


def test_con_una_build_completa_el_modo_clasico_esta_disponible(tmp_path: Path) -> None:
    view = cctv_capability_view(with_ffmpeg(tmp_path), probes(), loader(fixture_caps("gpl")))

    assert view.available is True
    assert view.reason_key is None
    assert view.unavailable_steps == ()


def test_sin_libx264_el_modo_no_esta_disponible(tmp_path: Path) -> None:
    view = cctv_capability_view(
        with_ffmpeg(tmp_path), probes(build=False), loader(fixture_caps("lgpl"))
    )

    assert view.available is False
    assert view.reason_key == "capability.setup.ffmpegBuildLacksCctv"


def test_los_pasos_sin_filtro_en_la_build_se_listan(tmp_path: Path) -> None:
    caps = without_filters(fixture_caps("gpl"))

    view = cctv_capability_view(with_ffmpeg(tmp_path), probes(), loader(caps))

    assert view.available is True
    assert view.unavailable_steps == unavailable_steps(caps)
    assert view.unavailable_steps


def test_el_carril_ia_sin_restore_core_dice_que_falta_el_pack(tmp_path: Path) -> None:
    view = cctv_capability_view(with_ffmpeg(tmp_path), probes(), loader(fixture_caps("gpl")))

    assert view.ai_available is False
    assert view.ai_reason_key == "capability.setup.missingPack"


def test_el_carril_ia_sin_gpu_dice_que_la_necesita(tmp_path: Path, monkeypatch) -> None:
    import app.services.capabilities as cap_mod

    monkeypatch.setattr(cap_mod, "_path_exists", lambda _settings, _requirement: True)

    view = cctv_capability_view(
        make_settings(tmp_path), probes(gpu=False), loader(fixture_caps("gpl"))
    )

    assert view.available is True
    assert view.ai_available is False
    assert view.ai_reason_key == "capability.setup.needsGpu"


def test_el_carril_ia_con_todo_esta_disponible(tmp_path: Path, monkeypatch) -> None:
    import app.services.capabilities as cap_mod

    monkeypatch.setattr(cap_mod, "_path_exists", lambda _settings, _requirement: True)

    view = cctv_capability_view(make_settings(tmp_path), probes(), loader(fixture_caps("gpl")))

    assert view.ai_available is True
    assert view.ai_reason_key is None


class FakeDevices:
    def __init__(self, devices: list[dict], unhealthy: frozenset[str] = frozenset()) -> None:
        self._devices = devices
        self._unhealthy = unhealthy

    def list_devices(self) -> list[dict]:
        return self._devices

    def is_healthy(self, device_id: str) -> bool:
        return device_id not in self._unhealthy


@pytest.mark.asyncio
async def test_la_ruta_de_video_expone_el_modo_cctv(tmp_path: Path) -> None:
    from app.api.routes import video_capabilities

    settings = make_settings(tmp_path, FFMPEG_BINARY=str(tmp_path / "no-ffmpeg.exe"))

    response = await video_capabilities(settings=settings, devices=FakeDevices([]))
    body = response.model_dump(by_alias=True)

    assert body["cctvAvailable"] is False
    assert body["cctvReasonKey"] == "capability.setup.missingPack"
    assert body["cctvAiAvailable"] is False
    assert body["cctvAiReasonKey"] == "capability.setup.missingPack"
    assert body["cctvUnavailableSteps"] == []


def fixture_cache(build: str):
    from app.services import ffmpeg_capabilities as ffcaps

    kinds = {
        ffcaps.VERSION_ARGS: "version",
        ffcaps.BUILDCONF_ARGS: "buildconf",
        ffcaps.FILTERS_ARGS: "filters",
        ffcaps.ENCODERS_ARGS: "encoders",
    }

    def run(_binary: Path, args) -> str:
        return (FIXTURES / f"{kinds[tuple(args)]}_{build}.txt").read_text(encoding="utf-8")

    tools = ffcaps.ProbeTools(run=run, hash_file=lambda _p: "0" * 64, cpu_extensions=lambda: ())
    return ffcaps.FfmpegCapabilityCache(tools)


@pytest.mark.asyncio
async def test_la_ruta_usa_la_salud_de_los_devices_de_la_app(tmp_path: Path, monkeypatch) -> None:
    import app.services.capabilities as cap_mod
    from app.api.routes import video_capabilities

    monkeypatch.setattr(cap_mod, "_path_exists", lambda _settings, _requirement: True)
    monkeypatch.setattr("app.services.ffmpeg_capabilities._PROCESS_CACHE", fixture_cache("gpl"))
    removida = FakeDevices([{"id": "dml:0", "backend": "directml"}], frozenset({"dml:0"}))

    response = await video_capabilities(settings=with_ffmpeg(tmp_path), devices=removida)

    assert response.cctv_available is True
    assert response.cctv_ai_available is False
    assert response.cctv_ai_reason_key == "capability.setup.needsGpu"


@pytest.mark.asyncio
async def test_la_ruta_con_una_build_lgpl_dice_por_que_no(tmp_path: Path, monkeypatch) -> None:
    from app.api.routes import video_capabilities

    monkeypatch.setattr("app.services.ffmpeg_capabilities._PROCESS_CACHE", fixture_cache("lgpl"))

    response = await video_capabilities(settings=with_ffmpeg(tmp_path), devices=FakeDevices([]))

    assert response.cctv_available is False
    assert response.cctv_reason_key == "capability.setup.ffmpegBuildLacksCctv"
    assert response.cctv_unavailable_steps == ["stabilize"]
