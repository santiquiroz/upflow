from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.video_upscaler import VideoUpscaler


class _FakeEngine:
    def available(self) -> bool:
        return True


class _FakeMediaTools:
    def available(self) -> bool:
        return True


class _FakeDevices:
    def __init__(self, name: str) -> None:
        self._name = name

    def list_devices(self):
        return [
            {"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"},
            {"id": "dml:0", "kind": "gpu", "name": self._name, "backend": "directml"},
        ]


def make_upscaler(tmp_path: Path, device_name: str = "AMD Radeon RX 7800 XT") -> VideoUpscaler:
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    return VideoUpscaler(settings, _FakeEngine(), _FakeMediaTools(), devices=_FakeDevices(device_name))


def make_job(video_encoder: str, video_codec: str = "libx265", device: str = "dml:0") -> VideoUpscaleJob:
    return VideoUpscaleJob(
        source_path=Path("x.mp4"),
        original_filename="x.mp4",
        model_name="realesr-animevideov3-x4",
        scale=4,
        output_container="mp4",
        video_codec=video_codec,
        video_preset="medium",
        crf=20,
        keep_audio=False,
        device=device,
        video_encoder=video_encoder,
    )


def test_video_encoder_defaults_to_auto(tmp_path: Path) -> None:
    # Regression guard: the software default was the dominant wall-time cost
    # (x265 slow at 4x = ~112 min/episode vs ~16 min on the GPU). Default must
    # stay "auto"; the HW->software fallback keeps it safe.
    job = VideoUpscaleJob(
        source_path=Path("x.mp4"),
        original_filename="x.mp4",
        model_name="realesr-animevideov3-x4",
        scale=4,
        output_container="mp4",
        video_codec="libx265",
        video_preset="slow",
        crf=20,
        keep_audio=False,
    )
    assert job.video_encoder == "auto"
    vu = make_upscaler(tmp_path, "AMD Radeon RX 7800 XT")
    job.device = "dml:0"
    assert vu._resolve_video_encoder(job) == "hevc_amf"


def test_resolve_encoder_software_keeps_codec(tmp_path: Path) -> None:
    vu = make_upscaler(tmp_path)
    assert vu._resolve_video_encoder(make_job("software", "libx265")) == "libx265"
    assert vu._resolve_video_encoder(make_job("software", "libx264")) == "libx264"


def test_resolve_encoder_auto_maps_amd_gpu(tmp_path: Path) -> None:
    vu = make_upscaler(tmp_path, "AMD Radeon RX 7800 XT")
    assert vu._resolve_video_encoder(make_job("auto", "libx265")) == "hevc_amf"
    assert vu._resolve_video_encoder(make_job("auto", "libx264")) == "h264_amf"


def test_resolve_encoder_auto_maps_nvidia_gpu(tmp_path: Path) -> None:
    vu = make_upscaler(tmp_path, "NVIDIA GeForce RTX 4090")
    assert vu._resolve_video_encoder(make_job("auto", "libx265")) == "hevc_nvenc"


def test_resolve_encoder_auto_falls_back_to_software_on_unknown_gpu(tmp_path: Path) -> None:
    vu = make_upscaler(tmp_path, "Mystery Accelerator 9000")
    assert vu._resolve_video_encoder(make_job("auto", "libx265")) == "libx265"


def test_resolve_encoder_auto_falls_back_when_device_is_cpu(tmp_path: Path) -> None:
    vu = make_upscaler(tmp_path)
    assert vu._resolve_video_encoder(make_job("auto", "libx264", device="cpu")) == "libx264"


async def test_encode_with_fallback_retries_software_when_hw_fails(tmp_path: Path) -> None:
    vu = make_upscaler(tmp_path)
    job = make_job("auto", "libx265")
    calls: list[str] = []

    async def fake_run(cmd: list[str]) -> None:
        encoder = cmd[cmd.index("-c:v") + 1]
        calls.append(encoder)
        if encoder == "hevc_amf":
            raise RuntimeError("AMF init failed")
        # software succeeds

    vu._run_process = fake_run  # type: ignore[method-assign]
    out = tmp_path / "out.mp4"
    await vu._encode_with_fallback(job, tmp_path, "24/1", None, [], out, "hevc_amf")

    assert calls == ["hevc_amf", "libx265"]  # tried HW, then fell back to software
    assert job.metadata["videoEncoderFallback"] == "hevc_amf"
    assert job.metadata["videoEncoder"] == "libx265"


async def test_encode_software_failure_propagates_without_retry(tmp_path: Path) -> None:
    vu = make_upscaler(tmp_path)
    job = make_job("software", "libx264")
    calls: list[str] = []

    async def fake_run(cmd: list[str]) -> None:
        calls.append(cmd[cmd.index("-c:v") + 1])
        raise RuntimeError("disk full")

    vu._run_process = fake_run  # type: ignore[method-assign]
    out = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="disk full"):
        await vu._encode_with_fallback(job, tmp_path, "24/1", None, [], out, "libx264")
    assert calls == ["libx264"]  # no fallback for a software failure
