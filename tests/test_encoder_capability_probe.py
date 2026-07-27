from __future__ import annotations

from pathlib import Path

from app.services import video_encoders
from app.services.encoder_capability_probe import EncoderCapabilityProbe


class FakeRunner:
    """Sustituye la corrida real de ffmpeg: registra que se probó y decide."""

    def __init__(self, fails: set[tuple[str, int, int]] | None = None) -> None:
        self.fails = fails or set()
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, encoder: str, width: int, height: int) -> bool:
        self.calls.append((encoder, width, height))
        return (encoder, width, height) not in self.fails


def make_probe(runner: FakeRunner) -> EncoderCapabilityProbe:
    return EncoderCapabilityProbe(Path("ffmpeg.exe"), runner=runner)


def test_supports_returns_true_when_the_trial_encode_succeeds() -> None:
    probe = make_probe(FakeRunner())
    assert probe.supports("hevc_amf", 7680, 4320) is True


def test_supports_returns_false_when_the_trial_encode_fails() -> None:
    # h264_amf revienta a 8K con 'encoder->Init() failed with error 5' (medido
    # en una RX 7800 XT); hevc_amf y av1_amf a la misma resolución andan.
    runner = FakeRunner(fails={("h264_amf", 7680, 4320)})
    probe = make_probe(runner)
    assert probe.supports("h264_amf", 7680, 4320) is False
    assert probe.supports("hevc_amf", 7680, 4320) is True


def test_result_is_cached_per_encoder_and_resolution() -> None:
    runner = FakeRunner()
    probe = make_probe(runner)

    probe.supports("hevc_amf", 3840, 2160)
    probe.supports("hevc_amf", 3840, 2160)
    probe.supports("hevc_amf", 7680, 4320)

    # Una prueba por combinación: el mismo encoder a otra resolución sí se
    # reevalúa, porque el límite de resolución es justo lo que se busca.
    assert runner.calls == [
        ("hevc_amf", 3840, 2160),
        ("hevc_amf", 7680, 4320),
    ]


def test_a_crashing_runner_is_treated_as_unsupported() -> None:
    def boom(encoder: str, width: int, height: int) -> bool:
        raise OSError("ffmpeg no está")

    probe = EncoderCapabilityProbe(Path("ffmpeg.exe"), runner=boom)
    # Nunca debe tumbar el job: sin certeza, el caller cae a software.
    assert probe.supports("hevc_nvenc", 1920, 1080) is False


def test_software_encoders_are_assumed_supported_without_probing() -> None:
    runner = FakeRunner()
    probe = make_probe(runner)

    assert probe.supports("libx264", 7680, 4320) is True
    assert probe.supports("libx265", 7680, 4320) is True
    assert runner.calls == [], "no tiene sentido probar encoders de software"


# --- AV1 en las tres plataformas -------------------------------------------


def test_av1_hardware_encoder_per_vendor() -> None:
    assert video_encoders.resolve_hardware_encoder("AMD Radeon RX 7800 XT", "libsvtav1") == "av1_amf"
    assert video_encoders.resolve_hardware_encoder("NVIDIA GeForce RTX 4070", "libsvtav1") == "av1_nvenc"
    assert video_encoders.resolve_hardware_encoder("Intel Arc A770", "libsvtav1") == "av1_qsv"


def test_av1_encode_options_exist_for_every_vendor_and_software() -> None:
    for encoder in ("av1_amf", "av1_nvenc", "av1_qsv", "libsvtav1"):
        options = video_encoders.encode_options(
            encoder=encoder, crf=20, preset="medium", x265_pools=4, software_threads=8
        )
        assert options[:2] == ["-c:v", encoder]
        assert "-pix_fmt" in options


def test_codec_family_recognises_av1() -> None:
    assert video_encoders.codec_family("libsvtav1") == "av1"


# --- integración con la selección de encoder del pipeline -------------------


def test_auto_falls_back_to_software_when_the_gpu_encoder_cannot_do_the_size(tmp_path: Path) -> None:
    # El caso medido: 1080p x4 = 7680x4320 con h264. Antes se elegía h264_amf,
    # fallaba al encodear y recién ahí _encode_with_fallback rehacía todo por
    # software: minutos convertidos en horas, en silencio.
    from tests.test_video_upscaler import make_stream_job, make_stream_upscaler

    upscaler = make_stream_upscaler(tmp_path)
    upscaler.devices = _FakeDevices("AMD Radeon RX 7800 XT")
    upscaler.encoder_probe = EncoderCapabilityProbe(
        Path("ffmpeg.exe"), runner=FakeRunner(fails={("h264_amf", 7680, 4320)})
    )
    job = make_stream_job(tmp_path, video_encoder="auto", video_codec="libx264", device="dml:0")

    assert upscaler._resolve_video_encoder(job, 7680, 4320) == "libx264"


def test_auto_keeps_the_gpu_encoder_when_the_size_is_supported(tmp_path: Path) -> None:
    from tests.test_video_upscaler import make_stream_job, make_stream_upscaler

    upscaler = make_stream_upscaler(tmp_path)
    upscaler.devices = _FakeDevices("AMD Radeon RX 7800 XT")
    upscaler.encoder_probe = EncoderCapabilityProbe(Path("ffmpeg.exe"), runner=FakeRunner())
    job = make_stream_job(tmp_path, video_encoder="auto", video_codec="libx265", device="dml:0")

    assert upscaler._resolve_video_encoder(job, 7680, 4320) == "hevc_amf"


def test_explicit_software_never_probes(tmp_path: Path) -> None:
    from tests.test_video_upscaler import make_stream_job, make_stream_upscaler

    runner = FakeRunner()
    upscaler = make_stream_upscaler(tmp_path)
    upscaler.devices = _FakeDevices("AMD Radeon RX 7800 XT")
    upscaler.encoder_probe = EncoderCapabilityProbe(Path("ffmpeg.exe"), runner=runner)
    job = make_stream_job(tmp_path, video_encoder="software", video_codec="libx265", device="dml:0")

    assert upscaler._resolve_video_encoder(job, 7680, 4320) == "libx265"
    assert runner.calls == []


class _FakeDevices:
    def __init__(self, name: str) -> None:
        self._name = name

    def list_devices(self) -> list[dict[str, str]]:
        return [{"id": "dml:0", "name": self._name}]
