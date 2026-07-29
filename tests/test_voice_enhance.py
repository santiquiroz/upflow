from __future__ import annotations

from pathlib import Path

import pytest

import app.services.engines.voice_enhance as voice_enhance_module
from app.config import Settings
from app.services.engines.voice_enhance import VoiceEnhancer
from app.services.voice_chain import ChainStep


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


def _filter(step_id: str, expr: str) -> ChainStep:
    return ChainStep(id=step_id, kind="filter", label_key=step_id, filter_expr=expr)


def _model(step_id: str, capability: str) -> ChainStep:
    return ChainStep(id=step_id, kind="model", label_key=step_id, model_capability=capability)


class FakeFfmpeg:
    """Sustituye run_guarded_process: registra los comandos y escribe la salida."""

    def __init__(self, returncode: int = 0, stderr: bytes = b"") -> None:
        self.commands: list[list[str]] = []
        self.returncode = returncode
        self.stderr = stderr

    async def __call__(self, command, timeout):
        self.commands.append(list(command))
        if self.returncode == 0:
            Path(command[-1]).write_bytes(b"wav")
        return b"", self.stderr, self.returncode

    @property
    def filter_args(self) -> list[str]:
        return [c[c.index("-af") + 1] for c in self.commands if "-af" in c]


@pytest.fixture
def ffmpeg(monkeypatch) -> FakeFfmpeg:
    fake = FakeFfmpeg()
    monkeypatch.setattr(voice_enhance_module, "run_guarded_process", fake)
    return fake


@pytest.mark.asyncio
async def test_consecutive_filters_run_as_a_single_ffmpeg_pass(tmp_path, ffmpeg):
    source = tmp_path / "in.wav"
    source.write_bytes(b"wav")
    enhancer = VoiceEnhancer(make_settings(tmp_path))

    await enhancer.run(
        [
            _filter("highpass", "highpass=f=80"),
            _filter("compress", "acompressor"),
            _filter("deesser", "deesser"),
        ],
        source,
        tmp_path / "out.wav",
        tmp_path,
    )

    # Una sola pasada: cada pasada extra seria un decode y encode completo.
    assert len(ffmpeg.commands) == 1
    assert ffmpeg.filter_args == ["highpass=f=80,acompressor,deesser"]


@pytest.mark.asyncio
async def test_the_last_stage_writes_the_requested_output_path(tmp_path, ffmpeg):
    source = tmp_path / "in.wav"
    source.write_bytes(b"wav")
    output = tmp_path / "final.wav"

    await VoiceEnhancer(make_settings(tmp_path)).run(
        [_filter("deesser", "deesser")], source, output, tmp_path
    )

    assert ffmpeg.commands[0][-1] == str(output)
    assert output.exists()


@pytest.mark.asyncio
async def test_a_model_stage_splits_the_run_and_receives_the_intermediate(tmp_path, ffmpeg):
    source = tmp_path / "in.wav"
    source.write_bytes(b"wav")
    seen: list[tuple[str, str, str]] = []

    async def runner(capability: str, src: Path, dst: Path) -> None:
        seen.append((capability, src.name, dst.name))
        dst.write_bytes(b"wav")

    enhancer = VoiceEnhancer(make_settings(tmp_path), model_stage_runner=runner)
    await enhancer.run(
        [
            _filter("highpass", "highpass=f=80"),
            _model("isolate", "audio.voice.isolate"),
            _filter("deesser", "deesser"),
        ],
        source,
        tmp_path / "out.wav",
        tmp_path,
    )

    assert len(ffmpeg.commands) == 2
    assert [c for c, _s, _d in seen] == ["audio.voice.isolate"]
    # El modelo recibe la salida de la pasada anterior, no el archivo original.
    assert seen[0][1] != source.name


@pytest.mark.asyncio
async def test_a_model_stage_without_a_runner_fails_with_a_clear_message(tmp_path, ffmpeg):
    source = tmp_path / "in.wav"
    source.write_bytes(b"wav")

    with pytest.raises(RuntimeError, match="resolutor"):
        await VoiceEnhancer(make_settings(tmp_path)).run(
            [_model("isolate", "audio.voice.isolate")], source, tmp_path / "out.wav", tmp_path
        )


@pytest.mark.asyncio
async def test_an_empty_chain_copies_the_input_through(tmp_path, ffmpeg):
    source = tmp_path / "in.wav"
    source.write_bytes(b"contenido")
    output = tmp_path / "out.wav"

    await VoiceEnhancer(make_settings(tmp_path)).run([], source, output, tmp_path)

    # El caller espera el archivo: no correr nada no puede significar no
    # producir salida.
    assert ffmpeg.commands == []
    assert output.read_bytes() == b"contenido"


@pytest.mark.asyncio
async def test_an_ffmpeg_failure_names_the_steps_that_failed(tmp_path, monkeypatch):
    fake = FakeFfmpeg(returncode=1, stderr=b"Invalid argument")
    monkeypatch.setattr(voice_enhance_module, "run_guarded_process", fake)
    source = tmp_path / "in.wav"
    source.write_bytes(b"wav")

    with pytest.raises(RuntimeError) as exc_info:
        await VoiceEnhancer(make_settings(tmp_path)).run(
            [_filter("deesser", "deesser"), _filter("compress", "acompressor")],
            source,
            tmp_path / "out.wav",
            tmp_path,
        )

    message = str(exc_info.value)
    assert "deesser" in message and "compress" in message
    assert "Invalid argument" in message


@pytest.mark.asyncio
async def test_a_stage_that_produces_nothing_is_reported(tmp_path, monkeypatch):
    class SilentFfmpeg(FakeFfmpeg):
        async def __call__(self, command, timeout):
            self.commands.append(list(command))
            return b"", b"", 0  # exit 0 pero sin escribir nada

    monkeypatch.setattr(voice_enhance_module, "run_guarded_process", SilentFfmpeg())
    source = tmp_path / "in.wav"
    source.write_bytes(b"wav")

    with pytest.raises(RuntimeError, match="no produjo"):
        await VoiceEnhancer(make_settings(tmp_path)).run(
            [_filter("deesser", "deesser")], source, tmp_path / "out.wav", tmp_path
        )


@pytest.mark.asyncio
async def test_a_model_stage_that_produces_nothing_is_reported(tmp_path, ffmpeg):
    async def runner(capability, src, dst):
        return None  # no escribe nada

    source = tmp_path / "in.wav"
    source.write_bytes(b"wav")

    with pytest.raises(RuntimeError, match="no produjo"):
        await VoiceEnhancer(make_settings(tmp_path), model_stage_runner=runner).run(
            [_model("isolate", "audio.voice.isolate")], source, tmp_path / "out.wav", tmp_path
        )


@pytest.mark.asyncio
async def test_intermediates_stay_inside_the_work_dir(tmp_path, ffmpeg):
    source = tmp_path / "in.wav"
    source.write_bytes(b"wav")
    work = tmp_path / "work"
    work.mkdir()

    async def runner(capability, src, dst):
        dst.write_bytes(b"wav")

    await VoiceEnhancer(make_settings(tmp_path), model_stage_runner=runner).run(
        [
            _filter("highpass", "highpass=f=80"),
            _model("isolate", "audio.voice.isolate"),
            _filter("deesser", "deesser"),
        ],
        source,
        tmp_path / "out.wav",
        work,
    )

    # Los intermedios van al work_dir del job, que el pipeline limpia en su
    # finally: nada queda tirado fuera de ahi.
    assert sorted(p.name for p in work.iterdir()) == ["voice-0.wav", "voice-1.wav"]
