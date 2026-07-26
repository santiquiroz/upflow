from __future__ import annotations

import io

import numpy as np
import pytest

from app.services.engines.ffmpeg_frame_sink import RawPipeEncoder


class FakeStdin(io.BytesIO):
    """BytesIO cuyo close() solo marca la bandera: el buffer sigue legible
    para que el test inspeccione los bytes escritos."""

    def __init__(self) -> None:
        super().__init__()
        self.closed_by_encoder = False

    def close(self) -> None:  # type: ignore[override]
        self.closed_by_encoder = True


class FakeEncodeProc:
    def __init__(self, returncode: int = 0) -> None:
        self.stdin = FakeStdin()
        self.stderr = io.BytesIO(b"ffmpeg noise\nultima linea util")
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self._final_returncode
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self._final_returncode = -9


def make_encoder(
    monkeypatch: pytest.MonkeyPatch, returncode: int = 0
) -> tuple[RawPipeEncoder, FakeEncodeProc]:
    encoder = RawPipeEncoder(["ffmpeg.exe", "-fake"])
    fake = FakeEncodeProc(returncode)
    monkeypatch.setattr(encoder, "_spawn", lambda command: fake)
    encoder.start()
    return encoder, fake


def test_write_frame_pipes_raw_bytes_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, fake = make_encoder(monkeypatch)
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    encoder.write_frame(frame)
    encoder.write_frame(frame)

    assert encoder.frames_written == 2
    assert fake.stdin.getvalue() == frame.tobytes() * 2


def test_finish_closes_stdin_and_passes_on_exit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, fake = make_encoder(monkeypatch)
    encoder.finish()
    assert fake.stdin.closed_by_encoder is True


def test_finish_raises_with_stderr_tail_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, fake = make_encoder(monkeypatch, returncode=1)
    with pytest.raises(RuntimeError, match="ultima linea util"):
        encoder.finish()


def test_finish_uses_injected_summarizer(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = RawPipeEncoder(["ffmpeg.exe"], summarize_error=lambda stderr: "mensaje amigable")
    fake = FakeEncodeProc(returncode=1)
    monkeypatch.setattr(encoder, "_spawn", lambda command: fake)
    encoder.start()
    with pytest.raises(RuntimeError, match="mensaje amigable"):
        encoder.finish()


def test_kill_kills_live_process_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, fake = make_encoder(monkeypatch)
    encoder.kill()
    encoder.kill()  # segunda llamada: no debe lanzar
    assert fake.killed is True


def test_kill_before_start_is_a_noop() -> None:
    RawPipeEncoder(["ffmpeg.exe"]).kill()  # no debe lanzar
