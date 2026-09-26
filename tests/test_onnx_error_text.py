from __future__ import annotations

from app.services.engines.onnx_common import readable_error_text, wrap_onnx_error
from app.services.engines.onnx_video_upscaler import is_device_removed_error, is_oom_error

# Texto real de P3-GPU-2 (Windows en español, ORT 1.24.4 DirectML): llega en cp1252 y
# pybind11 lo decodifica como UTF-8, así que Python ve un UnicodeDecodeError.
ANSI_DEVICE_REMOVED = (
    b"Non-zero status code returned while running MemcpyToHost node. Name:'Memcpy_token_51' "
    b"Exception(4) tid(1c26c) 887A0005 La instancia de dispositivo de GPU se ha suspendido. "
    b"Use GetDeviceRemovedReason para averiguar cu\xe1l es la acci\xf3n adecuada."
)


def ansi_decode_error() -> UnicodeDecodeError:
    try:
        ANSI_DEVICE_REMOVED.decode("utf-8")
    except UnicodeDecodeError as exc:
        return exc
    raise AssertionError("the sample must not be valid UTF-8")


def test_readable_error_text_recovers_the_ansi_message_behind_a_decode_error() -> None:
    text = readable_error_text(ansi_decode_error())

    assert "887A0005" in text
    assert "utf-8" not in text


def test_readable_error_text_keeps_ordinary_messages() -> None:
    assert readable_error_text(RuntimeError("boom")) == "boom"


def test_wrapped_decode_error_is_recognised_as_device_removal() -> None:
    wrapped = wrap_onnx_error("ONNX inference failed", ansi_decode_error())

    assert "887A0005" in str(wrapped)
    assert is_device_removed_error(wrapped)
    assert not is_oom_error(wrapped)


def test_raw_decode_error_is_recognised_as_device_removal() -> None:
    assert is_device_removed_error(ansi_decode_error())
