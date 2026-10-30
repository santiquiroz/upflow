from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import onnxruntime as ort
import pytest

from app.config import Settings
from app.services import ep_registry, generation_pipeline_modes
from app.services.engines import generation_onnx


TEXT_ENCODERS = ("text_encoder", "text_encoder_2", "text_encoder_3")
DEVICE_COMPONENTS = ("unet", "vae_encoder", "vae_decoder")


class RecordingSession:
    def __init__(self, path: str, *, sess_options: Any, providers: list[Any]) -> None:
        self.component = Path(path).parent.name
        self.options = sess_options
        self.providers = providers

    def get_providers(self) -> list[str]:
        return [provider[0] if isinstance(provider, tuple) else provider for provider in self.providers]


def write_pipeline_components(directory: Path, components: tuple[str, ...]) -> None:
    (directory / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
    )
    for component in components:
        component_dir = directory / component
        component_dir.mkdir()
        (component_dir / "model.onnx").write_bytes(b"fake")


@pytest.fixture
def fake_pipeline_loader(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    sessions: list[RecordingSession] = []
    calls: list[dict[str, Any]] = []
    tuned: list[tuple[Any, str]] = []
    tune = generation_onnx._tune_session_options_for_device

    def create_session(path: str, *, sess_options: Any = None, providers: Any = None, **kwargs: Any) -> RecordingSession:
        session = RecordingSession(path, sess_options=sess_options, providers=providers)
        sessions.append(session)
        return session

    def tune_options(options: Any, device: str) -> None:
        tuned.append((options, device))
        tune(options, device)
        if device != "cpu":
            options.add_session_config_entry("test.dml_only", "1")

    class FakePipeline:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            pipeline = SimpleNamespace()
            provider = kwargs["provider"]
            provider_options = kwargs.get("provider_options")
            providers = [(provider, provider_options)] if provider_options is not None else [provider]
            for component in (*TEXT_ENCODERS, *DEVICE_COMPONENTS):
                model_path = Path(path) / component / "model.onnx"
                if not model_path.is_file():
                    continue
                session = kwargs.get(f"{component}_session")
                if session is None:
                    session = create_session(
                        str(model_path),
                        sess_options=kwargs.get("session_options"),
                        providers=providers,
                    )
                setattr(pipeline, component, SimpleNamespace(session=session))
            return pipeline

    monkeypatch.setattr(ort, "InferenceSession", create_session)
    monkeypatch.setattr(ep_registry, "_initialize", lambda settings: None)
    monkeypatch.setattr(ep_registry, "_native_plugin_for", lambda device: None)
    monkeypatch.setattr(ep_registry, "record_session_providers", lambda *args, **kwargs: None)
    monkeypatch.setattr(generation_onnx, "_tune_session_options_for_device", tune_options)
    monkeypatch.setattr(generation_onnx, "_load_pipeline_class", lambda declared: FakePipeline)
    monkeypatch.setattr(generation_onnx, "load_img2img_class", lambda declared: FakePipeline)
    monkeypatch.setattr(generation_pipeline_modes, "load_inpaint_class", lambda declared: FakePipeline)
    return SimpleNamespace(sessions=sessions, calls=calls, tuned=tuned)


@pytest.mark.parametrize("mode", ["text2img", "img2img", "inpaint"])
@pytest.mark.parametrize("device", ["dml:0", "cpu"])
@pytest.mark.parametrize("encoders_on_cpu", [True, False])
@pytest.mark.parametrize("encoders", [("text_encoder",), TEXT_ENCODERS])
def test_pipeline_places_text_encoders_on_configured_device(
    tmp_path: Path,
    fake_pipeline_loader: SimpleNamespace,
    mode: str,
    device: str,
    encoders_on_cpu: bool,
    encoders: tuple[str, ...],
) -> None:
    write_pipeline_components(tmp_path, (*encoders, *DEVICE_COMPONENTS))
    settings = Settings(
        RUNTIME_DIR=str(tmp_path),
        GENERATION_TEXT_ENCODERS_ON_CPU=encoders_on_cpu,
        _env_file=None,
    )
    engine = generation_onnx.GenerationEngine(settings, SimpleNamespace())

    pipeline = engine._create_pipeline(tmp_path, device, mode)

    expected_provider = ep_registry.CPU_PROVIDER if device == "cpu" else ep_registry.DML_PROVIDER
    encoder_provider = ep_registry.CPU_PROVIDER if encoders_on_cpu else expected_provider
    kwargs = fake_pipeline_loader.calls[0]
    assert kwargs["provider"] == expected_provider
    assert kwargs["use_io_binding"] is False
    assert (kwargs["session_options"], device) in fake_pipeline_loader.tuned
    assert len(fake_pipeline_loader.sessions) == len(encoders) + len(DEVICE_COMPONENTS)
    for component in DEVICE_COMPONENTS:
        session = getattr(pipeline, component).session
        assert session.get_providers() == [expected_provider]
        assert session.options is kwargs["session_options"]
    for component in encoders:
        session = getattr(pipeline, component).session
        assert session.get_providers() == [encoder_provider]
        override = encoders_on_cpu and device != "cpu"
        assert (f"{component}_session" in kwargs) is override
        if override:
            assert kwargs[f"{component}_session"] is session
            assert session.options is not kwargs["session_options"]
            assert all(session.options is not options for options, tuned_device in fake_pipeline_loader.tuned if tuned_device != "cpu")
            if session.options is not None:
                with pytest.raises(RuntimeError):
                    session.options.get_session_config_entry("test.dml_only")
        else:
            assert session.options is kwargs["session_options"]
    for component in set(TEXT_ENCODERS) - set(encoders):
        assert f"{component}_session" not in kwargs


def test_text_encoders_cpu_setting_defaults_to_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GENERATION_TEXT_ENCODERS_ON_CPU", raising=False)

    settings = Settings(_env_file=None)

    assert settings.generation_text_encoders_on_cpu is True


def test_text_encoders_cpu_setting_accepts_environment_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATION_TEXT_ENCODERS_ON_CPU", "false")

    settings = Settings(_env_file=None)

    assert settings.generation_text_encoders_on_cpu is False
