from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.services import ep_registry

# ---------------------------------------------------------------------------
# Fase 1b (2026-07-31) - selector de EP por dispositivo. Todo mockeado al
# estilo test_gmfss_engine: se parchea onnxruntime.InferenceSession /
# SessionOptions y los seams internos del registry (_import_plugin_module,
# _adapter_vendor_ids). Nunca se registra un plugin real ni se toca la GPU.
# ---------------------------------------------------------------------------

NVIDIA = 0x10DE
AMD = 0x1002
INTEL = 0x8086


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class FakeEpDevice:
    def __init__(self, ep_name: str) -> None:
        self.ep_name = ep_name


class FakeSessionOptions:
    def __init__(self) -> None:
        self.provider_devices: list[Any] = []

    def add_provider_for_devices(self, devices: list[Any], options: dict[str, str]) -> None:
        self.provider_devices.extend(devices)


class FakeInferenceSession:
    calls: list[tuple[str, Any, Any]] = []
    fail_native = False
    fail_all = False

    def __init__(self, path: str, sess_options: Any = None, providers: Any = None) -> None:
        FakeInferenceSession.calls.append((path, sess_options, providers))
        if FakeInferenceSession.fail_all:
            raise RuntimeError("boom: session creation failed")
        is_native = providers is None and getattr(sess_options, "provider_devices", [])
        if FakeInferenceSession.fail_native and is_native:
            raise RuntimeError("boom: native EP rejected the model")


class FakePluginModule:
    def __init__(self, lib_path: str) -> None:
        self._lib_path = lib_path

    def get_library_path(self) -> str:
        return self._lib_path


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import onnxruntime as ort

    ep_registry.reset()
    FakeInferenceSession.calls = []
    FakeInferenceSession.fail_native = False
    FakeInferenceSession.fail_all = False

    registered: list[tuple[str, str]] = []
    monkeypatch.setattr(ort, "InferenceSession", FakeInferenceSession)
    monkeypatch.setattr(ort, "SessionOptions", FakeSessionOptions)
    monkeypatch.setattr(
        ort,
        "register_execution_provider_library",
        lambda name, path: registered.append((name, str(path))),
        raising=False,
    )
    monkeypatch.setattr(
        ort,
        "get_ep_devices",
        lambda: [FakeEpDevice(name) for name, _ in registered],
        raising=False,
    )
    monkeypatch.setattr(ep_registry, "_adapter_vendor_ids", lambda: [AMD])
    monkeypatch.setattr(
        ep_registry, "_import_plugin_module", lambda name: (_ for _ in ()).throw(ImportError(name))
    )
    monkeypatch.setattr(ep_registry, "_preload_plugin_deps", lambda lib_path: None)
    yield registered
    ep_registry.reset()


def install_nvidia_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    lib = tmp_path / "onnxruntime_providers_nv_tensorrt_rtx.dll"
    lib.write_bytes(b"fake")
    module = FakePluginModule(str(lib))
    monkeypatch.setattr(ep_registry, "_import_plugin_module", lambda name: module)
    monkeypatch.setattr(ep_registry, "_adapter_vendor_ids", lambda: [NVIDIA])
    return str(lib)


# --- baseline -----------------------------------------------------------


def test_cpu_device_uses_cpu_provider_only(tmp_path: Path) -> None:
    ep_registry.create_session("model.onnx", "cpu", make_settings(tmp_path))
    assert FakeInferenceSession.calls == [("model.onnx", None, ["CPUExecutionProvider"])]


def test_dml_device_without_native_plugin_uses_dml_baseline(tmp_path: Path) -> None:
    ep_registry.create_session("model.onnx", "dml:1", make_settings(tmp_path))
    (_, _, providers), = FakeInferenceSession.calls
    assert providers == [("DmlExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"]


def test_unsupported_device_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        ep_registry.create_session("model.onnx", "npu:0", make_settings(tmp_path))


def test_sess_options_factory_used_for_baseline(tmp_path: Path) -> None:
    options = FakeSessionOptions()
    ep_registry.create_session(
        "model.onnx", "cpu", make_settings(tmp_path), sess_options_factory=lambda: options
    )
    (_, sess_options, _), = FakeInferenceSession.calls
    assert sess_options is options


# --- lane nativo --------------------------------------------------------


def test_native_plugin_used_when_vendor_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    lib = install_nvidia_plugin(monkeypatch, tmp_path)

    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    assert clean_registry == [("NvTensorRTRTXExecutionProvider", lib)]
    (path, sess_options, providers), = FakeInferenceSession.calls
    assert path == "model.onnx"
    assert providers is None
    assert [d.ep_name for d in sess_options.provider_devices] == ["NvTensorRTRTXExecutionProvider"]


def test_vendor_mismatch_never_registers_plugin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    install_nvidia_plugin(monkeypatch, tmp_path)
    monkeypatch.setattr(ep_registry, "_adapter_vendor_ids", lambda: [AMD])

    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    assert clean_registry == []
    (_, _, providers), = FakeInferenceSession.calls
    assert providers[0][0] == "DmlExecutionProvider"


def test_native_disabled_by_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    install_nvidia_plugin(monkeypatch, tmp_path)

    ep_registry.create_session(
        "model.onnx", "dml:0", make_settings(tmp_path, NATIVE_EP_ENABLED=False)
    )

    assert clean_registry == []
    (_, _, providers), = FakeInferenceSession.calls
    assert providers[0][0] == "DmlExecutionProvider"


def test_registration_happens_once_across_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    install_nvidia_plugin(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)

    ep_registry.create_session("a.onnx", "dml:0", settings)
    ep_registry.create_session("b.onnx", "dml:0", settings)

    assert len(clean_registry) == 1


# --- fallback (el job jamás falla por el camino nuevo) -------------------


def test_native_session_failure_falls_back_to_dml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_nvidia_plugin(monkeypatch, tmp_path)
    FakeInferenceSession.fail_native = True

    session = ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    assert isinstance(session, FakeInferenceSession)
    assert len(FakeInferenceSession.calls) == 2
    _, _, fallback_providers = FakeInferenceSession.calls[1]
    assert fallback_providers == [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]


def test_fallback_uses_fresh_session_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Las options del intento nativo quedan contaminadas por
    # add_provider_for_devices: reutilizarlas en el fallback arrastraría el EP
    # roto. La factory debe llamarse de nuevo por intento.
    install_nvidia_plugin(monkeypatch, tmp_path)
    FakeInferenceSession.fail_native = True
    built: list[FakeSessionOptions] = []

    def factory() -> FakeSessionOptions:
        options = FakeSessionOptions()
        built.append(options)
        return options

    ep_registry.create_session(
        "model.onnx", "dml:0", make_settings(tmp_path), sess_options_factory=factory
    )

    assert len(built) == 2
    assert built[0].provider_devices and not built[1].provider_devices


def test_registration_failure_is_isolated_and_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import onnxruntime as ort

    install_nvidia_plugin(monkeypatch, tmp_path)
    attempts: list[str] = []

    def failing_register(name: str, path: str) -> None:
        attempts.append(name)
        raise RuntimeError("plugin DLL rejected")

    monkeypatch.setattr(ort, "register_execution_provider_library", failing_register, raising=False)
    settings = make_settings(tmp_path)

    ep_registry.create_session("a.onnx", "dml:0", settings)
    ep_registry.create_session("b.onnx", "dml:0", settings)

    assert attempts == ["NvTensorRTRTXExecutionProvider"]
    for _, _, providers in FakeInferenceSession.calls:
        assert providers[0][0] == "DmlExecutionProvider"


def test_native_failure_after_registration_reports_error_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_nvidia_plugin(monkeypatch, tmp_path)
    FakeInferenceSession.fail_native = True
    settings = make_settings(tmp_path)

    ep_registry.create_session("model.onnx", "dml:0", settings)

    status = ep_registry.active_ep_for_device("dml:0", settings)
    assert status.ep_name == "DmlExecutionProvider"
    assert status.state == ep_registry.EP_STATE_ERROR
    assert status.detail


# --- estado para Settings -------------------------------------------------


def test_active_ep_for_cpu_is_cpu_baseline(tmp_path: Path) -> None:
    status = ep_registry.active_ep_for_device("cpu", make_settings(tmp_path))
    assert status.ep_name == "CPUExecutionProvider"
    assert status.state == ep_registry.EP_STATE_BASELINE


def test_active_ep_baseline_without_plugins(tmp_path: Path) -> None:
    status = ep_registry.active_ep_for_device("dml:0", make_settings(tmp_path))
    assert status.ep_name == "DmlExecutionProvider"
    assert status.label == "DirectML"
    assert status.state == ep_registry.EP_STATE_BASELINE


def test_active_ep_native_when_plugin_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_nvidia_plugin(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)

    status = ep_registry.active_ep_for_device("dml:0", settings)

    assert status.ep_name == "NvTensorRTRTXExecutionProvider"
    assert status.label == "TensorRT-RTX"
    assert status.state == ep_registry.EP_STATE_NATIVE


def test_warmup_pending_during_native_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # "Preparando aceleración para tu GPU": la compilación del primer uso debe
    # ser observable mientras ocurre, nunca silencio.
    install_nvidia_plugin(monkeypatch, tmp_path)
    seen: list[bool] = []
    original_init = FakeInferenceSession.__init__

    def spying_init(self, path, sess_options=None, providers=None):  # type: ignore[no-untyped-def]
        if providers is None:
            seen.append(ep_registry.warmup_pending("dml:0"))
        original_init(self, path, sess_options=sess_options, providers=providers)

    monkeypatch.setattr(FakeInferenceSession, "__init__", spying_init)

    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    assert seen == [True]
    assert ep_registry.warmup_pending("dml:0") is False


# --- plugins por carpeta (OpenVINO desde el NuGet) ------------------------


def test_openvino_plugin_detected_from_plugins_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    plugin_dir = tmp_path / "ep-plugins" / "openvino"
    plugin_dir.mkdir(parents=True)
    dll = plugin_dir / "onnxruntime_providers_openvino_plugin.dll"
    dll.write_bytes(b"fake")
    monkeypatch.setattr(ep_registry, "_adapter_vendor_ids", lambda: [INTEL])
    settings = make_settings(tmp_path, EP_PLUGINS_DIR=str(tmp_path / "ep-plugins"))

    ep_registry.create_session("model.onnx", "dml:0", settings)

    assert clean_registry == [("OpenVINOExecutionProvider", str(dll))]


# --- Fase 2: catálogo Windows ML ------------------------------------------


class FakeReadyResult:
    def __init__(self, ok: bool = True) -> None:
        self.status = "SUCCESS" if ok else "FAILURE"


class FakeReadyOperation:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok

    def get(self) -> FakeReadyResult:
        if not self._ok:
            raise RuntimeError("download failed")
        return FakeReadyResult()


class FakeCatalogProvider:
    def __init__(self, name: str, library_path: str, ok: bool = True) -> None:
        self.name = name
        self.library_path = library_path
        self._ok = ok

    def ensure_ready_async(self) -> FakeReadyOperation:
        return FakeReadyOperation(self._ok)


def install_winml_catalog(
    monkeypatch: pytest.MonkeyPatch, providers: list[FakeCatalogProvider], build: int = 26100
) -> None:
    monkeypatch.setattr(ep_registry, "_windows_build", lambda: build)
    monkeypatch.setattr(ep_registry, "_winml_find_providers", lambda: providers)


def test_winml_catalog_registers_ready_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    dll = tmp_path / "onnxruntime_providers_migraphx.dll"
    dll.write_bytes(b"fake")
    install_winml_catalog(
        monkeypatch, [FakeCatalogProvider("MIGraphXExecutionProvider", str(dll))]
    )
    settings = make_settings(tmp_path)

    ep_registry.create_session("model.onnx", "dml:0", settings)

    assert clean_registry == [("MIGraphXExecutionProvider", str(dll))]
    status = ep_registry.active_ep_for_device("dml:0", settings)
    assert status.state == ep_registry.EP_STATE_NATIVE
    assert "Windows ML" in status.label


def test_winml_catalog_skipped_below_min_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    dll = tmp_path / "ep.dll"
    dll.write_bytes(b"fake")
    install_winml_catalog(
        monkeypatch, [FakeCatalogProvider("MIGraphXExecutionProvider", str(dll))], build=22631
    )

    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    assert clean_registry == []


def test_winml_catalog_absent_projection_is_clean_skip(tmp_path: Path, clean_registry: list) -> None:
    # Sin proyección Python instalada (el default del fixture lanza ImportError
    # vía _winml_find_providers real): baseline intacto, cero ruido.
    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))
    assert clean_registry == []


def test_winml_catalog_never_registers_embedded_eps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    # Regla issue microsoft/onnxruntime#29372: registrar como plugin un EP ya
    # embebido en el build (DML/CPU) causa double-free.
    dll = tmp_path / "dml.dll"
    dll.write_bytes(b"fake")
    install_winml_catalog(monkeypatch, [FakeCatalogProvider("DmlExecutionProvider", str(dll))])

    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    assert clean_registry == []


def test_winml_catalog_provider_failure_is_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    bad = FakeCatalogProvider("QNNExecutionProvider", str(tmp_path / "missing.dll"), ok=False)
    good_dll = tmp_path / "migraphx.dll"
    good_dll.write_bytes(b"fake")
    good = FakeCatalogProvider("MIGraphXExecutionProvider", str(good_dll))
    install_winml_catalog(monkeypatch, [bad, good])

    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    assert clean_registry == [("MIGraphXExecutionProvider", str(good_dll))]


def test_winml_catalog_disabled_by_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    dll = tmp_path / "ep.dll"
    dll.write_bytes(b"fake")
    install_winml_catalog(monkeypatch, [FakeCatalogProvider("MIGraphXExecutionProvider", str(dll))])

    ep_registry.create_session(
        "model.onnx", "dml:0", make_settings(tmp_path, WINML_CATALOG_ENABLED=False)
    )

    assert clean_registry == []
