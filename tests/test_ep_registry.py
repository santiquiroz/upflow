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


def test_active_ep_identifies_the_plugin_once_registered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Este test afirmaba EP_STATE_NATIVE apenas registrado, congelando el bug:
    la pantalla podia anunciar el acelerador en una maquina donde despues todos
    los trabajos caen a DirectML. El plugin se identifica igual, pero "native"
    ahora exige que una sesion de verdad haya corrido ahi."""
    install_nvidia_plugin(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)

    status = ep_registry.active_ep_for_device("dml:0", settings)

    assert status.ep_name == "NvTensorRTRTXExecutionProvider"
    assert status.label == "TensorRT-RTX"
    assert status.state == ep_registry.EP_STATE_READY


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

# --- ruteo por placa en maquinas con mas de una GPU ----------------------


def install_plugin_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spec_index: int, vendors: list[int]
) -> str:
    """Instala UN plugin (por indice en NATIVE_EP_SPECS) y declara los vendors
    de los adaptadores en el orden en que los expone dml:N."""
    spec = ep_registry.NATIVE_EP_SPECS[spec_index]
    lib = tmp_path / f"{spec.ep_name}.dll"
    lib.write_bytes(b"fake")
    module = FakePluginModule(str(lib))
    monkeypatch.setattr(ep_registry, "_import_plugin_module", lambda name: module)
    if spec.plugin_subdir and spec.dll_filename:
        plugin_dir = tmp_path / "ep-plugins" / spec.plugin_subdir
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / spec.dll_filename).write_bytes(b"fake")
    monkeypatch.setattr(ep_registry, "_adapter_vendor_ids", lambda: vendors)
    return str(lib)


def used_ep_names() -> list[str]:
    (_, sess_options, providers), = FakeInferenceSession.calls
    if providers is not None:
        return [providers[0][0]]
    return [d.ep_name for d in sess_options.provider_devices]


def test_plugin_is_not_used_for_a_card_of_another_vendor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    """dml:0 es la iGPU Intel y dml:1 la NVIDIA. Un trabajo fijado a dml:0 NO
    puede terminar corriendo en la NVIDIA: seria correr en otra placa en
    silencio, ignorando el dispositivo que pidio el usuario."""
    install_plugin_for(monkeypatch, tmp_path, 0, [INTEL, NVIDIA])  # plugin NVIDIA

    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    assert used_ep_names() == ["DmlExecutionProvider"]


def test_plugin_is_used_for_the_card_that_actually_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    install_plugin_for(monkeypatch, tmp_path, 0, [INTEL, NVIDIA])  # plugin NVIDIA

    ep_registry.create_session("model.onnx", "dml:1", make_settings(tmp_path))

    assert used_ep_names() == ["NvTensorRTRTXExecutionProvider"]


def test_a_device_index_beyond_the_adapter_list_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    """Si no se puede saber de que placa hablamos, no se adivina."""
    install_plugin_for(monkeypatch, tmp_path, 0, [NVIDIA])

    ep_registry.create_session("model.onnx", "dml:7", make_settings(tmp_path))

    assert used_ep_names() == ["DmlExecutionProvider"]


def test_single_gpu_machines_keep_working(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    install_plugin_for(monkeypatch, tmp_path, 0, [NVIDIA])

    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    assert used_ep_names() == ["NvTensorRTRTXExecutionProvider"]


def test_catalog_ep_is_not_attributed_to_a_card_on_a_multi_gpu_box(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    """Un EP del catalogo no dice de que placa es. Con una sola GPU no hay
    ambiguedad; con dos, usarlo seria adivinar."""
    dll = tmp_path / "onnxruntime_providers_migraphx.dll"
    dll.write_bytes(b"fake")
    install_winml_catalog(
        monkeypatch, [FakeCatalogProvider("MIGraphXExecutionProvider", str(dll))]
    )
    monkeypatch.setattr(ep_registry, "_adapter_vendor_ids", lambda: [INTEL, AMD])

    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    assert used_ep_names() == ["DmlExecutionProvider"]


# --- superficie para loaders que construyen la sesion ellos mismos --------
# optimum y transformers no aceptan una InferenceSession ya hecha: piden
# provider/session_options y la construyen adentro. Un EP de plugin NO se puede
# pedir por nombre (validate_provider_availability solo mira
# get_available_providers, que no ve plugins), asi que se pide con providers=[]
# y el EP puesto en el SessionOptions. Verificado contra onnxruntime real.


def test_loader_kwargs_cpu_asks_for_the_cpu_provider(tmp_path: Path) -> None:
    kwargs = ep_registry.loader_kwargs("cpu", make_settings(tmp_path))
    assert kwargs == {"provider": "CPUExecutionProvider"}


def test_loader_kwargs_falls_back_to_directml_with_the_right_device_id(tmp_path: Path) -> None:
    kwargs = ep_registry.loader_kwargs("dml:1", make_settings(tmp_path))
    assert kwargs["provider"] == "DmlExecutionProvider"
    assert kwargs["provider_options"] == {"device_id": 1}
    assert "session_options" not in kwargs


def test_loader_kwargs_uses_the_native_ep_through_session_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    install_nvidia_plugin(monkeypatch, tmp_path)

    kwargs = ep_registry.loader_kwargs("dml:0", make_settings(tmp_path))

    # Pedirlo por nombre no sirve: optimum validaria el provider contra
    # get_available_providers y un plugin no aparece ahi.
    assert kwargs["providers"] == []
    assert "provider" not in kwargs
    names = [d.ep_name for d in kwargs["session_options"].provider_devices]
    assert names == ["NvTensorRTRTXExecutionProvider"]


def test_loader_kwargs_respects_the_card_the_job_was_pinned_to(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    install_plugin_for(monkeypatch, tmp_path, 0, [INTEL, NVIDIA])  # plugin NVIDIA

    de_la_intel = ep_registry.loader_kwargs("dml:0", make_settings(tmp_path))
    de_la_nvidia = ep_registry.loader_kwargs("dml:1", make_settings(tmp_path))

    assert de_la_intel["provider"] == "DmlExecutionProvider"
    assert de_la_nvidia["providers"] == []


def test_loader_kwargs_hands_a_fresh_session_options_each_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    """add_provider_for_devices dos veces sobre el MISMO objeto falla con
    'Provider has already been registered' (comprobado contra onnxruntime real),
    asi que dos llamadas no pueden compartir el objeto."""
    install_nvidia_plugin(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)

    uno = ep_registry.loader_kwargs("dml:0", settings)["session_options"]
    otro = ep_registry.loader_kwargs("dml:0", settings)["session_options"]

    assert uno is not otro
    assert len(otro.provider_devices) == 1


def test_loader_kwargs_falls_back_when_native_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    install_nvidia_plugin(monkeypatch, tmp_path)

    kwargs = ep_registry.loader_kwargs("dml:0", make_settings(tmp_path, NATIVE_EP_ENABLED=False))

    assert kwargs["provider"] == "DmlExecutionProvider"


def test_the_shipped_plugin_folder_is_found_without_configuring_anything(tmp_path: Path) -> None:
    """El script de descarga deja los DLLs en vendor/ep-plugins/. Si el default
    de EP_PLUGINS_DIR es vacio, la app no mira ahi y el acelerador de Intel
    queda instalado pero muerto."""
    # Se mira el default DECLARADO y no una instancia: conftest pisa la variable
    # de entorno justamente para que los tests no dependan de si esta maquina
    # tiene el acelerador bajado.
    declarado = Settings.model_fields["ep_plugins_dir"].default
    assert declarado, "con el default vacio, los DLLs se instalan donde nadie los mira"
    assert Path(declarado).name == "ep-plugins"


def test_intel_plugin_is_picked_up_from_the_shipped_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    spec = next(s for s in ep_registry.NATIVE_EP_SPECS if s.vendor_id == INTEL)
    settings = make_settings(tmp_path, EP_PLUGINS_DIR=str(tmp_path / "ep-plugins"))
    plugin_dir = tmp_path / "ep-plugins" / spec.plugin_subdir
    plugin_dir.mkdir(parents=True)
    dll = plugin_dir / spec.dll_filename
    dll.write_bytes(b"fake")
    monkeypatch.setattr(ep_registry, "_adapter_vendor_ids", lambda: [INTEL])

    ep_registry.create_session("model.onnx", "dml:0", settings)

    assert clean_registry == [(spec.ep_name, str(dll))]


# --- el estado que se muestra tiene que ser verdad ----------------------


def test_a_plugin_that_failed_to_register_is_not_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    """El caso real: se instalan 93 MB del acelerador de NVIDIA, falta el driver,
    el registro falla. Antes se veia EXACTAMENTE igual que no tenerlo instalado."""
    import onnxruntime as ort

    install_nvidia_plugin(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ort,
        "register_execution_provider_library",
        lambda name, path: (_ for _ in ()).throw(RuntimeError("falta nvml.dll (Error 126)")),
        raising=False,
    )
    settings = make_settings(tmp_path)

    status = ep_registry.active_ep_for_device("dml:0", settings)

    assert status.state == ep_registry.EP_STATE_ERROR
    assert "nvml" in status.detail
    assert "TensorRT-RTX" in status.detail


def test_a_registered_plugin_is_not_called_native_until_it_actually_ran(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    """Decia 'native' apenas se registraba: la pantalla podia anunciar
    TensorRT-RTX en una maquina donde todos los trabajos caen a DirectML."""
    install_nvidia_plugin(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)

    status = ep_registry.active_ep_for_device("dml:0", settings)

    assert status.state == ep_registry.EP_STATE_READY
    assert status.label == "TensorRT-RTX"


def test_it_becomes_native_once_a_session_really_used_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    install_nvidia_plugin(monkeypatch, tmp_path)
    settings = make_settings(tmp_path)

    ep_registry.create_session("model.onnx", "dml:0", settings)

    assert ep_registry.active_ep_for_device("dml:0", settings).state == ep_registry.EP_STATE_NATIVE


def test_a_session_that_fell_back_is_reported_as_fallback_not_native(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    install_nvidia_plugin(monkeypatch, tmp_path)
    FakeInferenceSession.fail_native = True
    settings = make_settings(tmp_path)

    ep_registry.create_session("model.onnx", "dml:0", settings)

    status = ep_registry.active_ep_for_device("dml:0", settings)
    assert status.state == ep_registry.EP_STATE_ERROR


def test_a_failed_native_session_is_not_retried_forever(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    """Con TensorRT, reintentar el camino nativo en cada trabajo paga una
    compilacion cara para volver a caer a DirectML."""
    install_nvidia_plugin(monkeypatch, tmp_path)
    FakeInferenceSession.fail_native = True
    settings = make_settings(tmp_path)

    ep_registry.create_session("model.onnx", "dml:0", settings)
    intentos_primero = len(FakeInferenceSession.calls)
    FakeInferenceSession.calls = []
    ep_registry.create_session("model.onnx", "dml:0", settings)

    assert intentos_primero == 2  # nativo (falla) + fallback DirectML
    assert len(FakeInferenceSession.calls) == 1  # el segundo va directo a DirectML


def test_reset_unregisters_from_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_registry: list
) -> None:
    """Sin desregistrar, un segundo _initialize re-registra el mismo nombre en
    ORT real. Invisible hoy porque ORT esta fakeado en todos los tests."""
    import onnxruntime as ort

    desregistrados: list[str] = []
    monkeypatch.setattr(
        ort, "unregister_execution_provider_library", desregistrados.append, raising=False
    )
    install_nvidia_plugin(monkeypatch, tmp_path)
    ep_registry.create_session("model.onnx", "dml:0", make_settings(tmp_path))

    ep_registry.reset()

    assert desregistrados == ["NvTensorRTRTXExecutionProvider"]


# --- precarga de DLLs hermanas -------------------------------------------
# La UNICA funcion del registry que toca el sistema operativo, y la unica que
# estaba mockeada en todos los tests. Existe para evitar el "Error 126 engañoso"
# que da ORT cuando el plugin carga sin sus dependencias al lado.
#
# El fixture autouse la reemplaza por un no-op en todos los demas tests, asi que
# se captura la de verdad al importar el modulo, antes de cualquier parche.
_PRELOAD_REAL = ep_registry._preload_plugin_deps


def test_preload_adds_the_plugin_folder_to_the_dll_search_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    lib = plugin_dir / "main_plugin.dll"
    lib.write_bytes(b"fake")

    agregados: list[str] = []
    monkeypatch.setattr(ep_registry.os, "add_dll_directory", agregados.append, raising=False)
    monkeypatch.setattr(ep_registry.ctypes, "WinDLL", lambda p: None, raising=False)

    _PRELOAD_REAL(str(lib))

    assert agregados == [str(plugin_dir)]


def test_preload_loads_the_siblings_but_never_the_plugin_itself(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cargar el plugin aca lo dejaria mapeado dos veces: lo carga ORT despues."""
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    lib = plugin_dir / "main_plugin.dll"
    lib.write_bytes(b"fake")
    (plugin_dir / "dep_uno.dll").write_bytes(b"fake")
    (plugin_dir / "dep_dos.dll").write_bytes(b"fake")
    (plugin_dir / "notas.txt").write_bytes(b"no es una dll")

    cargados: list[str] = []
    monkeypatch.setattr(ep_registry.os, "add_dll_directory", lambda d: None, raising=False)
    monkeypatch.setattr(
        ep_registry.ctypes, "WinDLL", lambda p: cargados.append(Path(p).name), raising=False
    )

    _PRELOAD_REAL(str(lib))

    assert sorted(cargados) == ["dep_dos.dll", "dep_uno.dll"]


def test_preload_survives_a_sibling_that_cannot_be_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Una dep opcional que no carga no puede tumbar el registro: el fallo real
    lo tiene que reportar ORT, no esta precarga."""
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    lib = plugin_dir / "main_plugin.dll"
    lib.write_bytes(b"fake")
    (plugin_dir / "aaa_rota.dll").write_bytes(b"fake")
    (plugin_dir / "zzz_buena.dll").write_bytes(b"fake")

    cargados: list[str] = []

    def fake_windll(path: str) -> None:
        if "rota" in path:
            raise OSError("no se pudo cargar")
        cargados.append(Path(path).name)

    monkeypatch.setattr(ep_registry.os, "add_dll_directory", lambda d: None, raising=False)
    monkeypatch.setattr(ep_registry.ctypes, "WinDLL", fake_windll, raising=False)

    _PRELOAD_REAL(str(lib))

    # La rota va primero por orden alfabetico: si abortara, la buena no se carga.
    assert cargados == ["zzz_buena.dll"]


# --- fallback silencioso a CPU ------------------------------------------


def _fake_session_providers(monkeypatch: pytest.MonkeyPatch, providers: list[str]) -> None:
    monkeypatch.setattr(
        FakeInferenceSession, "get_providers", lambda self: list(providers), raising=False
    )


def test_dml_session_running_on_cpu_reports_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_session_providers(monkeypatch, ["CPUExecutionProvider"])
    settings = make_settings(tmp_path)
    ep_registry.create_session("model.onnx", "dml:0", settings)
    status = ep_registry.active_ep_for_device("dml:0", settings)
    assert status.state == ep_registry.EP_STATE_CPU_FALLBACK
    assert status.ep_name == "CPUExecutionProvider"
    assert status.detail


def test_dml_session_on_dml_keeps_baseline_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_session_providers(monkeypatch, ["DmlExecutionProvider", "CPUExecutionProvider"])
    settings = make_settings(tmp_path)
    ep_registry.create_session("model.onnx", "dml:0", settings)
    status = ep_registry.active_ep_for_device("dml:0", settings)
    assert status.state == ep_registry.EP_STATE_BASELINE


def test_session_without_get_providers_is_harmless(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    ep_registry.create_session("model.onnx", "dml:0", settings)
    status = ep_registry.active_ep_for_device("dml:0", settings)
    assert status.state == ep_registry.EP_STATE_BASELINE


def test_reset_clears_effective_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_session_providers(monkeypatch, ["CPUExecutionProvider"])
    settings = make_settings(tmp_path)
    ep_registry.create_session("model.onnx", "dml:0", settings)
    assert ep_registry.active_ep_for_device("dml:0", settings).state == ep_registry.EP_STATE_CPU_FALLBACK
    ep_registry.reset()
    assert ep_registry.active_ep_for_device("dml:0", settings).state == ep_registry.EP_STATE_BASELINE


def test_record_session_providers_warns_on_cpu_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    class _CpuSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    with caplog.at_level(logging.WARNING, logger="app.services.ep_registry"):
        ep_registry.record_session_providers("dml:0", _CpuSession(), context="test")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("dml:0" in r.getMessage() for r in warnings)
