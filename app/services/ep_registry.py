from __future__ import annotations

import ctypes
import importlib
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import Settings

# ---------------------------------------------------------------------------
# Fase 1b (2026-07-31) - selector de execution provider por dispositivo.
#
# DirectML es SIEMPRE el baseline universal (decisión explícita, no
# negociable); el EP nativo del vendor (TensorRT-RTX en NVIDIA, OpenVINO en
# Intel) se SUMA cuando su plugin está instalado y el hardware está presente.
# Todo el conocimiento empírico viene del spike gate 1a
# (docs/superpowers/research/2026-07-31-spike-ep-plugin-convivencia.md):
#
#   - El wheel onnxruntime-directml 1.24.4 trae la API plugin ABI completa
#     (register_execution_provider_library / get_ep_devices /
#     SessionOptions.add_provider_for_devices) y un plugin compilado contra
#     un ORT más nuevo carga igual (ABI append-only, verificado con webgpu).
#   - ORT no resuelve las dependencias del plugin desde el directorio del
#     plugin: hay que hacer add_dll_directory + preload con ctypes.
#   - get_available_providers() NO lista plugins registrados (by design,
#     issue #27832): la fuente de verdad es get_ep_devices().
#   - El registro se gatea por vendor ANTES de intentarlo (TensorRT-RTX ni
#     carga sin driver NVIDIA) y cualquier fallo del camino nativo cae a
#     DirectML→CPU sin fallar el job (patrón stream-pipeline).
#
# El EP elegido es estático por proceso (v1: selector read-only en Settings),
# así que los caches de sesiones de los engines nunca sirven un EP viejo.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"
DML_PROVIDER = "DmlExecutionProvider"

EP_STATE_NATIVE = "native"
EP_STATE_BASELINE = "baseline"
EP_STATE_PREPARING = "preparing"
EP_STATE_ERROR = "error"
# Registrado y elegible, pero todavia no lo uso ningun trabajo. Antes se decia
# "native" apenas se registraba: la pantalla podia anunciar un acelerador en una
# maquina donde despues todos los trabajos caian a DirectML.
EP_STATE_READY = "ready"
# Una sesion REAL en un device dml:N termino corriendo en CPU: ORT cae a CPU en
# silencio cuando DirectML no inicializa, y sin este estado el unico sintoma era
# un trabajo eterno con la GPU fria.
EP_STATE_CPU_FALLBACK = "cpu_fallback"

VENDOR_NVIDIA = 0x10DE
VENDOR_INTEL = 0x8086


@dataclass(frozen=True, slots=True)
class NativeEpSpec:
    ep_name: str
    label: str
    vendor_id: int
    # Paquete pip que expone get_library_path() (NVIDIA), o None si el plugin
    # solo se distribuye como DLLs sueltos en ep_plugins_dir (Intel/NuGet).
    import_module: str | None
    plugin_subdir: str | None = None
    dll_filename: str | None = None


NATIVE_EP_SPECS: tuple[NativeEpSpec, ...] = (
    NativeEpSpec(
        ep_name="NvTensorRTRTXExecutionProvider",
        label="TensorRT-RTX",
        vendor_id=VENDOR_NVIDIA,
        import_module="onnxruntime_ep_nv_tensorrt_rtx",
    ),
    NativeEpSpec(
        ep_name="OpenVINOExecutionProvider",
        label="OpenVINO",
        vendor_id=VENDOR_INTEL,
        import_module=None,
        plugin_subdir="openvino",
        dll_filename="onnxruntime_providers_openvino_plugin.dll",
    ),
)


# Fase 2: catálogo Windows ML (Win11 24H2+, build 26100). Los EPs del catálogo
# se registran sobre el MISMO onnxruntime de pip vía register_execution_provider
# _library — Microsoft advierte explícitamente que EnsureAndRegisterCertifiedAsync
# no sirve para el env de Python. EPs embebidos en nuestro build jamás se
# re-registran como plugin (issue #29372: double-free).
WINML_MIN_BUILD = 26100
EMBEDDED_EP_NAMES = frozenset({CPU_PROVIDER, DML_PROVIDER})


@dataclass(frozen=True, slots=True)
class EpStatus:
    ep_name: str
    label: str
    state: str
    detail: str = ""


@dataclass(slots=True)
class _PluginState:
    spec: NativeEpSpec
    devices: list[Any]
    error: str = ""


_lock = threading.Lock()
_initialized = False
_plugins: dict[str, _PluginState] = {}
_session_errors: dict[str, str] = {}
_warmup: set[str] = set()
# Dispositivos donde una sesion nativa REALMENTE se creo, y donde ya fallo. Sin
# lo segundo, cada trabajo posterior repetia el intento nativo y pagaba el
# fallback: con TensorRT eso es una compilacion cara para nada.
_native_ok: set[str] = set()
_native_failed: set[str] = set()
# Providers que las sesiones REALES reportaron por device (get_providers()).
# Fuente de verdad del fallback silencioso DML→CPU: la lista pedida no dice
# nada de la que ORT terminó usando.
_effective_providers: dict[str, list[str]] = {}


def reset() -> None:
    global _initialized
    with _lock:
        # Desregistrar en ORT: sin esto un segundo _initialize re-registra el
        # mismo nombre. Nunca se vio porque ORT esta fakeado en los tests.
        for ep_name, state in _plugins.items():
            if state.devices:
                _unregister_plugin(ep_name)
        _initialized = False
        _plugins.clear()
        _session_errors.clear()
        _warmup.clear()
        _native_ok.clear()
        _native_failed.clear()
        _effective_providers.clear()


def _unregister_plugin(ep_name: str) -> None:
    import onnxruntime as ort

    try:
        ort.unregister_execution_provider_library(ep_name)
    except Exception as exc:  # noqa: BLE001 -- limpiar jamas puede tumbar nada
        logger.debug("ep_registry: no se pudo desregistrar %s: %s", ep_name, exc)


def _import_plugin_module(name: str) -> Any:
    return importlib.import_module(name)


def _adapter_vendor_ids() -> list[int]:
    from app.services.devices_service import enumerate_adapter_vendor_ids

    return enumerate_adapter_vendor_ids()


def _preload_plugin_deps(lib_path: str) -> None:
    # ORT carga el plugin sin mirar su directorio: las DLLs hermanas
    # (cudart, tensorrt_rtx, openvino...) deben estar ya cargadas en el
    # proceso o el registro falla con un Error 126 engañoso.
    lib_dir = os.path.dirname(lib_path)
    os.add_dll_directory(lib_dir)
    lib_name = os.path.basename(lib_path).lower()
    for name in sorted(os.listdir(lib_dir)):
        if not name.lower().endswith(".dll") or name.lower() == lib_name:
            continue
        try:
            ctypes.WinDLL(os.path.join(lib_dir, name))
        except OSError:
            # Deps opcionales pueden faltar; el registro reporta el fallo real.
            continue


def _resolve_plugin_library(spec: NativeEpSpec, settings: Settings) -> str | None:
    if spec.import_module is not None:
        try:
            module = _import_plugin_module(spec.import_module)
        except ImportError:
            return None
        return str(module.get_library_path())
    plugins_dir = settings.ep_plugins_dir_path
    if plugins_dir is None or spec.plugin_subdir is None or spec.dll_filename is None:
        return None
    candidate = plugins_dir / spec.plugin_subdir / spec.dll_filename
    return str(candidate) if candidate.is_file() else None


def _windows_build() -> int:
    if sys.platform != "win32":
        return 0
    return sys.getwindowsversion().build


def _winml_find_providers() -> list[Any]:
    import winui3.microsoft.windows.ai.machinelearning as winml  # type: ignore[import-not-found]

    catalog = winml.ExecutionProviderCatalog.get_default()
    return list(catalog.find_all_providers())


def _register_winml_provider(provider: Any) -> _PluginState | None:
    import onnxruntime as ort

    spec = NativeEpSpec(
        ep_name=provider.name,
        label=f"{provider.name} (Windows ML)",
        vendor_id=0,
        import_module=None,
    )
    try:
        provider.ensure_ready_async().get()
        ort.register_execution_provider_library(provider.name, provider.library_path)
        devices = [d for d in ort.get_ep_devices() if d.ep_name == provider.name]
    except Exception as exc:  # noqa: BLE001 -- un EP del catálogo jamás tumba el baseline
        logger.warning("ep_registry: EP de Windows ML %s falló: %s", provider.name, exc)
        return None
    if not devices:
        return None
    logger.info("ep_registry: %s (Windows ML) registrado con %d device(s)", provider.name, len(devices))
    return _PluginState(spec=spec, devices=devices, error="")


def _register_winml_catalog(settings: Settings) -> None:
    if not settings.winml_catalog_enabled or _windows_build() < WINML_MIN_BUILD:
        return
    try:
        providers = _winml_find_providers()
    except Exception:  # noqa: BLE001 -- proyección no instalada / catálogo ausente: skip limpio
        return
    for provider in providers:
        name = getattr(provider, "name", "")
        if not name or name in EMBEDDED_EP_NAMES or name in _plugins:
            continue
        state = _register_winml_provider(provider)
        if state is not None:
            _plugins[name] = state


def _register_plugin(spec: NativeEpSpec, lib_path: str) -> _PluginState:
    import onnxruntime as ort

    try:
        _preload_plugin_deps(lib_path)
        ort.register_execution_provider_library(spec.ep_name, lib_path)
        devices = [d for d in ort.get_ep_devices() if d.ep_name == spec.ep_name]
    except Exception as exc:  # noqa: BLE001 -- el camino nativo jamás tumba el baseline
        logger.warning("ep_registry: registro de %s falló: %s", spec.label, exc)
        return _PluginState(spec=spec, devices=[], error=str(exc))
    if not devices:
        return _PluginState(spec=spec, devices=[], error="registrado pero sin devices")
    logger.info("ep_registry: %s registrado con %d device(s)", spec.label, len(devices))
    return _PluginState(spec=spec, devices=devices, error="")


def _initialize(settings: Settings) -> None:
    global _initialized
    with _lock:
        if _initialized:
            return
        _initialized = True
        if not settings.native_ep_enabled:
            return
        vendors = set(_adapter_vendor_ids())
        for spec in NATIVE_EP_SPECS:
            if spec.vendor_id not in vendors:
                continue
            lib_path = _resolve_plugin_library(spec, settings)
            if lib_path is None:
                continue
            _plugins[spec.ep_name] = _register_plugin(spec, lib_path)
        # El catálogo va después de los plugins explícitos: pip/EP_PLUGINS_DIR
        # son elección del usuario y ganan sobre lo que resuelva el sistema.
        _register_winml_catalog(settings)


def _parse_dml_device_id(device: str) -> int:
    _, _, suffix = device.partition(":")
    try:
        return int(suffix)
    except ValueError as exc:
        raise RuntimeError(f"Unsupported device for ONNX inference: {device!r}") from exc


def _dml_providers(device: str) -> list[Any]:
    return [(DML_PROVIDER, {"device_id": _parse_dml_device_id(device)}), CPU_PROVIDER]


# Los EPs del catálogo de Windows ML se registran con vendor_id 0: el catálogo
# no dice a qué adaptador pertenecen.
VENDOR_UNKNOWN = 0


def _vendor_of_adapter(device: str) -> int | None:
    """Vendor de la placa detrás de `dml:N`, o None si no se puede saber.

    `enumerate_adapter_vendor_ids` devuelve los vendors en el MISMO orden que
    numera dml:N, así que el índice del device es el índice del adaptador.
    """
    try:
        index = _parse_dml_device_id(device)
    except RuntimeError:
        return None
    vendors = _adapter_vendor_ids()
    return vendors[index] if 0 <= index < len(vendors) else None


def _failed_plugin_for(device: str) -> _PluginState | None:
    """El plugin de ESTA placa que se intento registrar y no pudo."""
    vendor = _vendor_of_adapter(device)
    if vendor is None:
        return None
    for state in _plugins.values():
        if not state.devices and state.error and state.spec.vendor_id == vendor:
            return state
    return None


def _native_plugin_for(device: str) -> _PluginState | None:
    """El plugin de la placa pedida, o None para caer a DirectML.

    Antes devolvía el primer plugin con devices para CUALQUIER dml:N. En una
    máquina con iGPU Intel y placa NVIDIA eso mandaba un trabajo fijado a la
    iGPU a correr en la NVIDIA, en silencio y contra lo que pidió el usuario.
    """
    if not device.startswith("dml:"):
        return None
    vendor = _vendor_of_adapter(device)
    if vendor is None:
        return None
    ready = [state for state in _plugins.values() if state.devices]
    for state in ready:
        if state.spec.vendor_id == vendor:
            return state
    # Un EP del catálogo no declara placa. Con una sola GPU no hay ambigüedad;
    # con varias, atribuirlo sería el mismo error que este método evita.
    if len(_adapter_vendor_ids()) == 1:
        return next((s for s in ready if s.spec.vendor_id == VENDOR_UNKNOWN), None)
    return None


def _build_options(factory: Callable[[], Any] | None) -> Any:
    return factory() if factory is not None else None


def _create_native_session(
    model_path: str, device: str, plugin: _PluginState, sess_options_factory: Callable[[], Any] | None
) -> Any:
    import onnxruntime as ort

    sess_options = _build_options(sess_options_factory)
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.add_provider_for_devices(plugin.devices[:1], {})
    _warmup.add(device)
    try:
        session = ort.InferenceSession(model_path, sess_options=sess_options)
    finally:
        _warmup.discard(device)
    _native_ok.add(device)
    return session


def warmup_pending(device: str) -> bool:
    return device in _warmup


def _is_cpu_fallback(device: str, providers: list[str]) -> bool:
    return device.startswith("dml:") and providers[:1] == [CPU_PROVIDER]


def record_session_providers(device: str, session: Any, context: str = "") -> list[str]:
    """Lee y registra los providers que la sesión REALMENTE usa.

    ORT cae de DirectML a CPU sin error: la única forma de saber dónde corre
    una sesión es preguntarle después de crearla.
    """
    get_providers = getattr(session, "get_providers", None)
    if get_providers is None:
        return []
    try:
        providers = list(get_providers())
    except Exception as exc:  # noqa: BLE001 -- observabilidad jamás tumba un job
        logger.debug("ep_registry: no se pudieron leer providers de %s: %s", device, exc)
        return []
    _effective_providers[device] = providers
    label = context or "sesión"
    if _is_cpu_fallback(device, providers):
        logger.warning(
            "ep_registry: %s pidió %s pero ORT la corre en CPU (providers efectivos: %s)",
            label,
            device,
            providers,
        )
    else:
        logger.info(
            "ep_registry: %s en %s, providers efectivos: %s", label, device, providers
        )
    return providers


def create_session(
    model_path: str,
    device: str,
    settings: Settings,
    *,
    sess_options_factory: Callable[[], Any] | None = None,
) -> Any:
    import onnxruntime as ort

    if device == "cpu":
        return ort.InferenceSession(
            model_path, sess_options=_build_options(sess_options_factory), providers=[CPU_PROVIDER]
        )
    if not device.startswith("dml:"):
        raise RuntimeError(f"Unsupported device for ONNX inference: {device!r}")

    _initialize(settings)
    plugin = _native_plugin_for(device)
    if plugin is not None and device not in _native_failed:
        try:
            session = _create_native_session(model_path, device, plugin, sess_options_factory)
            record_session_providers(device, session)
            return session
        except Exception as exc:  # noqa: BLE001 -- fallback garantizado: el job nunca falla por el EP nativo
            # Una sola vez: reintentarlo en cada trabajo paga el intento y el
            # fallback, y con TensorRT ese intento es una compilacion cara.
            _native_failed.add(device)
            _session_errors[device] = f"{plugin.spec.label}: {exc}"
            logger.warning(
                "ep_registry: sesión %s falló en %s, cayendo a DirectML: %s",
                plugin.spec.label,
                device,
                exc,
            )
    session = ort.InferenceSession(
        model_path,
        sess_options=_build_options(sess_options_factory),
        providers=_dml_providers(device),
    )
    record_session_providers(device, session)
    return session


def _with_session_options(
    kwargs: dict[str, Any], factory: Callable[[], Any] | None
) -> dict[str, Any]:
    sess_options = _build_options(factory)
    if sess_options is not None:
        kwargs["session_options"] = sess_options
    return kwargs


def loader_kwargs(
    device: str,
    settings: Settings,
    *,
    sess_options_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Argumentos de sesión para loaders que construyen la InferenceSession ellos.

    optimum y transformers no aceptan una sesión ya hecha, así que `create_session`
    no les sirve. Y un EP de plugin no se puede pedir por nombre: optimum valida el
    provider contra `get_available_providers()`, que no ve plugins. La forma que sí
    funciona es `providers=[]` con el EP puesto en el SessionOptions (verificado
    contra onnxruntime real, incluida la reutilización del mismo objeto en las
    varias sesiones que arma un pipeline de difusión).
    """
    import onnxruntime as ort

    if device == "cpu":
        return _with_session_options({"provider": CPU_PROVIDER}, sess_options_factory)
    if not device.startswith("dml:"):
        raise RuntimeError(f"Unsupported device for ONNX inference: {device!r}")

    _initialize(settings)
    plugin = _native_plugin_for(device)
    if plugin is not None and device not in _native_failed:
        sess_options = _build_options(sess_options_factory) or ort.SessionOptions()
        # Objeto nuevo por llamada: add_provider_for_devices dos veces sobre el
        # mismo SessionOptions falla con "Provider has already been registered".
        sess_options.add_provider_for_devices(plugin.devices[:1], {})
        return {"providers": [], "provider_options": [], "session_options": sess_options}

    return _with_session_options(
        {
            "provider": DML_PROVIDER,
            "provider_options": {"device_id": _parse_dml_device_id(device)},
        },
        sess_options_factory,
    )


def effective_provider_label(device: str) -> str | None:
    """Etiqueta humana del provider que las sesiones REALES usaron en el device.

    None si ningún trabajo creó todavía una sesión ahí (no inventar estado).
    """
    providers = _effective_providers.get(device)
    if not providers:
        return None
    primary = providers[0]
    if primary == CPU_PROVIDER:
        return "CPU (fallback)" if device.startswith("dml:") else "CPU"
    if primary == DML_PROVIDER:
        return "DirectML"
    for state in _plugins.values():
        if state.spec.ep_name == primary:
            return state.spec.label
    return primary


def active_ep_for_device(device: str, settings: Settings) -> EpStatus:
    if device == "cpu":
        return EpStatus(ep_name=CPU_PROVIDER, label="CPU", state=EP_STATE_BASELINE)
    _initialize(settings)
    if warmup_pending(device):
        return EpStatus(
            ep_name=DML_PROVIDER,
            label="DirectML",
            state=EP_STATE_PREPARING,
            detail="preparando aceleración para tu GPU",
        )
    effective = _effective_providers.get(device)
    if effective is not None and _is_cpu_fallback(device, effective):
        # Gana sobre cualquier otro estado: anunciar "DirectML" cuando las
        # sesiones reales corren en CPU es exactamente la mentira que este
        # estado existe para tapar.
        return EpStatus(
            ep_name=CPU_PROVIDER,
            label="CPU",
            state=EP_STATE_CPU_FALLBACK,
            detail="DirectML no inicializó en este dispositivo: los trabajos están corriendo en CPU",
        )
    session_error = _session_errors.get(device)
    if session_error:
        return EpStatus(
            ep_name=DML_PROVIDER, label="DirectML", state=EP_STATE_ERROR, detail=session_error
        )
    plugin = _native_plugin_for(device)
    if plugin is not None:
        # "native" solo cuando una sesion de verdad corrio ahi. Registrado pero
        # sin usar todavia es "ready": anunciar el acelerador antes de eso podia
        # ser mentira en una maquina donde despues todo cae a DirectML.
        state = EP_STATE_NATIVE if device in _native_ok else EP_STATE_READY
        return EpStatus(ep_name=plugin.spec.ep_name, label=plugin.spec.label, state=state)
    failed = _failed_plugin_for(device)
    if failed is not None:
        # Un plugin instalado que no pudo registrarse se veia EXACTAMENTE igual
        # que no tenerlo: el usuario bajaba el acelerador y no se enteraba.
        return EpStatus(
            ep_name=DML_PROVIDER,
            label="DirectML",
            state=EP_STATE_ERROR,
            detail=f"{failed.spec.label}: {failed.error}",
        )
    return EpStatus(ep_name=DML_PROVIDER, label="DirectML", state=EP_STATE_BASELINE)
