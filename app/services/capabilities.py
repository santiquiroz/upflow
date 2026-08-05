from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.config import Settings, resolve_against_project_root
from app.services.model_registry import ModelKind

# Que puede hacer la app, como lo reconoce el usuario ("reescalar video"), no
# como esta organizado el codigo. El arbol de la UI se DERIVA de este catalogo:
# el frontend no puede mentir sobre lo que hay porque no decide.

CapabilityStatus = Literal["available", "needs_setup", "not_implemented"]

# De donde sale lo que la capacidad necesita. Son los dos regimenes que ya
# coexisten en el repo: el registro con su instalador, y los paquetes
# vendorizados que hasta ahora se bajaban a mano con un script.
# "builtin" = anda sin bajar nada, y es distinto de "none". `none` significa NO
# IMPLEMENTADA: el status ni se calcula contra la maquina. Sin esta distincion,
# una capacidad que funciona perfecto aparecia en el mapa de ruta.
Provisioning = Literal["registry", "vendored_pack", "builtin", "none"]

# Una capacidad puede resolverse por DSP (rapido, determinista, sin descarga) o
# por modelo (mejor calidad, hay que instalarlo), y no son excluyentes: lo
# interesante es encadenarlas. Ver voice_chain.plan_stages.
Strategy = Literal["dsp", "model"]

Domain = Literal["video", "image", "audio", "generate", "print"]


@dataclass(frozen=True, slots=True)
class PathRequirement:
    """Un archivo o directorio que tiene que existir en disco.

    El status se deriva de disco y NUNCA de un flag persistido: borrar la
    carpeta a mano no puede dejar la UI diciendo que la capacidad esta lista.
    """

    setting_attr: str
    # Paquete al que pertenece, que es lo que la Fase 2 va a saber descargar.
    pack: str


@dataclass(frozen=True, slots=True)
class RegistryRequirement:
    """Al menos un modelo instalado de alguno de estos kinds."""

    kinds: tuple[ModelKind, ...]


Requirement = PathRequirement | RegistryRequirement


@dataclass(frozen=True, slots=True)
class Capability:
    id: str
    domain: Domain
    # Clave de traduccion, no copia: la oracion la arma el frontend en el idioma
    # activo (ver el spec de i18n de 2026-07-29).
    label_key: str
    provisioning: Provisioning
    job_kind: str | None
    strategies: tuple[Strategy, ...]
    requirements: tuple[Requirement, ...] = ()
    unavailable_reason_key: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedCapability:
    id: str
    domain: Domain
    label_key: str
    status: CapabilityStatus
    provisioning: Provisioning
    job_kind: str | None
    strategies: tuple[Strategy, ...]
    # Paquetes que faltan bajar. Vacio si no falta nada o si el motivo es otro.
    missing_packs: tuple[str, ...] = ()
    unavailable_reason_key: str | None = None
    # Que requisito no se cumple, para que el aviso diga algo concreto en vez de
    # "no disponible".
    setup_reason_key: str | None = None


_UPSCALE_REQUIREMENTS: tuple[Requirement, ...] = (
    # El reescalado necesita las DOS cosas: el binario del motor en disco y al
    # menos un modelo en el registro. Chequear solo el registro mentiria, porque
    # las entradas builtin se siembran solas aunque el binario no este.
    PathRequirement("engine_binary", "realesrgan"),
    RegistryRequirement((ModelKind.builtin_ncnn, ModelKind.onnx)),
)


_PRINT_CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="print.check",
        domain="print",
        label_key="capability.print.check",
        # numpy en proceso: no hay pack que bajar ni modelo que instalar.
        provisioning="builtin",
        job_kind=None,
        strategies=("dsp",),
    ),
    Capability(
        id="print.repair",
        domain="print",
        label_key="capability.print.repair",
        # numpy en proceso, igual que el chequeo.
        provisioning="builtin",
        job_kind=None,
        strategies=("dsp",),
    ),
    Capability(
        id="print.slice",
        domain="print",
        label_key="capability.print.slice",
        provisioning="none",
        job_kind=None,
        strategies=("dsp",),
        unavailable_reason_key="capability.reason.noSlicerPack",
    ),
    Capability(
        id="print.generate",
        domain="print",
        label_key="capability.print.generate",
        provisioning="none",
        job_kind=None,
        strategies=("model",),
        # Verificado el 2026-08-05 contra las fuentes primarias, con revision
        # adversarial (docs/superpowers/specs/2026-08-05-3d-engines-license-findings.md).
        # La distincion importa: TRELLIS es MIT en codigo Y pesos — lo bloquea de
        # que DEPENDE (kernels CUDA y nvdiffrast, que si es NVIDIA no comercial),
        # no su propia licencia, asi que un puerto sin nvdiffrast lo devolveria a
        # la mesa. Hunyuan3D no vuelve: ahi el problema es la licencia del modelo.
        # TripoSR resistio el intento de refutacion: MIT en codigo y pesos, y su
        # camino sin CUDA verificado es torch en CPU, o sea lento, no imposible.
        unavailable_reason_key="capability.reason.noCleanLicenseMeshEngine",
    ),
)


CATALOG: tuple[Capability, ...] = (
    # --- video -------------------------------------------------------------
    Capability(
        id="video.upscale",
        domain="video",
        label_key="capability.video.upscale",
        provisioning="registry",
        job_kind="video",
        strategies=("model",),
        requirements=_UPSCALE_REQUIREMENTS,
    ),
    Capability(
        id="video.interpolate",
        domain="video",
        label_key="capability.video.interpolate",
        provisioning="vendored_pack",
        job_kind="video",
        strategies=("model",),
        requirements=(PathRequirement("rife_binary", "rife"),),
    ),
    Capability(
        id="video.subtitles",
        domain="video",
        label_key="capability.video.subtitles",
        provisioning="none",
        job_kind=None,
        strategies=("model",),
        unavailable_reason_key="capability.reason.subtitles",
    ),
    # --- imagen ------------------------------------------------------------
    Capability(
        id="image.upscale",
        domain="image",
        label_key="capability.image.upscale",
        provisioning="registry",
        job_kind="image",
        strategies=("model",),
        requirements=_UPSCALE_REQUIREMENTS,
    ),
    # --- audio -------------------------------------------------------------
    Capability(
        id="audio.denoise",
        domain="audio",
        label_key="capability.audio.denoise",
        provisioning="vendored_pack",
        job_kind="audio",
        # DeepFilterNet es modelo; el rnnoise de ffmpeg y afftdn son DSP.
        strategies=("dsp", "model"),
        requirements=(PathRequirement("deepfilter_binary", "deepfilternet"),),
    ),
    Capability(
        id="audio.restore",
        domain="audio",
        label_key="capability.audio.restore",
        provisioning="vendored_pack",
        job_kind="audio",
        strategies=("model",),
        requirements=(PathRequirement("apollo_restore_model", "apollo"),),
    ),
    Capability(
        id="audio.voice",
        domain="audio",
        label_key="capability.audio.voice",
        provisioning="vendored_pack",
        job_kind="audio",
        # La cadena entera es DSP hoy: el ffmpeg vendorizado ya trae deesser,
        # loudnorm, acompressor y equalizer. El paso por modelo (aislar la voz de
        # una mezcla) todavia no tiene motor.
        strategies=("dsp",),
        requirements=(PathRequirement("ffmpeg_binary", "ffmpeg"),),
    ),
    Capability(
        id="audio.transcribe",
        domain="audio",
        label_key="capability.audio.transcribe",
        provisioning="registry",
        job_kind="transcribe",
        strategies=("model",),
        requirements=(RegistryRequirement((ModelKind.asr_onnx,)),),
    ),
    Capability(
        id="audio.stems",
        domain="audio",
        label_key="capability.audio.stems",
        provisioning="none",
        job_kind=None,
        strategies=("model",),
        unavailable_reason_key="capability.reason.stems",
    ),
    # --- generacion --------------------------------------------------------
    Capability(
        id="generate.textToImage",
        domain="generate",
        label_key="capability.generate.textToImage",
        provisioning="registry",
        job_kind="generation",
        strategies=("model",),
        requirements=(RegistryRequirement((ModelKind.diffusion_onnx,)),),
    ),
    Capability(
        id="generate.imageToImage",
        domain="generate",
        label_key="capability.generate.imageToImage",
        provisioning="registry",
        job_kind="generation",
        strategies=("model",),
        # Reusa el MISMO modelo instalado que texto a imagen -- los pesos son los
        # mismos y solo cambia la clase de pipeline. La cobertura si es mas
        # angosta (no hay flux ni sana), pero eso lo valida el job manager contra
        # la clase declarada del modelo elegido, no el catalogo.
        requirements=(RegistryRequirement((ModelKind.diffusion_onnx,)),),
    ),
    Capability(
        id="generate.textToVideo",
        domain="generate",
        label_key="capability.generate.textToVideo",
        # Enviada en la v0.27.0. Corre por el lane Vulkan de sd.cpp, NO por ONNX:
        # el motivo viejo ("no hay camino a un ONNX ejecutable") describia una
        # ruta que nunca fue la de esta feature, y dejaba la pantalla de entrada
        # diciendo que algo ya enviado no existia.
        provisioning="vendored_pack",
        job_kind="generation",
        strategies=("model",),
        requirements=(PathRequirement("sdcpp_binary", "wan-video"),),
    ),
    Capability(
        id="generate.videoToVideo",
        domain="generate",
        label_key="capability.generate.videoToVideo",
        provisioning="none",
        job_kind=None,
        strategies=("model",),
        unavailable_reason_key="capability.reason.noOnnxPath",
    ),
    Capability(
        id="generate.textTo3d",
        domain="generate",
        label_key="capability.generate.textTo3d",
        provisioning="none",
        job_kind=None,
        strategies=("model",),
        unavailable_reason_key="capability.reason.noOnnxPath",
    ),
    Capability(
        id="generate.imageTo3d",
        domain="generate",
        label_key="capability.generate.imageTo3d",
        provisioning="none",
        job_kind=None,
        strategies=("model",),
        unavailable_reason_key="capability.reason.noOnnxPath",
    ),
    Capability(
        id="generate.textToSound",
        domain="generate",
        label_key="capability.generate.textToSound",
        provisioning="none",
        job_kind=None,
        strategies=("model",),
        unavailable_reason_key="capability.reason.noOnnxPath",
    ),
    Capability(
        id="generate.soundToSound",
        domain="generate",
        label_key="capability.generate.soundToSound",
        provisioning="none",
        job_kind=None,
        strategies=("model",),
        unavailable_reason_key="capability.reason.noOnnxPath",
    ),
    *_PRINT_CAPABILITIES,
)

DOMAIN_ORDER: tuple[Domain, ...] = ("video", "image", "audio", "generate", "print")


def _path_exists(settings: Settings, requirement: PathRequirement) -> bool:
    raw = getattr(settings, requirement.setting_attr, None)
    if not raw:
        return False
    return resolve_against_project_root(str(raw)).exists()


def _registry_has_kind(installed_kinds: frozenset[ModelKind], req: RegistryRequirement) -> bool:
    return any(kind in installed_kinds for kind in req.kinds)


def _unmet(
    capability: Capability,
    settings: Settings,
    installed_kinds: frozenset[ModelKind],
) -> tuple[Requirement, ...]:
    return tuple(
        requirement
        for requirement in capability.requirements
        if not _is_met(requirement, settings, installed_kinds)
    )


def _is_met(
    requirement: Requirement,
    settings: Settings,
    installed_kinds: frozenset[ModelKind],
) -> bool:
    if isinstance(requirement, PathRequirement):
        return _path_exists(settings, requirement)
    return _registry_has_kind(installed_kinds, requirement)


def _setup_reason_key(unmet: tuple[Requirement, ...]) -> str:
    # El paquete faltante manda sobre el modelo faltante: sin el binario no
    # sirve de nada instalar un modelo, asi que es lo primero que hay que decir.
    if any(isinstance(requirement, PathRequirement) for requirement in unmet):
        return "capability.setup.missingPack"
    return "capability.setup.missingModel"


def _resolve_one(
    capability: Capability,
    settings: Settings,
    installed_kinds: frozenset[ModelKind],
) -> ResolvedCapability:
    if capability.provisioning == "builtin":
        # Anda sin bajar nada: es codigo en proceso, no un pack en disco.
        return _as_resolved(capability, "available")

    if capability.provisioning == "none":
        # No implementada: el status no se calcula contra la maquina. Que existan
        # archivos sueltos no la vuelve disponible.
        return _as_resolved(capability, "not_implemented")

    unmet = _unmet(capability, settings, installed_kinds)
    if not unmet:
        return _as_resolved(capability, "available")

    return _as_resolved(
        capability,
        "needs_setup",
        missing_packs=tuple(
            requirement.pack for requirement in unmet if isinstance(requirement, PathRequirement)
        ),
        setup_reason_key=_setup_reason_key(unmet),
    )


def _as_resolved(
    capability: Capability,
    status: CapabilityStatus,
    missing_packs: tuple[str, ...] = (),
    setup_reason_key: str | None = None,
) -> ResolvedCapability:
    return ResolvedCapability(
        id=capability.id,
        domain=capability.domain,
        label_key=capability.label_key,
        status=status,
        provisioning=capability.provisioning,
        job_kind=capability.job_kind,
        strategies=capability.strategies,
        missing_packs=missing_packs,
        unavailable_reason_key=capability.unavailable_reason_key,
        setup_reason_key=setup_reason_key,
    )


def installed_kinds(registry: object) -> frozenset[ModelKind]:
    """Kinds que el registro tiene instalados y usables.

    Una entrada en estado `converting` o `error` no cuenta: existe en el
    registro pero no se puede usar todavia.
    """
    from app.services.model_registry import ModelStatus

    entries = registry.list()  # type: ignore[attr-defined]
    return frozenset(
        entry.kind for entry in entries if entry.status == ModelStatus.installed
    )


def resolve_capabilities(settings: Settings, registry: object) -> list[ResolvedCapability]:
    kinds = installed_kinds(registry)
    return [_resolve_one(capability, settings, kinds) for capability in CATALOG]


@dataclass(frozen=True, slots=True)
class CapabilityDomain:
    domain: Domain
    label_key: str
    # Las vivas y las del mapa de ruta van separadas a proposito: nueve inertes
    # intercaladas con seis vivas harian parecer la app mas vacia de lo que esta.
    capabilities: tuple[ResolvedCapability, ...] = ()
    roadmap: tuple[ResolvedCapability, ...] = field(default=())


def group_by_domain(resolved: list[ResolvedCapability]) -> list[CapabilityDomain]:
    return [
        CapabilityDomain(
            domain=domain,
            label_key=f"capability.domain.{domain}",
            capabilities=tuple(
                item
                for item in resolved
                if item.domain == domain and item.status != "not_implemented"
            ),
            roadmap=tuple(
                item
                for item in resolved
                if item.domain == domain and item.status == "not_implemented"
            ),
        )
        for domain in DOMAIN_ORDER
    ]
