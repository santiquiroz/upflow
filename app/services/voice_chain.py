from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Destino de entrega. Los numeros no son gusto: son especificaciones publicadas
# y difieren por plataforma, asi que el usuario elige un DESTINO y no un numero
# crudo (verificado 2026-07-29).
DeliveryTarget = Literal["streaming", "apple_music", "ebu_r128", "atsc_a85", "none"]

# I = loudness integrado objetivo (LUFS). TP = techo de true peak (dBTP).
_DELIVERY_SPECS: dict[str, tuple[float, float, str]] = {
    # Spotify, YouTube, TIDAL, Amazon Music y SoundCloud convergen en -14.
    "streaming": (-14.0, -1.0, "Streaming (Spotify, YouTube, TIDAL, Amazon)"),
    "apple_music": (-16.0, -1.0, "Apple Music"),
    # EBU R128: broadcast y cine en Europa y buena parte del mundo.
    "ebu_r128": (-23.0, -1.0, "Broadcast / cine (EBU R128)"),
    # ATSC A/85: broadcast en EEUU. Techo de true peak mas permisivo.
    "atsc_a85": (-24.0, -2.0, "Broadcast EEUU (ATSC A/85)"),
}

DenoiseMode = Literal["rnnoise", "fft", "none"]
StepKind = Literal["filter", "model"]

# Banda de presencia de la voz humana. Subirla es lo que se siente como
# "enfocar" el dialogo en una mezcla.
_PRESENCE_HZ = 3000
_PRESENCE_WIDTH_HZ = 1500
# Por debajo de esto la voz no tiene contenido util, y sacarlo hace que el
# compresor no tenga que pelear con graves que empujan toda la senal.
_DEFAULT_HIGHPASS_HZ = 80


@dataclass(slots=True, frozen=True)
class VoiceChainOptions:
    denoise: DenoiseMode = "none"
    highpass_hz: int | None = _DEFAULT_HIGHPASS_HZ
    compress: bool = True
    presence_db: float | None = None
    deesser: bool = True
    delivery: DeliveryTarget = "none"


# El backend devuelve CLAVES de traduccion, no copia. Decision del spec de i18n
# de 2026-07-29, que revierte a proposito la de la cadena de voz: con dos
# idiomas, la unica fuente de verdad pasa a ser el catalogo de traducciones, y
# tener la copia aca significaria mantener dos catalogos en dos lenguajes
# distintos. Los DATOS siguen viniendo del backend; las oraciones las arma el
# frontend, que es la capa que conoce el locale.


@dataclass(slots=True, frozen=True)
class StepInfo:
    id: str
    label_key: str
    description_key: str
    kind: StepKind
    default_enabled: bool


def delivery_choices() -> list[dict[str, object]]:
    """Los destinos con sus numeros y las claves de su copia."""
    return [
        {
            "id": key,
            "labelKey": f"voice.delivery.{key}.label",
            "descriptionKey": f"voice.delivery.{key}.description",
            "lufs": lufs,
            "truePeakDb": tp,
        }
        for key, (lufs, tp, _label) in _DELIVERY_SPECS.items()
    ]


def step_catalog() -> list[StepInfo]:
    """Los pasos de la cadena, en el ORDEN correcto de ejecucion.

    El orden de esta lista no es alfabetico ni de importancia: es el que
    build_filter_chain documenta y respeta.
    """
    return [
        StepInfo("denoise", "voice.step.denoise.label",
                 "voice.step.denoise.description", "filter", False),
        StepInfo("highpass", "voice.step.highpass.label",
                 "voice.step.highpass.description", "filter", True),
        StepInfo("compress", "voice.step.compress.label",
                 "voice.step.compress.description", "filter", True),
        StepInfo("presence", "voice.step.presence.label",
                 "voice.step.presence.description", "filter", False),
        StepInfo("deesser", "voice.step.deesser.label",
                 "voice.step.deesser.description", "filter", True),
        StepInfo("loudness", "voice.step.loudness.label",
                 "voice.step.loudness.description", "filter", False),
    ]


def _denoise_filter(mode: DenoiseMode, rnnoise_model: str | None) -> str | None:
    if mode == "rnnoise":
        if not rnnoise_model:
            raise ValueError(
                "El modo de denoise 'rnnoise' necesita la ruta del modelo rnnoise."
            )
        # Mismo escape que AudioEnhancer._escape_filter_path: sin el, el ':'
        # de la letra de unidad en Windows rompe el parser de filtergraph
        # (verificado contra el ffmpeg vendoreado en audio_enhance.py).
        escaped = rnnoise_model.replace("\\", "/").replace(":", r"\\:")
        return f"arnndn=m={escaped}"
    if mode == "fft":
        return "afftdn"
    return None


def build_filter_chain(
    options: VoiceChainOptions,
    rnnoise_model: str | None = None,
) -> list[str]:
    """Cadena de filtros de ffmpeg para mejora de voz, en orden.

    El ORDEN no es preferencia, tiene causalidad documentada en la practica de
    mezcla de dialogo (verificado 2026-07-29):

    1. Reduccion de ruido primero: quitar es menos destructivo que agregar, y
       todo lo que venga despues trabaja sobre una senal mas limpia.
    2. Highpass antes del compresor: el contenido de graves empuja toda la
       senal, asi que sacarlo hace que el compresor no tenga que pelear.
    3. Compresor tecnico: redondea picos con mano suave.
    4. EQ de presencia: modela el tono ya con la dinamica controlada.
    5. De-esser al final del procesado tonal: se aplica sobre la sibilancia que
       realmente quedo, no sobre la que el EQ iba a cambiar despues.
    6. loudnorm de ultimo, siempre: mide loudness integrado de la senal
       TERMINADA. Medir antes de procesar daria un numero que ya no es cierto.
    """
    chain: list[str] = []

    denoise = _denoise_filter(options.denoise, rnnoise_model)
    if denoise:
        chain.append(denoise)

    if options.highpass_hz:
        chain.append(f"highpass=f={options.highpass_hz}")

    if options.compress:
        # Ratio moderado y umbral alto: control de picos, no color.
        chain.append("acompressor=threshold=-18dB:ratio=3:attack=5:release=120")

    if options.presence_db:
        chain.append(
            f"equalizer=f={_PRESENCE_HZ}:t=h:w={_PRESENCE_WIDTH_HZ}:g={options.presence_db}"
        )

    if options.deesser:
        chain.append("deesser")

    if options.delivery != "none":
        lufs, true_peak, _label = _DELIVERY_SPECS[options.delivery]
        # loudnorm aplica el techo de true peak por su cuenta, asi que no hace
        # falta un alimiter aparte detras.
        chain.append(f"loudnorm=I={lufs}:TP={true_peak}:LRA=11")

    return chain


def build_filter_argument(
    options: VoiceChainOptions,
    rnnoise_model: str | None = None,
) -> str | None:
    """La cadena lista para `-af`, o None si no hay nada que aplicar."""
    chain = build_filter_chain(options, rnnoise_model)
    return ",".join(chain) if chain else None


# ---------------------------------------------------------------------------
# Pasos y etapas: la cadena mezcla DSP y modelos
#
# Cada capacidad ofrece dos estrategias, y no son excluyentes: DSP (rapido,
# determinista, sin descarga) y modelo (mejor calidad, requiere instalarlo).
# Un paso de modelo NO puede vivir dentro de un filtergraph -- ffmpeg no sabe
# invocar DeepFilterNet a mitad de camino -- asi que corta la cadena en dos
# pasadas. Por eso los filtros consecutivos se fusionan en UNA pasada: cada
# pasada extra es un ciclo completo de decode y encode sobre todo el audio.
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ChainStep:
    id: str
    kind: StepKind
    label_key: str
    filter_expr: str | None = None
    model_capability: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "filter" and not self.filter_expr:
            raise ValueError(f"El paso de filtro {self.id!r} no trae expresion.")
        if self.kind == "model" and not self.model_capability:
            raise ValueError(f"El paso de modelo {self.id!r} no trae capacidad.")


@dataclass(slots=True, frozen=True)
class FfmpegStage:
    filter_argument: str
    step_ids: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ModelStage:
    model_capability: str
    step_id: str


Stage = FfmpegStage | ModelStage


def steps_from_selection(
    selected_ids: list[str],
    delivery: str | None = None,
    presence_db: float | None = None,
    denoise: DenoiseMode = "fft",
    highpass_hz: int = _DEFAULT_HIGHPASS_HZ,
    rnnoise_model: str | None = None,
) -> list[ChainStep]:
    """Traduce lo que el usuario tildo en la UI a pasos ejecutables.

    El ORDEN sale del catalogo, NO del orden en que llegaron los ids: la cadena
    tiene causalidad documentada y un request no deberia poder invertirla por
    accidente. Si en el futuro se quiere permitir reordenar, es una decision
    explicita y no un efecto de como el frontend serializo una lista.
    """
    wanted = set(selected_ids)
    options = VoiceChainOptions(
        denoise=denoise if "denoise" in wanted else "none",
        highpass_hz=highpass_hz if "highpass" in wanted else None,
        compress="compress" in wanted,
        presence_db=presence_db if "presence" in wanted else None,
        deesser="deesser" in wanted,
        delivery=delivery if ("loudness" in wanted and delivery) else "none",  # type: ignore[arg-type]
    )
    exprs = build_filter_chain(options, rnnoise_model)

    # build_filter_chain ya devuelve las expresiones en el orden correcto; solo
    # hay que emparejarlas con su entrada del catalogo para llevar la etiqueta.
    marker_by_step = {
        "denoise": ("afftdn", "arnndn"),
        "highpass": ("highpass",),
        "compress": ("acompressor",),
        "presence": ("equalizer",),
        "deesser": ("deesser",),
        "loudness": ("loudnorm",),
    }
    info_by_id = {info.id: info for info in step_catalog()}

    steps: list[ChainStep] = []
    for expr in exprs:
        for step_id, markers in marker_by_step.items():
            if any(expr.startswith(marker) for marker in markers):
                info = info_by_id[step_id]
                steps.append(
                    ChainStep(
                        id=info.id,
                        kind=info.kind,
                        label_key=info.label_key,
                        filter_expr=expr,
                    )
                )
                break
    return steps


def plan_stages(steps: list[ChainStep]) -> list[Stage]:
    """Agrupa los pasos en las pasadas reales de ejecucion.

    Filtros consecutivos colapsan en una sola FfmpegStage; cada paso de modelo
    es su propia ModelStage y fuerza el corte. El orden de los pasos se respeta
    tal cual: quien lo arma es responsable de que sea el correcto (ver el orden
    documentado en build_filter_chain).
    """
    stages: list[Stage] = []
    pending: list[ChainStep] = []

    def flush() -> None:
        if not pending:
            return
        stages.append(
            FfmpegStage(
                filter_argument=",".join(s.filter_expr or "" for s in pending),
                step_ids=tuple(s.id for s in pending),
            )
        )
        pending.clear()

    for step in steps:
        if step.kind == "model":
            flush()
            stages.append(
                ModelStage(
                    model_capability=step.model_capability or "",
                    step_id=step.id,
                )
            )
            continue
        pending.append(step)

    flush()
    return stages
