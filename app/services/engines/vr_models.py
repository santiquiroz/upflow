"""Modelos VR (UVR 5.1 CascadedNet) del catalogo de separacion: De-Echo,
De-Reverb y De-Noise de FoxJoy, los cuatro de la familia "Limpieza".

Politica de origen: los PESOS son de FoxJoy, distribuidos por el canal oficial
de descargas de Ultimate Vocal Remover (los .pth del release
all_public_uvr_models); los .onnx son un port propio, publico y MIT
(github.com/santiquiroz/port-uvr-deecho-onnx, release models-v1.1), conversion
mecanica de formato sin re-entrenar nada. `source_uvr_hash` deja anotado de que
checkpoint salio cada grafo (MD5 de los ultimos 10000 KiB del .pth, que es como
UVR identifica sus modelos); `sha256` es el del .onnx del release, tomado del
manifest.json de ESE release y verificado contra los archivos descargados el
2026-08-10.

A diferencia de MDX, aca no hay resta compensada (por eso no hay `compensate`):
un stem es la mascara y el otro es su complemento. CUAL de los dos es la señal
limpia depende del modelo y NO se puede adivinar del grafo:

* De-Echo / De-Reverb: la mascara predice DIRECTO la señal limpia
  (primary_stem "No Echo" / "No Reverb"), el eco/reverb es el complemento.
* De-Noise: la mascara predice el RUIDO (primary_stem "Other"), asi que la
  señal limpia es el COMPLEMENTO. Por eso su `stems` arranca con el stem de
  source "secondary" — el orden del par es "lo que el usuario quiere primero",
  no "primary primero".

Ese mismo dato (primary_stem "Other" esta en NON_ACCOM_STEMS de UVR) ademas da
vuelta la curva de aggression a `1 - aggr`, y va en `is_non_accom_stem`: el
separador se lo pasa al driver y el port lo gatea contra la tabla de UVR.

El catalogo COMPLETO (MDX + VR) se arma en separation_models.py; este modulo
solo aporta la mitad VR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from app.services.engines.separation_spec import (
    Architecture,
    SeparationModelSpec,
    SeparationStem,
)

VR_SAMPLE_RATE = 44100

# Aggression de referencia para estos modelos en UVR (y la que gatea la paridad
# del port: golden capturado con aggression=5, window_size=512, batch_size=1,
# sin TTA ni post-process). "Aggressive" es OTRO checkpoint, no otra aggression.
VR_DEFAULT_AGGRESSION = 5.0

# models-v1.1 = los 4 grafos + su manifest. v1.0 (los 3 primeros) sigue
# publicado y con los MISMOS bytes; se repunta todo a v1.1 para que un solo tag
# describa el set completo. No fuerza re-descarga: el provisioner y el .ps1
# verifican el archivo que ya esta en disco por SHA-256, no por URL.
_RELEASE_BASE = (
    "https://github.com/santiquiroz/port-uvr-deecho-onnx/releases/download/models-v1.1"
)

_STEM_NO_ECHO = "audio.stem.no_echo"
_STEM_ECHO = "audio.stem.echo"
_STEM_NO_REVERB = "audio.stem.no_reverb"
_STEM_REVERB = "audio.stem.reverb"
_STEM_NO_NOISE = "audio.stem.no_noise"
_STEM_NOISE = "audio.stem.noise"


@dataclass(frozen=True, slots=True)
class VrModelSpec(SeparationModelSpec):
    # Clave del modelo en MODEL_SPECS del driver vendorizado
    # (app/services/engines/vr_deecho/pipeline.py): es el enlace entre esta
    # entrada del catalogo y los parametros que el port validó.
    vr_model_name: str
    # MD5 de los ultimos 10000 KiB del .pth de origen (hash UVR). Procedencia,
    # no verificacion: lo que se descarga es el .onnx exportado.
    source_uvr_hash: str
    # Canales de la ultima capa del CascadedNet. No lo usa la inferencia (esta
    # horneado en el grafo); queda para poder detectar drift contra el driver.
    nout: int
    # Curva de aggression que se aplica a la mascara despues de inferir.
    aggression: float
    # `primary_stem in NON_ACCOM_STEMS` de UVR. Da vuelta el exponente de la
    # aggression a `1 - aggr`. No es cosmetico: con el valor equivocado sale
    # una mascara silenciosamente distinta, no un error. El port lo gatea
    # contra la tabla del propio UVR (tests/test_aggression.py).
    is_non_accom_stem: bool = False

    architecture: ClassVar[Architecture] = "vr"


VR_MODELS: dict[str, VrModelSpec] = {
    "deecho_normal": VrModelSpec(
        id="deecho_normal",
        name="UVR De-Echo Normal by FoxJoy",
        filename="UVR-De-Echo-Normal.onnx",
        url=f"{_RELEASE_BASE}/UVR-De-Echo-Normal.onnx",
        sha256="fc2f9df26060672b72324d6f77a046812361fd8a0dc79ba4f5258a944fc45e14",
        vr_model_name="UVR-De-Echo-Normal",
        source_uvr_hash="f200a145434efc7dcf0cd093f517ed52",
        nout=48,
        aggression=VR_DEFAULT_AGGRESSION,
        primary_stem="No Echo",
        category="cleanup",
        description_key="audio.karaoke.model.deecho_normal.description",
        stems=(
            SeparationStem("no_echo", _STEM_NO_ECHO, 0),
            SeparationStem("echo", _STEM_ECHO, 1),
        ),
    ),
    "deecho_aggressive": VrModelSpec(
        id="deecho_aggressive",
        name="UVR De-Echo Aggressive by FoxJoy",
        filename="UVR-De-Echo-Aggressive.onnx",
        url=f"{_RELEASE_BASE}/UVR-De-Echo-Aggressive.onnx",
        sha256="c5f95ecf29cb0be50144ea0ab461ac920854576df47c3ede82420846f699037c",
        vr_model_name="UVR-De-Echo-Aggressive",
        source_uvr_hash="6857b2972e1754913aad0c9a1678c753",
        nout=48,
        aggression=VR_DEFAULT_AGGRESSION,
        primary_stem="No Echo",
        category="cleanup",
        description_key="audio.karaoke.model.deecho_aggressive.description",
        stems=(
            SeparationStem("no_echo", _STEM_NO_ECHO, 0),
            SeparationStem("echo", _STEM_ECHO, 1),
        ),
    ),
    "deecho_dereverb": VrModelSpec(
        id="deecho_dereverb",
        name="UVR DeEcho-DeReverb by FoxJoy",
        filename="UVR-DeEcho-DeReverb.onnx",
        url=f"{_RELEASE_BASE}/UVR-DeEcho-DeReverb.onnx",
        sha256="fe64dfbbeb744cf8a648a25a473ce319bbfb59771eac01f4ff47a77312839bd3",
        vr_model_name="UVR-DeEcho-DeReverb",
        source_uvr_hash="0fb9249ffe4ffc38d7b16243f394c0ff",
        nout=64,
        aggression=VR_DEFAULT_AGGRESSION,
        primary_stem="No Reverb",
        category="cleanup",
        description_key="audio.karaoke.model.deecho_dereverb.description",
        stems=(
            SeparationStem("no_reverb", _STEM_NO_REVERB, 0),
            SeparationStem("reverb", _STEM_REVERB, 1),
        ),
    ),
    "denoise": VrModelSpec(
        id="denoise",
        name="UVR DeNoise by FoxJoy",
        filename="UVR-DeNoise.onnx",
        url=f"{_RELEASE_BASE}/UVR-DeNoise.onnx",
        sha256="3285c155a0f8f7295ad971e1fb43fdcb9d8cdbc493c28aececcddb61af26cc63",
        vr_model_name="UVR-DeNoise",
        source_uvr_hash="44c55d8b5d2e3edea98c2b2bf93071c7",
        nout=48,
        aggression=VR_DEFAULT_AGGRESSION,
        # El unico del catalogo VR cuya mascara saca el RUIDO y no la señal
        # limpia; de ahi el par de stems invertido y is_non_accom_stem.
        primary_stem="Noise",
        is_non_accom_stem=True,
        category="cleanup",
        description_key="audio.karaoke.model.denoise.description",
        stems=(
            SeparationStem("no_noise", _STEM_NO_NOISE, 1),
            SeparationStem("noise", _STEM_NOISE, 0),
        ),
    ),
}
