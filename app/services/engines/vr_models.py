"""Modelos VR (UVR 5.1 CascadedNet) del catalogo de separacion: De-Echo y
De-Reverb de FoxJoy, los tres de la familia "Limpieza".

Politica de origen: los PESOS son de FoxJoy, distribuidos por el canal oficial
de descargas de Ultimate Vocal Remover (los .pth del release
all_public_uvr_models); los .onnx son un port propio, publico y MIT
(github.com/santiquiroz/port-uvr-deecho-onnx, release models-v1.0), conversion
mecanica de formato sin re-entrenar nada. `source_uvr_hash` deja anotado de que
checkpoint salio cada grafo (MD5 de los ultimos 10000 KiB del .pth, que es como
UVR identifica sus modelos); `sha256` es el del .onnx del release, tomado del
manifest.json de ESE release y verificado contra los archivos descargados el
2026-08-09.

A diferencia de MDX, aca el modelo predice DIRECTO la señal limpia (su
primary_stem es "No Echo" / "No Reverb") y el eco/reverb sale de la mascara
complementaria — no hay resta compensada, por eso no hay `compensate`.

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

_RELEASE_BASE = (
    "https://github.com/santiquiroz/port-uvr-deecho-onnx/releases/download/models-v1.0"
)

_STEM_NO_ECHO = "audio.stem.no_echo"
_STEM_ECHO = "audio.stem.echo"
_STEM_NO_REVERB = "audio.stem.no_reverb"
_STEM_REVERB = "audio.stem.reverb"


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
            SeparationStem("no_echo", _STEM_NO_ECHO, "primary"),
            SeparationStem("echo", _STEM_ECHO, "secondary"),
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
            SeparationStem("no_echo", _STEM_NO_ECHO, "primary"),
            SeparationStem("echo", _STEM_ECHO, "secondary"),
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
            SeparationStem("no_reverb", _STEM_NO_REVERB, "primary"),
            SeparationStem("reverb", _STEM_REVERB, "secondary"),
        ),
    ),
}
