"""Genera una malla desde una imagen con Hunyuan3D 2.1. Corre en el venv del motor.

Mismo contrato de centinela que el resto: se invoca como
`<python-del-motor> hunyuan3d_generate.py '<json>'` y contesta por una linea de
stdout. No se importa desde la app.

SOLO FORMA, NUNCA TEXTURA, y es una decision y no una limitacion que se sufre:
la textura de Hunyuan3D pasa por `custom_rasterizer`, que hay que compilar como
kernels HIP y que en AMD viene reportando salidas corruptas. La forma, en
cambio, es PyTorch puro y corre tal cual. Ademas el banco mide SILUETA: una
textura preciosa no mueve el numero ni un punto, asi que pagar esa compilacion
seria gastar en algo que no se mide.

LICENCIA, que no es un detalle administrativo: los pesos son
`tencent-hunyuan-community`, NO una licencia libre. Permite uso comercial por
debajo de 1 millon de usuarios activos mensuales pero EXCLUYE la Union Europea,
el Reino Unido y Corea del Sur. Por eso este motor viaja marcado como
restringido y no se descarga solo: quien lo instala esta aceptando eso, y tiene
que poder saberlo antes.

GPU: en un Ryzen con grafica integrada, ROCm enumera PRIMERO la iGPU. El
servicio fija `HIP_VISIBLE_DEVICES` antes de lanzar este proceso; aca solo se
comprueba que el dispositivo elegido sea el que se esperaba, porque correr en la
integrada no falla al arrancar sino que revienta al primer calculo.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RESULT_SENTINEL = "UPFLOW_RESULT "

PASOS_POR_DEFECTO = 50
GUIA_POR_DEFECTO = 5.0

# Igual que en TripoSG: semilla fija porque esto alimenta un banco, y sin ella
# dos corridas del mismo motor sobre la misma imagen dan mallas distintas.
SEMILLA_POR_DEFECTO = 42

REPO_DE_PESOS = "tencent/Hunyuan3D-2.1"
LICENCIA = "tencent-hunyuan-community"


def emit(datos: dict) -> None:
    print(RESULT_SENTINEL + json.dumps(datos, ensure_ascii=False))
    sys.stdout.flush()


def fail(mensaje: str) -> None:
    emit({"error": mensaje})
    sys.exit(0)


def payload() -> dict:
    return json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}


def entrada_rgba(ruta: str):
    """La imagen con fondo transparente, que es lo que la tuberia espera.

    Si ya trae alfa se respeta. Si no, se deriva de la tinta en vez de correr el
    quitafondos: estas entradas son dibujo de linea sobre papel blanco, donde el
    fondo ya esta separado, y un quitafondos entrenado en fotos agrega su propio
    criterio sobre arte plano.
    """
    import numpy as np
    from PIL import Image

    with Image.open(ruta) as abierta:
        if abierta.mode == "RGBA":
            return abierta.copy()
        rgb = abierta.convert("RGB")

    datos = np.array(rgb)
    tinta = (datos.max(axis=2) < 244).astype("uint8") * 255
    rgba = np.dstack([datos, tinta])
    return Image.fromarray(rgba, mode="RGBA")


def atencion_compatible() -> None:
    """Ultimo recurso si el backend de atencion sigue sin existir.

    Lo normal es que el servicio lance este proceso con
    `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, y entonces los tres backends de
    `scaled_dot_product_attention` funcionan en gfx1101. Sin esa variable, flash
    y memory-efficient revientan con "No available kernel. Aborting execution."
    —un mensaje que no nombra ni la atencion ni la GPU— y solo queda el
    matematico. Se comprueba y se cae a matematico en vez de morir: mas lento es
    mejor que no correr.

    Los interruptores globales NO alcanzan si la tuberia abre su propio contexto
    de backend, cosa que se midio: por eso el arreglo de verdad es la variable
    de entorno y esto es la red.
    """
    import torch
    import torch.nn.functional as F

    prueba = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.float16)
    try:
        F.scaled_dot_product_attention(prueba, prueba, prueba)
        torch.cuda.synchronize()
        return
    except RuntimeError:
        pass

    for apagar in ("enable_flash_sdp", "enable_mem_efficient_sdp"):
        interruptor = getattr(torch.backends.cuda, apagar, None)
        if interruptor is not None:
            interruptor(False)
    encender = getattr(torch.backends.cuda, "enable_math_sdp", None)
    if encender is not None:
        encender(True)


def comprobar_gpu(dispositivo: str) -> dict:
    import torch

    if dispositivo == "cpu":
        return {"device": "cpu"}
    if not torch.cuda.is_available():
        fail("se pidio GPU y torch no ve ninguna: revisar la instalacion de ROCm/CUDA")
    props = torch.cuda.get_device_properties(0)
    return {
        "device": dispositivo,
        "gpu": props.name,
        "arch": getattr(props, "gcnArchName", ""),
        "vramGb": round(props.total_memory / 1e9, 1),
    }


def main() -> None:
    datos = payload()
    imagen = datos.get("image", "")
    salida = datos.get("output", "")
    fuente = datos.get("sourceDir", "")
    pasos = int(datos.get("steps") or PASOS_POR_DEFECTO)
    guia = float(datos.get("guidance") or GUIA_POR_DEFECTO)
    semilla = int(datos.get("seed") or SEMILLA_POR_DEFECTO)
    dispositivo = datos.get("device") or "cuda"

    if not imagen or not salida:
        fail("faltan 'image' y 'output'")
    if not os.path.exists(imagen):
        fail(f"no existe la imagen: {imagen}")
    if not fuente or not os.path.isdir(fuente):
        fail(f"no existe el codigo del motor: {fuente}")

    sys.path.insert(0, os.path.join(fuente, "hy3dshape"))

    try:
        import torch
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    except Exception as exc:  # noqa: BLE001
        fail(f"el entorno del motor no esta completo: {type(exc).__name__}: {exc}")

    equipo = comprobar_gpu(dispositivo)
    if dispositivo != "cpu":
        atencion_compatible()

    try:
        tuberia = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(REPO_DE_PESOS)
        tuberia.to(dispositivo)
        resultado = tuberia(
            image=entrada_rgba(imagen),
            num_inference_steps=pasos,
            guidance_scale=guia,
            generator=torch.Generator(device=dispositivo).manual_seed(semilla),
        )
        malla = resultado[0]
    except Exception as exc:  # noqa: BLE001
        fail(f"la generacion fallo: {type(exc).__name__}: {exc}")

    try:
        Path(salida).parent.mkdir(parents=True, exist_ok=True)
        malla.export(salida)
    except Exception as exc:  # noqa: BLE001
        fail(f"no se pudo escribir la malla: {type(exc).__name__}: {exc}")

    emit({
        "mesh": salida,
        "vertices": int(len(malla.vertices)),
        "faces": int(len(malla.faces)),
        "steps": pasos,
        "guidance": guia,
        "seed": semilla,
        "license": LICENCIA,
        "textured": False,
        # Igual que todo lo generado: NO esta aprobado por haber salido.
        "audited": False,
        **equipo,
    })


if __name__ == "__main__":
    main()
