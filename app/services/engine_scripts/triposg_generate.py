"""Genera una malla desde una imagen con TripoSG. Corre en el venv del motor.

Se invoca como `<python-del-motor> triposg_generate.py '<json>'` y contesta por
una linea de stdout con centinela, igual que los scripts de Blender. No se
importa desde la app: sus dependencias (numpy 1.22, su propio torch) chocan de
frente con las del resto del arbol.

TripoSG es MIT — codigo y pesos —, que es la razon por la que es el primero del
banco y no el mejor de la comparativa. De los tres motores que valia la pena
probar, es el unico que ademas de ser libre CORRE en esta maquina: TRELLIS.2 es
MIT pero CUDA-only, y Hunyuan3D 2.1 tiene camino AMD soportado pero su licencia
excluye la Union Europea, el Reino Unido y Corea del Sur.

SE FUERZA EL DECODIFICADOR LENTO (`use_flash_decoder=False`). El rapido pasa por
`diso`, que es CUDA-only; sin este parametro la tuberia se cae al final, despues
de haber gastado todos los minutos de CPU.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RESULT_SENTINEL = "UPFLOW_RESULT "

# Numero de pasos de difusion. TripoSG documenta 50; menos degrada la
# superficie antes que la silueta, que es justo lo que el banco mide.
PASOS_POR_DEFECTO = 50
GUIA_POR_DEFECTO = 7.0


def emit(datos: dict) -> None:
    print(RESULT_SENTINEL + json.dumps(datos, ensure_ascii=False))
    sys.stdout.flush()


def fail(mensaje: str) -> None:
    """Reporta el fallo por el mismo canal que el exito y termina bien.

    Igual que los scripts de Blender: el que llama distingue exito de fallo por
    el contenido del JSON. Mezclar canales hace que un warning de torch parezca
    un error de la operacion.
    """
    emit({"error": mensaje})
    sys.exit(0)


def payload() -> dict:
    return json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}


def main() -> None:
    datos = payload()
    imagen = datos.get("image", "")
    salida = datos.get("output", "")
    fuente = datos.get("sourceDir", "")
    pasos = int(datos.get("steps") or PASOS_POR_DEFECTO)
    guia = float(datos.get("guidance") or GUIA_POR_DEFECTO)
    caras_objetivo = int(datos.get("faceLimit") or 0)

    if not imagen or not salida:
        fail("faltan 'image' y 'output'")
    if not os.path.exists(imagen):
        fail(f"no existe la imagen: {imagen}")
    if not fuente or not os.path.isdir(fuente):
        fail(f"no existe el codigo del motor: {fuente}")

    sys.path.insert(0, fuente)

    try:
        import torch
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    except Exception as exc:  # noqa: BLE001 - el motor ausente es un dato, no un crash
        fail(f"el entorno del motor no esta completo: {type(exc).__name__}: {exc}")

    # CPU explicito: esta maquina es AMD y torch viaja sin CUDA. Dejarlo
    # implicito hace que un torch con CUDA en otra maquina cambie el resultado
    # sin que nadie lo haya pedido.
    dispositivo = datos.get("device") or "cpu"
    precision = torch.float32 if dispositivo == "cpu" else torch.float16

    try:
        from PIL import Image

        tuberia = TripoSGPipeline.from_pretrained("VAST-AI/TripoSG", torch_dtype=precision)
        tuberia.to(dispositivo)
        with Image.open(imagen) as abierta:
            entrada = abierta.convert("RGB")

        resultado = tuberia(
            image=entrada,
            num_inference_steps=pasos,
            guidance_scale=guia,
            # El decodificador rapido pasa por `diso`, que es CUDA-only.
            use_flash_decoder=False,
        )
        malla = resultado.samples[0]
    except Exception as exc:  # noqa: BLE001
        fail(f"la generacion fallo: {type(exc).__name__}: {exc}")

    try:
        import numpy as np
        import trimesh

        vertices = np.asarray(malla[0], dtype=np.float64)
        caras = np.asarray(malla[1], dtype=np.int64)
        objeto = trimesh.Trimesh(vertices=vertices, faces=caras)
        if caras_objetivo and len(objeto.faces) > caras_objetivo:
            objeto = objeto.simplify_quadric_decimation(caras_objetivo)

        Path(salida).parent.mkdir(parents=True, exist_ok=True)
        objeto.export(salida)
    except Exception as exc:  # noqa: BLE001
        fail(f"no se pudo escribir la malla: {type(exc).__name__}: {exc}")

    emit({
        "mesh": salida,
        "vertices": int(len(objeto.vertices)),
        "faces": int(len(objeto.faces)),
        "steps": pasos,
        "guidance": guia,
        "device": dispositivo,
        # Lo que sale de aca NO esta aprobado por salir: pasa por el banco como
        # cualquier otra malla.
        "audited": False,
    })


if __name__ == "__main__":
    main()
