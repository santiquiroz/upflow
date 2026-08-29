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

# La semilla va fija por defecto porque esto alimenta un BANCO: dos corridas
# del mismo motor sobre la misma imagen tienen que dar la misma malla, o la
# comparacion entre motores mide el azar.
SEMILLA_POR_DEFECTO = 42

# El margen que deja el preprocesado oficial de TripoSG alrededor del objeto.
# Sale de `scripts/image_process.py::load_image(padding_ratio=0.1)`.
MARGEN_ENCUADRE = 0.1

# Un pixel mas claro que esto es fondo. Mismo criterio que usa el resto de
# Upflow para leer tinta sobre papel blanco.
UMBRAL_BLANCO = 244

# De donde salen los pesos. No se descargan desde aca: el motor "listo" o no
# lo decide el servicio, y bajar 8 GB no debe ser efecto de apretar generar.
REPO_DE_PESOS = "VAST-AI/TripoSG"


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


def pesos_en_cache() -> str:
    """La carpeta LOCAL con los pesos ya descargados.

    Hay que darle una carpeta y no el id del repo. Medido: con el id, diffusers
    resuelve el scheduler propio de TripoSG como si fuera un archivo remoto y
    falla con "scheduler/triposg.schedulers.scheduling_rectified_flow.py ... does
    not exist in VAST-AI/TripoSG"; con la carpeta, lo importa de `sys.path` y
    carga. Es la misma forma que usa el script oficial del repo.

    `local_files_only=True` es deliberado: esta funcion resuelve una ruta, no
    descarga. Si los pesos no estan, el motor no esta listo y eso se reporta
    como tal en vez de arrancar una descarga de 8 GB que nadie pidio.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO_DE_PESOS, local_files_only=True)


def encuadrar(ruta: str):
    """Deja la imagen como la espera el modelo: objeto centrado sobre blanco.

    Es una reimplementacion FIEL de `scripts/image_process.py::load_image` del
    propio TripoSG, y no un encuadre inventado. Se reimplementa porque el
    original esta cableado a `.cuda()` de punta a punta y esta maquina es AMD;
    llamarlo tal cual no falla al principio sino en la primera linea que toca
    la GPU. Los numeros —recortar a la caja del objeto, 10% de margen, cuadrar
    con relleno blanco— salen de ahi y no de una preferencia: cambiarlos
    significaria medir MI encuadre en vez del motor.

    El alfa se deriva de la tinta en vez de pedirle a BriaRMBG que quite el
    fondo. Estas entradas son dibujo de linea sobre papel blanco, donde el
    fondo ya esta separado; correr un quitafondos entrenado en fotos sobre
    arte plano agrega su propio criterio al resultado.
    """
    import numpy as np
    from PIL import Image

    with Image.open(ruta) as abierta:
        if abierta.mode == "RGBA":
            rgba = np.array(abierta)
            rgb = rgba[:, :, :3].astype(np.float32) / 255.0
            alfa = (rgba[:, :, 3] > 127).astype(np.float32)
        else:
            rgb = np.array(abierta.convert("RGB")).astype(np.float32) / 255.0
            alfa = (rgb.max(axis=2) * 255 < UMBRAL_BLANCO).astype(np.float32)

    filas = np.where(alfa.any(axis=1))[0]
    columnas = np.where(alfa.any(axis=0))[0]
    if not filas.size or not columnas.size:
        raise ValueError("la imagen no tiene ningun objeto sobre el fondo")

    # Sobre blanco: lo que no es objeto se borra, para que el modelo no lea
    # como geometria una firma o un marco del papel.
    compuesta = rgb * alfa[:, :, None] + (1.0 - alfa[:, :, None])
    y0, y1 = int(filas[0]), int(filas[-1]) + 1
    x0, x1 = int(columnas[0]), int(columnas[-1]) + 1
    recorte = compuesta[y0:y1, x0:x1]

    alto, ancho = recorte.shape[:2]
    if ancho > alto:
        izq = int(ancho * MARGEN_ENCUADRE)
        arriba = int(izq + (ancho - alto) / 2)
    else:
        arriba = int(alto * MARGEN_ENCUADRE)
        izq = int(arriba + (alto - ancho) / 2)

    cuadrada = np.pad(
        recorte,
        ((arriba, arriba), (izq, izq), (0, 0)),
        mode="constant",
        constant_values=1.0,
    )
    return Image.fromarray((cuadrada * 255).astype("uint8"))


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
        pesos = datos.get("weightsDir") or pesos_en_cache()
    except Exception as exc:  # noqa: BLE001
        fail(f"no estan los pesos del motor: {type(exc).__name__}: {exc}")

    semilla = int(datos.get("seed") or SEMILLA_POR_DEFECTO)
    try:
        entrada = encuadrar(imagen)
        tuberia = TripoSGPipeline.from_pretrained(pesos, torch_dtype=precision)
        tuberia.to(dispositivo)

        resultado = tuberia(
            image=entrada,
            # Semilla explicita: sin esto dos corridas del mismo motor dan
            # mallas distintas y el banco compara azar.
            generator=torch.Generator(device=dispositivo).manual_seed(semilla),
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
        "seed": semilla,
        "device": dispositivo,
        # Lo que sale de aca NO esta aprobado por salir: pasa por el banco como
        # cualquier otra malla.
        "audited": False,
    })


if __name__ == "__main__":
    main()
