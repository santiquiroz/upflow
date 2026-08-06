"""Spike: ¿lo que produce Shap-E se puede imprimir?

Uso:
    .venv\Scripts\python scripts\spike_shape3d.py

Medido el 2026-08-05 en una máquina sin CUDA: 3 de 4 mallas salieron estancas,
manifold y sólidas directamente, en 116-137 s cada una.

Shap-E es MIT (OpenAI) y `diffusers` —que Upflow ya trae— lo soporta. Corre en
CPU: verificado, 120 s por malla en esta maquina.

La pregunta que decide todo no es si genera, sino si lo generado pasa el banco:
estanco, manifold y solido. Se lo pasa por el MISMO verificador que verifica un
STL bajado de internet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.services.mesh_fit import PRINTER_BEDS, fits_on_bed, scale_to_dimension  # noqa: E402
from app.services.mesh_inspect import inspect_mesh  # noqa: E402
from app.services.mesh_overhang import overhang_report  # noqa: E402
from app.services.mesh_repair import repair_mesh  # noqa: E402

PROMPTS = [
    "a simple mechanical bracket with two bolt holes",
    "a hexagonal nut",
    "a coffee mug",
    "a gear wheel",
]


def main() -> int:
    from diffusers import ShapEPipeline

    print("Cargando openai/shap-e (MIT) en CPU ...")
    pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float32).to("cpu")

    print()
    print(f"{'prompt':42s} {'tri':>7} {'estanca':>8} {'manif':>6} {'solida':>7} {'volad':>6} {'seg':>5}")
    resultados = []
    for prompt in PROMPTS:
        arranque = time.perf_counter()
        salida = pipe(
            prompt,
            guidance_scale=15.0,
            num_inference_steps=16,
            frame_size=64,
            output_type="mesh",
        )
        elapsed = time.perf_counter() - arranque

        malla = salida.images[0]
        verts = np.asarray(malla.verts, dtype=np.float64)
        faces = np.asarray(malla.faces, dtype=np.int64)
        triangulos = verts[faces]

        reporte = inspect_mesh(triangulos)
        voladizo = overhang_report(triangulos).overhang_area_ratio
        print(
            f"{prompt[:42]:42s} {reporte.triangle_count:>7} {str(reporte.is_watertight):>8} "
            f"{str(reporte.is_manifold):>6} {str(reporte.is_solid):>7} {voladizo:>5.0%} {elapsed:>5.0f}"
        )
        for problema in reporte.problems[:2]:
            print(f"      problema: {problema[:95]}")

        escalada = scale_to_dimension(triangulos, axis="x", millimetres=80.0)
        medida = inspect_mesh(escalada)
        cabe, _motivo = fits_on_bed(medida.size, bed=PRINTER_BEDS["ender-3"])
        print(
            f"      a 80 mm: {medida.size[0]:.0f} x {medida.size[1]:.0f} x {medida.size[2]:.0f} mm, "
            f"entra en Ender-3: {cabe}"
        )

        if not reporte.printable:
            _reparada, tras = repair_mesh(triangulos)
            print(
                f"      tras reparar: estanca={tras.is_watertight} manifold={tras.is_manifold} "
                f"solida={tras.is_solid}"
            )
            resultados.append(tras.printable)
        else:
            resultados.append(True)
        print()

    listos = sum(1 for r in resultados if r)
    print("================ VEREDICTO ================")
    print(f"{listos} de {len(PROMPTS)} mallas quedaron imprimibles (con reparacion si hizo falta).")
    if listos == 0:
        print("NO SIRVE tal cual: nada de lo generado pasa el banco.")
        return 1
    print("Shap-E genera geometria que el banco acepta. Lo que NO da: cotas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
