"""Spike: ¿el carril de descripción a pieza ACOTADA funciona de punta a punta?

Uso:
    .venv\\Scripts\\python scripts\\spike_openscad_lane.py [modelo]

Ésta es la pregunta que decide el carril entero. Que un modelo emita algo que
parece OpenSCAD no sirve de nada: tiene que **compilar**, la pieza tiene que ser
**imprimible**, y sobre todo tiene que **medir lo que se pidió**. Esa última es la
única que separa esto de un generador de malla, y la única que importa para una
pieza que tiene que encajar.

Por eso el veredicto sale de MEDIR el STL producido, nunca de leer el código.

Y se mide el BUCLE, no solo el primer intento: cuando el código no compila, el
error del compilador vuelve al modelo. Un modelo que falla una vez suele acertar
al segundo con el error a la vista.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.services.mesh_inspect import inspect_mesh  # noqa: E402
from app.services.openscad_llm import LlmUnavailable, describe_to_stl  # noqa: E402
from app.services.stl_reader import read_stl  # noqa: E402

OPENSCAD = Path(r"C:/Program Files/OpenSCAD/openscad.exe")
TOLERANCIA_MM = 0.15

CASOS = [
    (
        "Un espaciador cilindrico de 20 mm de diametro exterior, 8.4 mm de agujero "
        "pasante centrado, y 12 mm de alto.",
        (20.0, 20.0, 12.0),
    ),
    (
        "Una placa rectangular de 60 mm de largo, 40 mm de ancho y 4 mm de espesor, "
        "con cuatro agujeros de 5 mm de diametro, uno en cada esquina, a 8 mm de cada borde.",
        (60.0, 40.0, 4.0),
    ),
    (
        "Un taco rectangular de 30 mm por 25 mm por 10 mm de alto, con un agujero "
        "pasante de 6 mm en el centro.",
        (30.0, 25.0, 10.0),
    ),
    (
        "Una escuadra plana en L: un brazo de 80 mm y otro de 50 mm, ambos de 18 mm "
        "de ancho y 5 mm de espesor, con un agujero de 6 mm cerca del extremo de cada brazo.",
        (80.0, 50.0, 5.0),
    ),
]


class OllamaClient:
    """Habla con `ollama run` directo. El servidor OpenAI-compatible es el camino
    de la app; acá se mide el modelo sin depender de que el servidor esté arriba."""

    def __init__(self, modelo: str) -> None:
        self.modelo = modelo

    def complete(self, system: str, user: str) -> str:
        proceso = subprocess.run(
            ["ollama", "run", self.modelo, f"{system}\n\n{user}"],
            capture_output=True,
            timeout=900,
        )
        return proceso.stdout.decode("utf-8", "replace")


def main() -> int:
    modelo = sys.argv[1] if len(sys.argv) > 1 else "qwen3-coder:30b"
    if not OPENSCAD.exists():
        print(f"Falta OpenSCAD en {OPENSCAD}")
        return 1

    cliente = OllamaClient(modelo)
    print(f"modelo: {modelo}\n")
    aciertos = 0
    reintentos_totales = 0

    for pedido, esperado in CASOS:
        arranque = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                resultado = describe_to_stl(
                    pedido,
                    client=cliente,
                    openscad=OPENSCAD,
                    destination=Path(tmp) / "pieza.stl",
                    # La realimentacion que importa: si mide mal, vuelve al modelo
                    # con la medida real contra la pedida.
                    expected_size=esperado,
                )
            except LlmUnavailable as exc:
                print(f"  FALLO   {pedido[:58]}\n          {str(exc)[:90]}")
                continue

            reporte = inspect_mesh(read_stl(resultado.stl_path))

        elapsed = time.perf_counter() - arranque
        reintentos_totales += resultado.retries
        medido = reporte.size
        cotas_ok = all(abs(m - e) < TOLERANCIA_MM for m, e in zip(medido, esperado))
        bien = cotas_ok and reporte.printable
        aciertos += 1 if bien else 0

        print(
            f"  {'OK  ' if bien else 'MAL '}    {pedido[:58]}\n"
            f"          pedido {esperado[0]:.0f}x{esperado[1]:.0f}x{esperado[2]:.0f}  "
            f"medido {medido[0]:.2f}x{medido[1]:.2f}x{medido[2]:.2f}  "
            f"imprimible={reporte.printable}  reintentos={resultado.retries}  {elapsed:.0f}s"
        )

    print("\n================ VEREDICTO ================")
    print(f"{aciertos} de {len(CASOS)} con las cotas exactas e imprimibles.")
    print(f"reintentos que hicieron falta: {reintentos_totales}")
    if aciertos == 0:
        print("EL CARRIL NO SIRVE con este modelo.")
        return 1
    print("Las cotas salen de la descripcion. Eso es lo que ningun generador de malla da.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
