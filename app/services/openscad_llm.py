"""Pide una pieza en palabras y devuelve codigo OpenSCAD que COMPILA.

Este es el carril que cierra el circulo entre "describilo" y "esto mide 80 mm".
Un generador de malla nunca va a dar cotas — no hay medida que salga de un
prompt. Un modelo que escribe CODIGO CAD si, porque las cotas estan en el codigo.

Medido el 2026-08-05 con `qwen3-coder:30b` corriendo local: de tres piezas
pedidas en castellano, dos salieron con las cotas EXACTAS (20,00 x 20,00 x 12,00
y 30,00 x 25,00 x 10,00) y ambas imprimibles. La tercera no compilo — y ese es
justamente el caso que este modulo resuelve.

EL BUCLE ES LA PIEZA CENTRAL, y se cierra con DOS realimentaciones distintas:

  1. No compila -> vuelve el error del compilador.
  2. Compila pero MIDE MAL -> vuelve la medida real contra la pedida.

La segunda es la que importa de verdad, y salio de medir: con `qwen3-coder:30b`,
las cuatro piezas pedidas compilaron a la primera y las cuatro salieron
imprimibles, pero DOS midieron mal — un espaciador de 12 mm de alto salio de 24, y
una escuadra de 5 mm de espesor salio de 7,5. O sea que el fallo tipico de este
carril no es de sintaxis: es de COTAS, que es justo lo unico que este carril
existe para dar.

Sin esa segunda realimentacion, el usuario recibe una pieza que compila, que se
imprime, y que no entra donde tenia que entrar. El banco que ya verifica cualquier
STL es lo que permite cerrar ese bucle: se mide lo generado y se le dice al modelo
en que se equivoco.

El servidor del modelo es cualquiera que hable el protocolo de OpenAI: Ollama,
LM Studio, llama.cpp server. No se vendoriza ninguno — se apunta al que el
usuario ya tenga.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from app.services.mesh_inspect import inspect_mesh
from app.services.openscad_render import OpenScadError, RenderResult, render_to_stl
from app.services.stl_reader import read_stl

# Dos intentos extra. Con el error del compilador a la vista, el modelo suele
# arreglarlo al primero; a partir del tercero deja de mejorar y solo cuesta
# minutos.
DEFAULT_ATTEMPTS = 3

# Cuanto puede apartarse una medida de lo pedido antes de considerarla mal. Una
# decima de milimetro es mas fino que lo que cualquier impresora FDM resuelve, y
# mas grueso que el redondeo de la triangulacion de un circulo.
SIZE_TOLERANCE_MM = 0.15

SYSTEM_PROMPT = (
    "Sos un ingeniero mecanico que escribe OpenSCAD para imprimir en 3D. "
    "Respondes UNICAMENTE con codigo OpenSCAD dentro de un bloque ```openscad, "
    "sin explicaciones.\n"
    "Reglas que no se negocian:\n"
    "- Las medidas van en milimetros y son EXACTAS: la pieza tiene que medir lo "
    "que se pide.\n"
    "- La pieza tiene que ser un solido cerrado, sin caras sueltas.\n"
    "- Apoyala en el plano Z=0.\n"
    "- Usa $fn=64 en cilindros y esferas.\n"
    "- Los agujeros pasantes tienen que sobresalir del solido por los dos lados, "
    "o queda una cara de espesor cero.\n"
    "- No uses include, use, import ni surface: la pieza se describe con "
    "geometria, no leyendo archivos."
)


class LlmUnavailable(RuntimeError):
    pass


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True, slots=True)
class OpenAiCompatibleClient:
    """Habla el protocolo de OpenAI, que es el que exponen todos los servidores locales."""

    base_url: str
    model: str
    timeout_s: int = 600

    def complete(self, system: str, user: str) -> str:
        cuerpo = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # Baja a proposito: una pieza mecanica no se beneficia de que el
                # modelo sea creativo con las medidas.
                "temperature": 0.2,
                "stream": False,
            }
        ).encode("utf-8")
        peticion = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=cuerpo,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(peticion, timeout=self.timeout_s) as respuesta:
                datos = json.load(respuesta)
        except urllib.error.URLError as exc:
            raise LlmUnavailable(
                f"No se pudo hablar con el modelo en {self.base_url}: {exc}. "
                "Revisá que el servidor local esté corriendo."
            ) from exc
        try:
            return datos["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LlmUnavailable(
                f"El servidor respondió algo que no entiendo: {str(datos)[:200]}"
            ) from exc


@dataclass(frozen=True, slots=True)
class CodeAttempt:
    code: str
    error: str | None
    # Lo que midio la pieza de este intento, si llego a compilar.
    measured: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class DescribeResult:
    stl_path: Path
    code: str
    attempts: list[CodeAttempt]
    measured: tuple[float, float, float] | None = None

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)


def _size_mismatch(
    medido: tuple[float, float, float], esperado: tuple[float, float, float]
) -> str | None:
    """Que dimension quedo mal, dicho como para que el modelo pueda corregirla.

    Se comparan ORDENADAS de mayor a menor: al modelo se le pidio una pieza, no
    una orientacion, y rechazarla por estar acostada seria rechazar algo correcto.
    """
    m, e = sorted(medido, reverse=True), sorted(esperado, reverse=True)
    if all(abs(a - b) < SIZE_TOLERANCE_MM for a, b in zip(m, e)):
        return None
    return (
        f"La pieza tenia que medir {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm y midio "
        f"{m[0]:.2f} x {m[1]:.2f} x {m[2]:.2f} mm."
    )


def _size_retry_prompt(pedido: str, codigo: str, problema: str) -> str:
    return (
        f"PIEZA: {pedido}\n\n"
        f"Este codigo compila pero las medidas estan MAL:\n"
        f"```openscad\n{codigo}\n```\n\n"
        f"{problema}\n\n"
        "Errores tipicos que causan esto: centrar un cilindro con `center=true` "
        "duplica el alto visible, y sumar dos solidos apilados suma sus espesores.\n"
        "Devolve el codigo corregido, entero, sin explicaciones."
    )


def _retry_prompt(pedido: str, codigo: str, error: str) -> str:
    return (
        f"PIEZA: {pedido}\n\n"
        f"Este codigo NO compilo:\n```openscad\n{codigo}\n```\n\n"
        f"OpenSCAD dijo:\n{error}\n\n"
        "Devolve el codigo corregido, entero, sin explicaciones."
    )


def describe_to_stl(
    prompt: str,
    *,
    client: ChatClient,
    openscad: Path,
    destination: Path,
    expected_size: tuple[float, float, float] | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    render: Callable[..., RenderResult] = render_to_stl,
) -> DescribeResult:
    """Pide la pieza y no la da por buena hasta que compile Y mida lo pedido."""
    if not prompt.strip():
        raise LlmUnavailable("Hace falta una descripcion de la pieza.")
    if attempts < 1:
        raise ValueError("Hace falta al menos un intento.")

    historial: list[CodeAttempt] = []
    pedido = f"PIEZA: {prompt.strip()}"
    for numero in range(attempts):
        respuesta = client.complete(SYSTEM_PROMPT, pedido)
        try:
            resultado = render(respuesta, openscad=openscad, destination=destination)
        except OpenScadError as exc:
            error = str(exc)
            historial.append(CodeAttempt(code=respuesta, error=error))
            if numero == attempts - 1:
                raise LlmUnavailable(
                    f"El modelo no logro escribir codigo que compile en {attempts} "
                    f"intentos. El ultimo error fue: {error}"
                ) from exc
            pedido = _retry_prompt(prompt.strip(), respuesta, error)
            continue

        medido = tuple(inspect_mesh(read_stl(resultado.stl_path)).size)

        problema = _size_mismatch(medido, expected_size) if expected_size else None
        historial.append(CodeAttempt(code=respuesta, error=problema, measured=medido))
        if problema is not None and numero < attempts - 1:
            # Compila y se imprime, pero no entra donde tenia que entrar. Ese es
            # el fallo tipico de este carril, y el unico que importa.
            pedido = _size_retry_prompt(prompt.strip(), respuesta, problema)
            continue

        return DescribeResult(
            stl_path=resultado.stl_path,
            code=resultado.stl_path.with_suffix(".scad").read_text(encoding="utf-8"),
            attempts=historial,
            measured=medido,
        )

    raise LlmUnavailable("No se pudo generar la pieza.")
