"""Ningun texto que vea el usuario puede mandarlo a abrir una terminal.

El 2026-08-05 el usuario lo dijo directo: "esto de colocar al usuario a usar la
terminal no va con lo intuitivo". Un barrido encontro 35 superficies que le
decian que corriera un `.ps1` — la app sabe correr esos scripts sola desde hace
varias versiones, pero el boton vivia en UNA sola pantalla.

Esta prueba es lo que impide que vuelva a pasar: cualquier mensaje nuevo que
nombre un script queda en rojo apenas se escribe, no cuando alguien lo reporta
desde su instalacion.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
APP = RAIZ / "app"
FRONTEND = RAIZ / "frontend" / "src"

MENCIONA_SCRIPT = re.compile(r"scripts[/\\]\S+\.ps1|\.ps1\b|npm run build|pip install")

# Comentarios y nombres de constantes SI pueden nombrar el script: explican de
# donde sale un archivo. Lo que no puede es un string que termine frente al
# usuario.
CAMPOS_QUE_VE_EL_USUARIO = {"detail", "reason", "message", "error"}


def strings_visibles(archivo: Path) -> list[tuple[int, str]]:
    """Literales que terminan frente al usuario: excepciones y campos de respuesta.

    Se parsea el AST en vez de barrer el texto: un comentario que explica de que
    script sale un binario es informacion util para quien lee el codigo, y
    marcarlo seria ruido que enseña a ignorar la prueba.
    """
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    encontrados: list[tuple[int, str]] = []

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Raise):
            for texto in ast.walk(nodo):
                if isinstance(texto, ast.Constant) and isinstance(texto.value, str):
                    encontrados.append((texto.lineno, texto.value))
        elif isinstance(nodo, ast.keyword) and nodo.arg in CAMPOS_QUE_VE_EL_USUARIO:
            for texto in ast.walk(nodo):
                if isinstance(texto, ast.Constant) and isinstance(texto.value, str):
                    encontrados.append((texto.lineno, texto.value))

    return encontrados


def ofensas_del_backend() -> list[str]:
    ofensas: list[str] = []
    for archivo in sorted(APP.rglob("*.py")):
        if "__pycache__" in archivo.parts:
            continue
        for linea, texto in strings_visibles(archivo):
            if MENCIONA_SCRIPT.search(texto):
                ofensas.append(f"{archivo.relative_to(RAIZ)}:{linea}: {texto[:80]}")
    return ofensas


def ofensas_del_frontend() -> list[str]:
    ofensas: list[str] = []
    for archivo in sorted(FRONTEND.rglob("*.ts")) + sorted(FRONTEND.rglob("*.tsx")):
        if archivo.name.endswith((".test.ts", ".test.tsx")):
            continue
        for numero, linea in enumerate(archivo.read_text(encoding="utf-8").splitlines(), 1):
            sin_comentario = linea.split("//", 1)[0]
            if MENCIONA_SCRIPT.search(sin_comentario):
                ofensas.append(f"{archivo.relative_to(RAIZ)}:{numero}: {linea.strip()[:80]}")
    return ofensas


AYUDA = (
    "\n\nEstos textos mandan al usuario a la terminal. La app puede bajar el pack "
    "sola: usa el mecanismo de packs (PACK_SCRIPTS + el boton de descarga) en vez "
    "de dictarle un comando."
)


@pytest.mark.parametrize("ofensas,donde", [(ofensas_del_backend(), "backend")])
def test_ningun_mensaje_del_backend_manda_a_la_terminal(ofensas: list[str], donde: str) -> None:
    assert ofensas == [], donde + AYUDA + "\n" + "\n".join(ofensas)


def test_ningun_texto_del_frontend_manda_a_la_terminal() -> None:
    ofensas = ofensas_del_frontend()
    assert ofensas == [], AYUDA + "\n" + "\n".join(ofensas)
