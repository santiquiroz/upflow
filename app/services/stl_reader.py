"""Lee un STL, binario o ASCII, sin depender de nada externo.

El STL es una SOPA DE TRIANGULOS: no tiene indice de vertices, cada triangulo
repite sus tres puntos completos. Eso importa para todo lo que venga despues —
analizar la malla sobre los vertices crudos da que TODAS las aristas son borde,
porque ningun triangulo comparte memoria con su vecino. Aca solo se lee; soldar
y analizar viven aparte.

El formato se decide por el TAMANO, no por la palabra "solid": muchos
exportadores escriben "solid" en el header de un archivo binario, y decidir por
esa palabra rompe el archivo.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

BINARY_HEADER_BYTES = 80
TRIANGLE_RECORD_BYTES = 50  # normal (3f) + 3 vertices (9f) + atributo (H)


class StlUnreadable(RuntimeError):
    pass


def read_stl(path: Path) -> np.ndarray:
    """Devuelve los triangulos con forma (n, 3, 3): n triangulos, 3 vertices, xyz."""
    path = Path(path)
    if not path.exists():
        raise StlUnreadable(f"El archivo no existe: {path}")

    datos = path.read_bytes()
    triangulos = (
        _read_binary(datos) if _looks_binary(datos) else _read_ascii(datos, path)
    )
    if triangulos.size == 0:
        raise StlUnreadable("El archivo no tiene triangulos: una malla sin triangulos no es una malla.")
    return triangulos


def _looks_binary(datos: bytes) -> bool:
    """El tamano manda, no la palabra 'solid'.

    Un STL binario mide exactamente 84 + 50*n bytes. Se prueba esa cuenta
    primero, y solo si no cierra se mira el texto — porque muchos exportadores
    escriben "solid" en el header de un archivo BINARIO y esa palabra miente.
    Un binario cortado tampoco cierra la cuenta, asi que se lo reconoce por lo
    que NO tiene: la palabra `facet`, que todo ASCII repite por triangulo.
    """
    if len(datos) < BINARY_HEADER_BYTES + 4:
        return False
    declarados = struct.unpack_from("<I", datos, BINARY_HEADER_BYTES)[0]
    esperado = BINARY_HEADER_BYTES + 4 + declarados * TRIANGLE_RECORD_BYTES
    if len(datos) == esperado:
        return True
    return b"facet" not in datos[:4096].lower()


def _read_binary(datos: bytes) -> np.ndarray:
    cantidad = struct.unpack_from("<I", datos, BINARY_HEADER_BYTES)[0]
    cuerpo = datos[BINARY_HEADER_BYTES + 4 :]
    if len(cuerpo) < cantidad * TRIANGLE_RECORD_BYTES:
        raise StlUnreadable(
            f"El archivo esta incompleto: dice tener {cantidad} triangulos y "
            f"solo trae {len(cuerpo) // TRIANGLE_RECORD_BYTES}."
        )

    registros = np.frombuffer(
        cuerpo[: cantidad * TRIANGLE_RECORD_BYTES],
        dtype=np.dtype(
            [("normal", "<3f4"), ("vertices", "<9f4"), ("attr", "<u2")]
        ),
        count=cantidad,
    )
    return registros["vertices"].reshape(cantidad, 3, 3).astype(np.float64)


def _read_ascii(datos: bytes, path: Path) -> np.ndarray:
    try:
        texto = datos.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StlUnreadable(
            f"{path.name} no es un STL: ni binario valido ni texto legible."
        ) from exc

    vertices: list[list[float]] = []
    for linea in texto.splitlines():
        partes = linea.split()
        if not partes or partes[0] != "vertex":
            continue
        if len(partes) < 4:
            raise StlUnreadable(f"Vertice mal formado: {linea.strip()!r}")
        vertices.append([float(v) for v in partes[1:4]])

    if len(vertices) % 3 != 0:
        raise StlUnreadable(
            f"El archivo tiene {len(vertices)} vertices, que no forman triangulos enteros."
        )
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
