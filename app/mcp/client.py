"""HTTP client hacia el servidor Upflow local.

El MCP server NO importa servicios de la app: habla con la API REST del
servidor ya corriendo. Eso evita duplicar el lifespan (GPU, colas, sweeper)
y garantiza que MCP y web UI ven exactamente el mismo estado de jobs.

Auth: en AUTH_MODE=off (default desktop) no hace falta nada — loopback pasa.
En AUTH_MODE=multi la sesión es una cookie firmada (`upflow_session`); si
UPFLOW_USERNAME/UPFLOW_PASSWORD están seteados se hace login una vez y el
cookie jar del cliente persistente la reenvía en cada request.

No se manda Origin/Referer: OriginGuardMiddleware deja pasar requests sin
esos headers (clientes scripteados), así que httpx pasa limpio.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8090"
DEFAULT_TIMEOUT = 120.0
UPLOAD_TIMEOUT = 1800.0

_client: httpx.AsyncClient | None = None
_login_attempted = False


def base_url() -> str:
    return os.environ.get("UPFLOW_URL", DEFAULT_BASE_URL).rstrip("/")


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(base_url=base_url(), timeout=DEFAULT_TIMEOUT)
    return _client


async def close_client() -> None:
    global _client, _login_attempted
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
    _login_attempted = False


class UpflowUnavailableError(Exception):
    pass


def _connect_error_message() -> str:
    return (
        f"Error: no se pudo conectar al servidor Upflow en {base_url()}. "
        "Verificá que esté corriendo (Upflow.bat o "
        "`uvicorn app.main:app --port 8090`). Si usa otro puerto, "
        "seteá la variable de entorno UPFLOW_URL."
    )


async def _maybe_login(client: httpx.AsyncClient) -> bool:
    """Intenta login una sola vez si hay credenciales en el entorno."""
    global _login_attempted
    if _login_attempted:
        return False
    _login_attempted = True
    username = os.environ.get("UPFLOW_USERNAME", "")
    password = os.environ.get("UPFLOW_PASSWORD", "")
    if not username or not password:
        return False
    response = await client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    return response.status_code == 200


async def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    client = _get_client()
    try:
        response = await client.request(method, path, **kwargs)
        if response.status_code == 401 and await _maybe_login(client):
            response = await client.request(method, path, **kwargs)
    except httpx.ConnectError as exc:
        raise UpflowUnavailableError(_connect_error_message()) from exc
    response.raise_for_status()
    return response


async def api_get(path: str, *, params: dict[str, Any] | None = None) -> Any:
    response = await _request("GET", path, params=params)
    return response.json()


async def api_post(
    path: str,
    *,
    data: dict[str, Any] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    json_body: Any | None = None,
    timeout: float | None = None,
) -> Any:
    kwargs: dict[str, Any] = {"data": data, "files": files, "json": json_body}
    if timeout is not None:
        kwargs["timeout"] = timeout
    response = await _request("POST", path, **kwargs)
    if not response.content:
        return {"ok": True}
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return response.content


async def api_download(path: str, destination: Path, *, params: dict[str, Any] | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = _get_client()
    try:
        async with client.stream(
            "GET", path, params=params, timeout=UPLOAD_TIMEOUT
        ) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    handle.write(chunk)
    except httpx.ConnectError as exc:
        raise UpflowUnavailableError(_connect_error_message()) from exc
    return destination


def read_upload(file_path: str) -> tuple[str, bytes]:
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Error: el archivo '{file_path}' no existe o no es un archivo. "
            "Pasá una ruta absoluta a un archivo local."
        )
    return path.name, path.read_bytes()


def resolve_output_path(destination: str, default_name: str) -> Path:
    path = Path(destination).expanduser()
    if path.suffix == "" or path.is_dir():
        return path / default_name
    return path
