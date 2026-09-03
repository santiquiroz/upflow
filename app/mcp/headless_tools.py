"""Tools MCP que pueden ejecutar Upflow sin un servidor HTTP."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app import headless
from app.mcp import client

MODE_ENV = "UPFLOW_MCP_MODE"
AUTOSTART_TIMEOUT_SECONDS = 60
AUTOSTART_POLL_SECONDS = 1.0
# Inyectable: asyncio usa time.monotonic por dentro, parchearlo global rompe el loop.
_clock = time.monotonic

_context: headless.HeadlessContext | None = None


def mode() -> str:
    return os.environ.get(MODE_ENV, "auto").lower()


def get_context() -> headless.HeadlessContext:
    global _context
    if _context is None:
        _context = headless.build_context()
    return _context


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def server_reachable() -> bool:
    try:
        await client.api_get("/api/v1/health")
        return True
    except Exception:
        return False


async def use_server() -> bool:
    selected = mode()
    if selected == "server":
        return True
    if selected == "inprocess":
        return False
    return await server_reachable()


def inprocess_only() -> bool:
    return mode() == "inprocess"


def should_fallback(exc: Exception) -> bool:
    # Solo cuando el servidor NO esta (conexion rechazada): un 4xx/5xx real se
    # reporta como error del servidor, no se disimula corriendo in-process.
    return mode() == "auto" and isinstance(exc, client.UpflowUnavailableError)


async def upflow_list_models_headless() -> str:
    """Modelos de reescalado instalados, leidos del registro local (sin servidor)."""
    try:
        return _dump(await asyncio.to_thread(headless.list_models, get_context()))
    except Exception as exc:
        return f"Error: {exc}"


async def upflow_health() -> str:
    """Muestra la salud del servidor o del motor en proceso."""
    try:
        if await use_server():
            return _dump({"mode": "server", "server": await client.api_get("/api/v1/health")})
        return _dump({"mode": "inprocess", **await asyncio.to_thread(headless.health, get_context())})
    except Exception as exc:
        return f"Error: {exc}"


async def upflow_preflight_upscaler(repo_id: str) -> str:
    """Compatibilidad y capacidad (VRAM/disco/RAM) de un upscaler de Hugging Face,
    medido en proceso sin servidor. Equivale a `upflow preflight --repo`."""
    try:
        return _dump(await headless.preflight_model(get_context(), repo_id))
    except Exception as exc:
        return f"Error: {exc}"


async def upflow_install_upscaler(repo_id: str, confirm: bool = False) -> str:
    """Descarga e instala un upscaler de Hugging Face en proceso y ESPERA a que
    termine (equivale a `upflow install --repo --yes`). Sin confirm=true no descarga.
    Para instalar por el servidor (ASR, generacion, sin bloquear) usa upflow_install_model."""
    if not confirm:
        return "Error: downloading a model needs confirm=true (mirrors the CLI --yes)"
    try:
        return _dump(await headless.install_model(get_context(), repo_id))
    except Exception as exc:
        return f"Error: {exc}"


async def upflow_upscale_image_headless(
    file_path: str,
    destination_path: str = "",
    model: str = "realesrgan-x4plus",
    scale: int = 4,
    tile_size: int | None = None,
    tile_overlap: int | None = None,
    device: str = "",
    output_format: str = "",
) -> str:
    """Reescala una imagen local directamente, sin servidor."""
    try:
        source = Path(file_path)
        fmt = output_format or "png"
        destination = (
            Path(destination_path)
            if destination_path
            else source.parent / f"{source.stem}-upscaled.{fmt}"
        )
        result = await headless.upscale_image(
            get_context(),
            source,
            destination,
            model=model,
            scale=scale,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            device=device or None,
            output_format=output_format or None,
        )
        return _dump(result)
    except headless.HeadlessError as exc:
        return _dump({"ok": False, "error": str(exc), "code": exc.exit_code})
    except Exception as exc:
        return f"Error: {exc}"


async def autostart_server(port: int) -> bool:
    """Arranca Uvicorn si hace falta y espera a que responda."""
    if await server_reachable():
        return True
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    kwargs: dict[str, Any] = {
        "cwd": str(Path(__file__).resolve().parents[2]),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(command, **kwargs)
    except Exception:
        return False
    deadline = _clock() + AUTOSTART_TIMEOUT_SECONDS
    while _clock() < deadline:
        await asyncio.sleep(AUTOSTART_POLL_SECONDS)
        if await server_reachable():
            return True
    return False


def register(mcp: Any) -> None:
    """Registra estas funciones en la instancia FastMCP compartida."""
    read_only = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
    creates_job = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}
    mcp.tool(name="upflow_health", annotations={"title": "Salud de Upflow", **read_only})(upflow_health)
    mcp.tool(name="upflow_preflight_upscaler", annotations={"title": "Preflight de upscaler (sin servidor)", **read_only})(
        upflow_preflight_upscaler
    )
    mcp.tool(name="upflow_install_upscaler", annotations={"title": "Instalar upscaler (sin servidor)", **creates_job})(
        upflow_install_upscaler
    )
    mcp.tool(
        name="upflow_upscale_image_headless",
        annotations={"title": "Reescalar imagen sin servidor", **creates_job},
    )(upflow_upscale_image_headless)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsea las opciones de arranque del servidor MCP."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument("--mode", choices=("auto", "server", "inprocess"))
    parser.add_argument("--port", type=int)
    return parser.parse_args(argv)


def configured_port() -> int:
    parsed = urlparse(client.base_url())
    return parsed.port or 8090
