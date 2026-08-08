"""Formateo consistente de errores para las tools MCP.

Cada tool devuelve SIEMPRE un string; los errores se convierten en mensajes
accionables en vez de propagar excepciones al cliente MCP.
"""

from __future__ import annotations

import httpx

from app.mcp.client import UpflowUnavailableError


def _detail_from_response(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict):
        return str(payload.get("detail", payload))[:300]
    return str(payload)[:300]


def format_tool_error(exc: Exception) -> str:
    if isinstance(exc, UpflowUnavailableError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        detail = _detail_from_response(exc.response)
        if status == 404:
            return f"Error: recurso no encontrado ({detail}). Verificá el job_id o la ruta."
        if status == 401:
            return (
                "Error: el servidor requiere autenticación (AUTH_MODE=multi). "
                "Seteá UPFLOW_SESSION_TOKEN con un token de sesión válido."
            )
        if status == 409:
            return f"Error: conflicto — {detail}"
        if status == 413:
            return f"Error: archivo demasiado grande — {detail}"
        if status == 422:
            return f"Error: parámetros inválidos — {detail}"
        if status == 429:
            return (
                f"Error: cola llena — {detail}. Esperá a que terminen jobs activos "
                "(upflow_list_jobs) y reintentá."
            )
        return f"Error: la API respondió {status} — {detail}"
    if isinstance(exc, httpx.TimeoutException):
        return "Error: timeout hablando con el servidor Upflow. Reintentá."
    return f"Error inesperado: {type(exc).__name__}: {exc}"
