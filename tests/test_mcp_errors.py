import httpx

from app.mcp.client import UpflowUnavailableError
from app.mcp.errors import format_tool_error


def _http_error(status: int, detail: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://x")
    response = httpx.Response(status, json={"detail": detail}, request=request)
    return httpx.HTTPStatusError("msg", request=request, response=response)


def test_upflow_unavailable_passes_through():
    exc = UpflowUnavailableError("no se pudo conectar")
    assert format_tool_error(exc) == "no se pudo conectar"


def test_file_not_found_passes_through():
    exc = FileNotFoundError("el archivo no existe")
    assert format_tool_error(exc) == "el archivo no existe"


def test_http_404_mentions_detail():
    message = format_tool_error(_http_error(404, "Job not found"))
    assert message.startswith("Error")
    assert "Job not found" in message


def test_http_401_points_to_session_token():
    message = format_tool_error(_http_error(401, "not_authenticated"))
    assert message.startswith("Error")
    assert "UPFLOW_SESSION_TOKEN" in message


def test_http_409_mentions_detail():
    message = format_tool_error(_http_error(409, "Job not finished"))
    assert message.startswith("Error")
    assert "Job not finished" in message


def test_http_422_mentions_detail():
    message = format_tool_error(_http_error(422, "scale must be 2-4"))
    assert message.startswith("Error")
    assert "scale must be 2-4" in message


def test_http_429_suggests_waiting():
    message = format_tool_error(_http_error(429, "queue full"))
    assert message.startswith("Error")
    assert "queue full" in message
    assert "upflow_list_jobs" in message


def test_http_500_mentions_status_code():
    message = format_tool_error(_http_error(500, "boom"))
    assert message.startswith("Error")
    assert "500" in message


def test_timeout_mentions_timeout():
    message = format_tool_error(httpx.TimeoutException("slow"))
    assert "timeout" in message.lower()


def test_generic_exception_mentions_class_name():
    message = format_tool_error(ValueError("valor invalido"))
    assert "ValueError" in message
    assert "valor invalido" in message
