from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"

NETWORK_MODULES = frozenset(
    {
        "httpx",
        "requests",
        "urllib.request",
        "socket",
        "socketserver",
        "http.client",
        "urllib3",
        "aiohttp",
        "websocket",
        "websockets",
        "ftplib",
        "smtplib",
        "app.services.hf_client",
        "app.services.update_service",
        "app.services.download_job_manager",
        "app.services.model_installer",
        "app.services.openscad_llm",
        "app.mcp.client",
    }
)

RESTORE_MODULES = (
    "api/licenses_routes.py",
    "api/restore_routes.py",
    "headless_restore.py",
    "schemas_restore.py",
    "services/engines/colorize.py",
    "services/engines/drunet_restore.py",
    "services/engines/face_detect.py",
    "services/engines/face_restore.py",
    "services/engines/photo_restore_engine.py",
    "services/engines/restore_canary.py",
    "services/engines/scratch_detect.py",
    "services/engines/scratch_fill.py",
    "services/engines/tiled_restore_runner.py",
    "services/face_geometry.py",
    "services/image_io.py",
    "services/job_artifacts.py",
    "services/licenses_view.py",
    "services/png_chunks.py",
    "services/restore_session.py",
    "services/xmp_packet.py",
)

RESTORE_GLOBS = (
    "services/photo_*.py",
    "services/restore_*.py",
    "services/face_*.py",
    "services/engines/face_*.py",
    "services/engines/scratch_*.py",
)

# CCTV lands on another branch: these globs match nothing until it is merged.
CCTV_GLOBS = (
    "api/cctv_routes.py",
    "headless_cctv.py",
    "schemas_cctv.py",
    "services/cctv_*.py",
    "services/ffmpeg_*.py",
    "services/media_signature.py",
    "services/osd_check.py",
    "services/video_analysis.py",
    "services/roi_*.py",
    "services/frame_export.py",
    "services/side_by_side.py",
    "services/handover_package.py",
    "services/label_band.py",
    "services/engines/ffmpeg_frame_source.py",
    "services/engines/frame_model_runner.py",
    "services/engines/frame_restorer.py",
)


def is_network_module(name: str) -> bool:
    return any(name == banned or name.startswith(banned + ".") for banned in NETWORK_MODULES)


def module_package(path: Path) -> str:
    return ".".join(("app", *path.relative_to(APP).parent.parts))


def absolute_base(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    parts = package.split(".")[: len(package.split(".")) - node.level + 1]
    return ".".join([*parts, node.module] if node.module else parts)


def names_from_import(node: ast.Import) -> list[str]:
    return [alias.name for alias in node.names]


def names_from_import_from(node: ast.ImportFrom, package: str) -> list[str]:
    base = absolute_base(node, package)
    return [base, *(f"{base}.{alias.name}" for alias in node.names)]


def is_dynamic_import(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "__import__"
    return isinstance(func, ast.Attribute) and func.attr == "import_module"


def names_from_dynamic_import(node: ast.Call) -> list[str]:
    first = node.args[0] if node.args else None
    return [first.value] if isinstance(first, ast.Constant) and isinstance(first.value, str) else []


def imported_names(node: ast.AST, package: str) -> list[str]:
    if isinstance(node, ast.Import):
        return names_from_import(node)
    if isinstance(node, ast.ImportFrom):
        return names_from_import_from(node, package)
    if isinstance(node, ast.Call) and is_dynamic_import(node):
        return names_from_dynamic_import(node)
    return []


def network_imports(source: str, package: str = "app") -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(ast.parse(source)):
        found.extend((node.lineno, name) for name in imported_names(node, package) if is_network_module(name))
    return sorted(set(found))


def modules_matching(patterns: tuple[str, ...]) -> set[Path]:
    return {path for pattern in patterns for path in APP.glob(pattern)}


def local_only_modules() -> list[Path]:
    explicit = {APP / relative for relative in RESTORE_MODULES}
    return sorted(explicit | modules_matching(RESTORE_GLOBS) | modules_matching(CCTV_GLOBS))


@pytest.mark.parametrize(
    "source",
    [
        "import httpx",
        "import requests.adapters",
        "import socket as s",
        "from socket import create_connection",
        "import urllib.request",
        "from urllib.request import urlopen",
        "from urllib import parse, request",
        "import http.client",
        "def fetch():\n    import httpx\n    return httpx",
        "import importlib\nimportlib.import_module('requests')",
        "__import__('socket')",
        "from app.services.hf_client import HfClient",
        "from app.services import update_service",
    ],
)
def test_network_imports_are_detected(source: str) -> None:
    assert network_imports(source)


@pytest.mark.parametrize(
    "source",
    [
        "import json",
        "import urllib.parse",
        "from urllib.parse import quote",
        "from urllib import parse",
        "import socketlike",
        "import requests_cache_helper",
        "import importlib\nimportlib.import_module(name)",
        "from app.services import image_io",
    ],
)
def test_local_imports_are_not_flagged(source: str) -> None:
    assert network_imports(source) == []


@pytest.mark.parametrize(
    ("source", "package"),
    [
        ("from . import hf_client", "app.services"),
        ("from .update_service import check", "app.services"),
        ("from ..hf_client import HfClient", "app.services.engines"),
        ("from ..mcp import client", "app.api"),
    ],
)
def test_relative_imports_resolve_against_the_package(source: str, package: str) -> None:
    assert network_imports(source, package)


def test_relative_import_of_a_local_module_is_not_flagged() -> None:
    assert network_imports("from .image_io import load_image", "app.services") == []


def test_the_reported_line_is_the_import_line() -> None:
    assert network_imports("import json\n\nimport socket\n") == [(3, "socket")]


def test_every_listed_restore_module_exists() -> None:
    missing = [relative for relative in RESTORE_MODULES if not (APP / relative).is_file()]
    assert missing == []


def test_every_restore_glob_matches_a_module() -> None:
    empty = [pattern for pattern in RESTORE_GLOBS if not modules_matching((pattern,))]
    assert empty == []


def test_the_pipeline_modules_are_covered() -> None:
    covered = {path.relative_to(APP).as_posix() for path in local_only_modules()}
    assert {"services/photo_restore_pipeline.py", "services/restore_provenance.py"} <= covered


@pytest.mark.parametrize("path", local_only_modules(), ids=lambda path: path.relative_to(APP).as_posix())
def test_restore_and_cctv_modules_do_not_import_network_libraries(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert network_imports(source, module_package(path)) == []
