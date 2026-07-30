from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import aiofiles
import httpx

from app.config import Settings
from app.exceptions import HfAuthError, HfDownloadTooLargeError, HfInvalidSourceError

logger = logging.getLogger(__name__)

# A model download is tens-to-hundreds of MB over the public internet; a single
# transient blip shouldn't abort the whole install. Retries restart the transfer
# from zero (the temp file is opened 'wb'), so keep the attempt count low.
DOWNLOAD_ATTEMPTS = 3
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_TRANSIENT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


def _is_retryable_download_error(exc: BaseException) -> bool:
    """Only transient transport/server failures retry. A 401/404 (bad token, wrong
    file) or a size-limit rejection is permanent and must surface immediately."""
    if isinstance(exc, _TRANSIENT_ERRORS):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


def _wrap_hf_auth_error(exc: Exception, repo_id: str) -> Exception:
    if not isinstance(exc, httpx.HTTPStatusError):
        return exc
    status = exc.response.status_code
    if status == 401:
        return HfAuthError(
            "Tu HF_TOKEN no es válido o no está configurado — revisalo en Settings."
        )
    if status == 403:
        return HfAuthError(
            f"El repo {repo_id} requiere aceptar su licencia en "
            f"huggingface.co/{repo_id} antes de poder descargarlo."
        )
    return exc


# ---------------------------------------------------------------------------
# Hugging Face REST endpoints used here (verified live against the real API,
# 2026-07-14):
#
#   search:      GET https://huggingface.co/api/models
#                    ?search=<q>&filter=image-to-image&filter=super-resolution
#                    &limit=<n>&full=true
#                Repeated `filter` params are ANDed by the Hub, so this
#                narrows results to models tagged with BOTH task tags.
#
#   repo_files:  GET https://huggingface.co/api/models/<repo_id>?blobs=true
#                `siblings[].size` is only populated with blobs=true; without
#                it, siblings only carry `rfilename`.
#
#   download:    GET https://huggingface.co/<repo_id>/resolve/main/<filename>
#                Real LFS-backed weights answer with a redirect to a
#                different host (observed: cas-bridge.xethub.hf.co) carrying
#                a pre-signed, time-limited URL. httpx's follow_redirects
#                must chase that -- host validation below therefore only
#                guards the *initial* request we build, not the redirect
#                target, matching how huggingface_hub's own downloader works.
# ---------------------------------------------------------------------------

HF_HOST = "huggingface.co"
HF_API_BASE = "https://huggingface.co/api"
HF_RESOLVE_BASE = "https://huggingface.co"
SEARCH_TASK_TAGS = ("image-to-image", "super-resolution")
GENERATION_SEARCH_TASK_TAGS = ("text-to-image",)
# El tag es el filtro REAL de que es un modelo de ASR: los nombres de archivo no
# alcanzan para distinguirlo de otro modelo de audio (ver AsrCompatStrategy).
ASR_SEARCH_TASK_TAGS = ("automatic-speech-recognition",)
WEIGHT_EXTENSION_PRIORITY = (".onnx", ".safetensors", ".pth")
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30.0
PARTIAL_DOWNLOAD_SUFFIX = ".part"
# Cota del header de un .safetensors leido por Range. Medidos en repos reales:
# 25 KB (un VAE) a 358 KB (un SDXL de 2515 tensores). 64 MB deja mas de 100x de
# margen y acota el gasto si un repo declara un largo absurdo.
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024

ProgressCallback = Callable[[int, "int | None"], None]


@dataclass(slots=True, frozen=True)
class HfModelSummary:
    id: str
    author: str | None
    pipeline_tag: str | None
    downloads: int
    likes: int
    tags: tuple[str, ...]
    # `full=true` trae estos datos en la misma respuesta de busqueda; no hace
    # falta un request adicional por resultado para clasificar compatibilidad.
    filenames: tuple[str, ...] = ()
    gated: bool | str | None = None


@dataclass(slots=True, frozen=True)
class HfFile:
    path: str
    size: int


def _parse_model_summary(item: dict) -> HfModelSummary:
    siblings = item.get("siblings") or []
    return HfModelSummary(
        id=item["id"],
        author=item.get("author"),
        pipeline_tag=item.get("pipeline_tag"),
        downloads=item.get("downloads", 0),
        likes=item.get("likes", 0),
        tags=tuple(item.get("tags", [])),
        filenames=tuple(
            sibling["rfilename"]
            for sibling in siblings
            if "rfilename" in sibling
        ),
        gated=item.get("gated"),
    )


def _parse_sibling_file(sibling: dict) -> HfFile:
    return HfFile(path=sibling["rfilename"], size=sibling.get("size", 0))


def _weight_extension_rank(file: HfFile) -> int | None:
    suffix = Path(file.path).suffix.lower()
    if suffix not in WEIGHT_EXTENSION_PRIORITY:
        return None
    return WEIGHT_EXTENSION_PRIORITY.index(suffix)


def pick_weight_file(files: list[HfFile]) -> HfFile:
    """Picks the preferred weight file: .onnx > .safetensors > .pth.

    When several files share the winning extension, the largest one wins
    (e.g. fp32 over fp16 opset variants).
    """
    ranked = [(rank, file) for file in files if (rank := _weight_extension_rank(file)) is not None]
    if not ranked:
        raise ValueError("No .onnx/.safetensors/.pth weight file found")
    best_rank = min(rank for rank, _ in ranked)
    candidates = [file for rank, file in ranked if rank == best_rank]
    return max(candidates, key=lambda file: file.size)


def _download_url(repo_id: str, filename: str) -> str:
    return f"{HF_RESOLVE_BASE}/{repo_id}/resolve/main/{quote(filename)}"


def _validate_https_huggingface_host(url: str) -> None:
    parsed = httpx.URL(url)
    if parsed.scheme != "https" or parsed.host != HF_HOST:
        raise HfInvalidSourceError(f"Download source must be HTTPS {HF_HOST}, got: {url!r}")


def _parse_content_length(headers: httpx.Headers) -> int | None:
    value = headers.get("content-length")
    return int(value) if value is not None else None


def _reject_declared_size_over_limit(
    total: int | None,
    max_bytes: int | None,
) -> None:
    if max_bytes is None or total is None:
        return
    if total > max_bytes:
        raise HfDownloadTooLargeError(
            f"Declared size {total} bytes exceeds MAX_MODEL_DOWNLOAD_MB limit ({max_bytes} bytes)"
        )


async def _write_response_to_file(
    response: httpx.Response,
    tmp_path: Path,
    max_bytes: int | None,
    total: int | None,
    progress_cb: ProgressCallback | None,
) -> None:
    downloaded = 0
    async with aiofiles.open(tmp_path, "wb") as handle:
        async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_BYTES):
            downloaded += len(chunk)
            if max_bytes is not None and downloaded > max_bytes:
                raise HfDownloadTooLargeError(
                    f"Download exceeds MAX_MODEL_DOWNLOAD_MB limit ({max_bytes} bytes)"
                )
            await handle.write(chunk)
            if progress_cb is not None:
                progress_cb(downloaded, total)


class HfClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport

    def _build_client(self) -> httpx.AsyncClient:
        # Built fresh per call: HfClient calls are low-frequency (user-driven
        # search/install actions, not a hot path), so there is no benefit to
        # keeping a pooled connection alive, and this keeps the injected
        # MockTransport trivial to swap in tests.
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )

    def _auth_headers(self) -> dict[str, str]:
        if not self.settings.hf_token:
            return {}
        return {"Authorization": f"Bearer {self.settings.hf_token}"}

    async def search(
        self,
        query: str,
        limit: int = 20,
        task_tags: tuple[str, ...] | None = None,
        sort: str | None = None,
    ) -> list[HfModelSummary]:
        tags = SEARCH_TASK_TAGS if task_tags is None else task_tags
        params: dict[str, object] = {
            "filter": list(tags),
            "limit": limit,
            "full": "true",
        }
        # Query vacia es browse: omitir `search` permite ordenar todo el tag.
        if query:
            params["search"] = query
        if sort:
            params["sort"] = sort
            params["direction"] = -1
        async with self._build_client() as client:
            response = await client.get(
                f"{HF_API_BASE}/models", params=params, headers=self._auth_headers()
            )
            response.raise_for_status()
            payload = response.json()
        return [_parse_model_summary(item) for item in payload]

    async def repo_files(self, repo_id: str) -> list[HfFile]:
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                async with self._build_client() as client:
                    response = await client.get(
                        f"{HF_API_BASE}/models/{repo_id}",
                        params={"blobs": "true"},
                        headers=self._auth_headers(),
                    )
                    response.raise_for_status()
                    payload = response.json()
                return [_parse_sibling_file(sibling) for sibling in payload.get("siblings", [])]
            except Exception as exc:  # noqa: BLE001 -- CancelledError is BaseException, so cancel still propagates
                if attempt == DOWNLOAD_ATTEMPTS or not _is_retryable_download_error(exc):
                    raise _wrap_hf_auth_error(exc, repo_id) from exc
                logger.warning(
                    "Hugging Face repo_files attempt %d/%d failed (%s); retrying",
                    attempt,
                    DOWNLOAD_ATTEMPTS,
                    type(exc).__name__,
                )
                await asyncio.sleep(2 ** (attempt - 1))

    async def read_safetensors_header(
        self,
        repo_id: str,
        path: str,
    ) -> tuple[dict, int]:
        url = _download_url(repo_id, path)
        _validate_https_huggingface_host(url)

        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                async with self._build_client() as client:
                    length_response = await client.get(
                        url,
                        headers={**self._auth_headers(), "Range": "bytes=0-7"},
                    )
                    length_response.raise_for_status()
                    if len(length_response.content) != 8:
                        raise ValueError(
                            "El prefijo safetensors debe contener exactamente 8 bytes."
                        )
                    header_length = int.from_bytes(
                        length_response.content,
                        byteorder="little",
                        signed=False,
                    )
                    # El repo_id lo escribe el usuario, asi que estos 8 bytes son
                    # contenido no confiable: sin cota, un repo que declare un
                    # header de 2**40 haria pedir un Range de un terabyte y
                    # buffearlo entero en memoria. Los headers reales medidos van
                    # de 25 KB a 358 KB; 64 MB deja 100x de margen.
                    if not 0 < header_length <= MAX_SAFETENSORS_HEADER_BYTES:
                        raise ValueError(
                            f"El header safetensors declara {header_length} bytes, "
                            f"fuera del maximo de {MAX_SAFETENSORS_HEADER_BYTES}."
                        )

                    header_response = await client.get(
                        url,
                        headers={
                            **self._auth_headers(),
                            "Range": f"bytes=8-{8 + header_length - 1}",
                        },
                    )
                    header_response.raise_for_status()
                    if len(header_response.content) != header_length:
                        raise ValueError(
                            "El header safetensors recibido no coincide con el largo declarado."
                        )
                    header = json.loads(header_response.content)
                    if not isinstance(header, dict):
                        raise ValueError("El header safetensors no es un objeto JSON.")
                return header, header_length
            except Exception as exc:  # noqa: BLE001 -- CancelledError is BaseException
                if attempt == DOWNLOAD_ATTEMPTS or not _is_retryable_download_error(exc):
                    raise _wrap_hf_auth_error(exc, repo_id) from exc
                logger.warning(
                    "Hugging Face safetensors header attempt %d/%d failed (%s); retrying",
                    attempt,
                    DOWNLOAD_ATTEMPTS,
                    type(exc).__name__,
                )
                await asyncio.sleep(2 ** (attempt - 1))

    async def download(
        self,
        repo_id: str,
        filename: str,
        dest: Path,
        progress_cb: ProgressCallback | None = None,
        max_bytes: int | None = None,
        unlimited: bool = False,
    ) -> Path:
        url = _download_url(repo_id, filename)
        _validate_https_huggingface_host(url)
        effective_max: int | None
        if unlimited:
            effective_max = None
        elif max_bytes is None:
            effective_max = self.settings.max_model_download_mb * 1024 * 1024
        else:
            effective_max = max_bytes
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_name(f"{dest.name}{PARTIAL_DOWNLOAD_SUFFIX}")

        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                await self._stream_to_file(
                    url,
                    tmp_path,
                    effective_max,
                    progress_cb,
                )
                break
            except Exception as exc:  # noqa: BLE001 -- CancelledError is BaseException, so cancel still propagates
                tmp_path.unlink(missing_ok=True)
                if attempt == DOWNLOAD_ATTEMPTS or not _is_retryable_download_error(exc):
                    raise _wrap_hf_auth_error(exc, repo_id) from exc
                logger.warning(
                    "Hugging Face download attempt %d/%d failed (%s); retrying",
                    attempt,
                    DOWNLOAD_ATTEMPTS,
                    type(exc).__name__,
                )
                await asyncio.sleep(2 ** (attempt - 1))

        tmp_path.replace(dest)
        return dest

    async def _stream_to_file(
        self,
        url: str,
        tmp_path: Path,
        max_bytes: int | None,
        progress_cb: ProgressCallback | None,
    ) -> None:
        async with self._build_client() as client:
            async with client.stream("GET", url, headers=self._auth_headers()) as response:
                response.raise_for_status()
                total = _parse_content_length(response.headers)
                _reject_declared_size_over_limit(total, max_bytes)
                try:
                    await _write_response_to_file(response, tmp_path, max_bytes, total, progress_cb)
                except BaseException:
                    tmp_path.unlink(missing_ok=True)
                    raise
