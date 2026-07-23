from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import aiofiles
import httpx

from app.config import Settings
from app.exceptions import HfDownloadTooLargeError, HfInvalidSourceError

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
WEIGHT_EXTENSION_PRIORITY = (".onnx", ".safetensors", ".pth")
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30.0
PARTIAL_DOWNLOAD_SUFFIX = ".part"

ProgressCallback = Callable[[int, "int | None"], None]


@dataclass(slots=True, frozen=True)
class HfModelSummary:
    id: str
    author: str | None
    pipeline_tag: str | None
    downloads: int
    likes: int
    tags: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class HfFile:
    path: str
    size: int


def _parse_model_summary(item: dict) -> HfModelSummary:
    return HfModelSummary(
        id=item["id"],
        author=item.get("author"),
        pipeline_tag=item.get("pipeline_tag"),
        downloads=item.get("downloads", 0),
        likes=item.get("likes", 0),
        tags=tuple(item.get("tags", [])),
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


def _reject_declared_size_over_limit(total: int | None, max_bytes: int) -> None:
    if total is not None and total > max_bytes:
        raise HfDownloadTooLargeError(
            f"Declared size {total} bytes exceeds MAX_MODEL_DOWNLOAD_MB limit ({max_bytes} bytes)"
        )


async def _write_response_to_file(
    response: httpx.Response,
    tmp_path: Path,
    max_bytes: int,
    total: int | None,
    progress_cb: ProgressCallback | None,
) -> None:
    downloaded = 0
    async with aiofiles.open(tmp_path, "wb") as handle:
        async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_BYTES):
            downloaded += len(chunk)
            if downloaded > max_bytes:
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

    async def search(self, query: str, limit: int = 20) -> list[HfModelSummary]:
        params = {
            "search": query,
            "filter": list(SEARCH_TASK_TAGS),
            "limit": limit,
            "full": "true",
        }
        async with self._build_client() as client:
            response = await client.get(
                f"{HF_API_BASE}/models", params=params, headers=self._auth_headers()
            )
            response.raise_for_status()
            payload = response.json()
        return [_parse_model_summary(item) for item in payload]

    async def repo_files(self, repo_id: str) -> list[HfFile]:
        async with self._build_client() as client:
            response = await client.get(
                f"{HF_API_BASE}/models/{repo_id}",
                params={"blobs": "true"},
                headers=self._auth_headers(),
            )
            response.raise_for_status()
            payload = response.json()
        return [_parse_sibling_file(sibling) for sibling in payload.get("siblings", [])]

    async def download(
        self,
        repo_id: str,
        filename: str,
        dest: Path,
        progress_cb: ProgressCallback | None = None,
        max_bytes: int | None = None,
    ) -> Path:
        url = _download_url(repo_id, filename)
        _validate_https_huggingface_host(url)
        if max_bytes is None:
            max_bytes = self.settings.max_model_download_mb * 1024 * 1024
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_name(f"{dest.name}{PARTIAL_DOWNLOAD_SUFFIX}")

        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                await self._stream_to_file(url, tmp_path, max_bytes, progress_cb)
                break
            except Exception as exc:  # noqa: BLE001 -- CancelledError is BaseException, so cancel still propagates
                tmp_path.unlink(missing_ok=True)
                if attempt == DOWNLOAD_ATTEMPTS or not _is_retryable_download_error(exc):
                    raise
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
        max_bytes: int,
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
