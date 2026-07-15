from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.exceptions import HfDownloadTooLargeError, HfInvalidSourceError
from app.services.hf_client import (
    HfClient,
    HfFile,
    HfModelSummary,
    _validate_https_huggingface_host,
    pick_weight_file,
)

# ---------------------------------------------------------------------------
# SP1 Task 3 - hf_client: Hugging Face REST client (search / repo_files /
# download) + pick_weight_file priority helper.
#
# All HTTP calls go through an injected httpx.MockTransport -- no real
# network in unit tests. Response shapes below were verified against the
# real API on 2026-07-14:
#   - search: GET https://huggingface.co/api/models
#       ?search=<q>&filter=image-to-image&filter=super-resolution&limit=<n>&full=true
#     Repeated `filter` params are ANDed by the Hub (confirmed live).
#   - repo_files: GET https://huggingface.co/api/models/<repo_id>?blobs=true
#     `siblings[].size` is only present with blobs=true (absent otherwise).
#   - download: GET https://huggingface.co/<repo_id>/resolve/main/<filename>
#     Real LFS-backed weights 302/307-redirect to a different host
#     (cas-bridge.xethub.hf.co) with a pre-signed URL -- httpx's
#     follow_redirects=True must chase that, so host validation only guards
#     the *initial* request we build ourselves, not the redirect target.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def transport_for(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

SEARCH_RESPONSE_FIXTURE = [
    {
        "id": "Kim2091/2x-AnimeSharpV4",
        "author": "Kim2091",
        "pipeline_tag": "image-to-image",
        "downloads": 0,
        "likes": 45,
        "tags": ["onnx", "pytorch", "super-resolution", "image-to-image"],
    },
    {
        "id": "SnJake/Baikal-Swin-Anime",
        "author": "SnJake",
        "pipeline_tag": "image-to-image",
        "downloads": 19,
        "likes": 1,
        "tags": ["pytorch", "super-resolution", "image-to-image"],
    },
]


async def test_search_sends_query_filter_and_limit_params(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["filters"] = request.url.params.get_list("filter")
        return httpx.Response(200, json=SEARCH_RESPONSE_FIXTURE)

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))

    await client.search("anime", limit=5)

    assert captured["url"].startswith("https://huggingface.co/api/models?")
    assert "search=anime" in captured["url"]
    assert "limit=5" in captured["url"]
    assert "full=true" in captured["url"]
    assert captured["filters"] == ["image-to-image", "super-resolution"]


async def test_search_parses_model_summaries(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SEARCH_RESPONSE_FIXTURE)

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))

    results = await client.search("anime")

    assert results == [
        HfModelSummary(
            id="Kim2091/2x-AnimeSharpV4",
            author="Kim2091",
            pipeline_tag="image-to-image",
            downloads=0,
            likes=45,
            tags=("onnx", "pytorch", "super-resolution", "image-to-image"),
        ),
        HfModelSummary(
            id="SnJake/Baikal-Swin-Anime",
            author="SnJake",
            pipeline_tag="image-to-image",
            downloads=19,
            likes=1,
            tags=("pytorch", "super-resolution", "image-to-image"),
        ),
    ]


async def test_search_defaults_limit_to_20(tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["limit"] = request.url.params.get("limit")
        return httpx.Response(200, json=[])

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))

    await client.search("anime")

    assert captured["limit"] == "20"


async def test_search_tolerates_missing_optional_fields(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "sparse/model"}])

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))

    results = await client.search("sparse")

    assert results == [
        HfModelSummary(
            id="sparse/model", author=None, pipeline_tag=None, downloads=0, likes=0, tags=()
        )
    ]


async def test_search_raises_on_http_error_status(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))

    with pytest.raises(httpx.HTTPStatusError):
        await client.search("anime")


async def test_search_sends_bearer_token_when_configured(tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, json=[])

    settings = make_settings(tmp_path, HF_TOKEN="hf_secrettoken")
    client = HfClient(settings, transport=transport_for(handler))

    await client.search("anime")

    assert captured["auth"] == "Bearer hf_secrettoken"


async def test_search_omits_auth_header_when_no_token(tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, json=[])

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))

    await client.search("anime")

    assert captured["auth"] == ""


# ---------------------------------------------------------------------------
# repo_files()
# ---------------------------------------------------------------------------

REPO_INFO_FIXTURE = {
    "id": "Kim2091/ClearRealityV1",
    "modelId": "Kim2091/ClearRealityV1",
    "siblings": [
        {"rfilename": ".gitattributes", "size": 1519},
        {"rfilename": "4x-ClearRealityV1.pth", "size": 9016074},
        {"rfilename": "4x-ClearRealityV1.safetensors", "size": 4492232},
        {"rfilename": "ONNX/4x-ClearRealityV1-opset17.onnx", "size": 866325},
    ],
}


async def test_repo_files_requests_blobs_true(tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["blobs"] = request.url.params.get("blobs")
        return httpx.Response(200, json=REPO_INFO_FIXTURE)

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))

    await client.repo_files("Kim2091/ClearRealityV1")

    assert captured["path"] == "/api/models/Kim2091/ClearRealityV1"
    assert captured["blobs"] == "true"


async def test_repo_files_parses_siblings_into_hf_files(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=REPO_INFO_FIXTURE)

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))

    files = await client.repo_files("Kim2091/ClearRealityV1")

    assert files == [
        HfFile(path=".gitattributes", size=1519),
        HfFile(path="4x-ClearRealityV1.pth", size=9016074),
        HfFile(path="4x-ClearRealityV1.safetensors", size=4492232),
        HfFile(path="ONNX/4x-ClearRealityV1-opset17.onnx", size=866325),
    ]


async def test_repo_files_returns_empty_list_when_no_siblings(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "empty/repo"})

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))

    files = await client.repo_files("empty/repo")

    assert files == []


async def test_repo_files_raises_on_404(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Repository not found"})

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))

    with pytest.raises(httpx.HTTPStatusError):
        await client.repo_files("does-not/exist")


# ---------------------------------------------------------------------------
# pick_weight_file()
# ---------------------------------------------------------------------------


def test_pick_weight_file_prefers_onnx_over_safetensors_and_pth() -> None:
    files = [
        HfFile(path="model.pth", size=9_000_000),
        HfFile(path="model.safetensors", size=4_000_000),
        HfFile(path="model.onnx", size=800_000),
    ]

    assert pick_weight_file(files) == HfFile(path="model.onnx", size=800_000)


def test_pick_weight_file_prefers_safetensors_over_pth_when_no_onnx() -> None:
    files = [
        HfFile(path="model.pth", size=9_000_000),
        HfFile(path="model.safetensors", size=4_000_000),
    ]

    assert pick_weight_file(files) == HfFile(path="model.safetensors", size=4_000_000)


def test_pick_weight_file_falls_back_to_pth_when_no_better_option() -> None:
    files = [
        HfFile(path="README.md", size=100),
        HfFile(path="model.pth", size=9_000_000),
    ]

    assert pick_weight_file(files) == HfFile(path="model.pth", size=9_000_000)


def test_pick_weight_file_picks_largest_among_same_priority_extension() -> None:
    files = [
        HfFile(path="fp16/model-opset14.onnx", size=866_325),
        HfFile(path="fp32/model-opset14.onnx", size=1_718_947),
        HfFile(path="fp16/model-opset17.onnx", size=866_325),
    ]

    assert pick_weight_file(files) == HfFile(path="fp32/model-opset14.onnx", size=1_718_947)


def test_pick_weight_file_raises_when_no_weight_file_present() -> None:
    files = [HfFile(path="README.md", size=100), HfFile(path=".gitattributes", size=50)]

    with pytest.raises(ValueError, match="weight file"):
        pick_weight_file(files)


def test_pick_weight_file_raises_on_empty_list() -> None:
    with pytest.raises(ValueError, match="weight file"):
        pick_weight_file([])


# ---------------------------------------------------------------------------
# download()
# ---------------------------------------------------------------------------


async def test_download_requests_resolve_main_url(tmp_path: Path) -> None:
    captured: dict[str, str] = {}
    payload = b"onnx-bytes-payload"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, content=payload)

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))
    dest = tmp_path / "downloads" / "model.onnx"

    await client.download("Kim2091/ClearRealityV1", "model.onnx", dest)

    assert captured["url"] == "https://huggingface.co/Kim2091/ClearRealityV1/resolve/main/model.onnx"


async def test_download_streams_body_to_dest_file(tmp_path: Path) -> None:
    payload = b"0123456789" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))
    dest = tmp_path / "downloads" / "model.onnx"

    result = await client.download("org/repo", "model.onnx", dest)

    assert result == dest
    assert dest.read_bytes() == payload


async def test_download_creates_parent_directories(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data")

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))
    dest = tmp_path / "nested" / "deep" / "model.onnx"

    await client.download("org/repo", "model.onnx", dest)

    assert dest.exists()


async def test_download_leaves_no_leftover_tmp_file_on_success(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data")

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))
    dest = tmp_path / "downloads" / "model.onnx"

    await client.download("org/repo", "model.onnx", dest)

    leftovers = list(dest.parent.glob("*.part"))
    assert leftovers == []


async def test_download_invokes_progress_callback_with_bytes_and_total(tmp_path: Path) -> None:
    payload = b"x" * (5 * 1024 * 1024)  # 5 MiB, spans several chunks

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=payload, headers={"content-length": str(len(payload))}
        )

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))
    dest = tmp_path / "downloads" / "model.onnx"
    calls: list[tuple[int, int | None]] = []

    def record_progress(downloaded: int, total: int | None) -> None:
        calls.append((downloaded, total))

    await client.download("org/repo", "model.onnx", dest, progress_cb=record_progress)

    assert calls
    assert calls[-1][0] == len(payload)
    assert all(total == len(payload) for _, total in calls)


async def test_download_sends_bearer_token_when_configured(tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, content=b"data")

    settings = make_settings(tmp_path, HF_TOKEN="hf_secrettoken")
    client = HfClient(settings, transport=transport_for(handler))
    dest = tmp_path / "downloads" / "model.onnx"

    await client.download("org/repo", "model.onnx", dest)

    assert captured["auth"] == "Bearer hf_secrettoken"


async def test_download_raises_on_http_error_status(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = HfClient(make_settings(tmp_path), transport=transport_for(handler))
    dest = tmp_path / "downloads" / "model.onnx"

    with pytest.raises(httpx.HTTPStatusError):
        await client.download("org/repo", "missing.onnx", dest)

    assert not dest.exists()


async def test_download_rejects_declared_size_over_limit_before_writing(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 10, headers={"content-length": "99999999999"})

    settings = make_settings(tmp_path, MAX_MODEL_DOWNLOAD_MB=1)
    client = HfClient(settings, transport=transport_for(handler))
    dest = tmp_path / "downloads" / "model.onnx"

    with pytest.raises(HfDownloadTooLargeError):
        await client.download("org/repo", "model.onnx", dest)

    assert not dest.exists()
    assert list(dest.parent.glob("*.part")) == []


async def test_download_aborts_and_cleans_up_when_stream_exceeds_limit(tmp_path: Path) -> None:
    # No content-length header this time: the streaming guard (not the
    # declared-size pre-check) must be the one that catches the overflow.
    oversized_payload = b"x" * (2 * 1024 * 1024)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized_payload)

    settings = make_settings(tmp_path, MAX_MODEL_DOWNLOAD_MB=1)
    client = HfClient(settings, transport=transport_for(handler))
    dest = tmp_path / "downloads" / "model.onnx"

    with pytest.raises(HfDownloadTooLargeError):
        await client.download("org/repo", "model.onnx", dest)

    assert not dest.exists()
    assert list(dest.parent.glob("*.part")) == []


def test_download_rejects_non_https_source() -> None:
    with pytest.raises(HfInvalidSourceError):
        _validate_https_huggingface_host("http://huggingface.co/org/repo/resolve/main/model.onnx")


def test_download_rejects_non_huggingface_host() -> None:
    with pytest.raises(HfInvalidSourceError):
        _validate_https_huggingface_host("https://evil.example.com/org/repo/resolve/main/model.onnx")


def test_download_accepts_https_huggingface_host() -> None:
    _validate_https_huggingface_host("https://huggingface.co/org/repo/resolve/main/model.onnx")
