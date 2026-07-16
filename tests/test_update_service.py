from __future__ import annotations

import importlib.metadata

import httpx
import pytest

import app.core.version as version_module
import app.services.update_service as update_service_module
from app.config import Settings
from app.services.update_service import UpdateService

# ---------------------------------------------------------------------------
# SP8 Task 1 - UpdateService: checks the GitHub Releases API for a newer
# version and never lets a network failure break the app. All HTTP goes
# through an injected httpx.MockTransport (same idiom as tests/test_hf_client.py)
# so no real network is touched. get_app_version() is pinned per test so the
# comparison is independent of the version actually installed in the env.
# ---------------------------------------------------------------------------

PINNED_CURRENT = "0.1.0"


@pytest.fixture(autouse=True)
def _pin_current_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only rebinds the reference the SERVICE uses; the real
    # version_module.get_app_version stays intact for the version tests below.
    monkeypatch.setattr(update_service_module, "get_app_version", lambda *_a, **_k: PINNED_CURRENT)


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def release_payload(
    tag: str = "v0.2.0",
    url: str = "https://github.com/santiquiroz/upflow/releases/tag/v0.2.0",
    published: str = "2026-07-16T00:00:00Z",
) -> dict:
    return {"tag_name": tag, "html_url": url, "published_at": published}


class CountingHandler:
    """Wraps a response factory and records how many requests it served."""

    def __init__(self, factory) -> None:  # type: ignore[no-untyped-def]
        self.calls = 0
        self._factory = factory

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self._factory(request)


def make_service(handler, **overrides: object) -> UpdateService:  # type: ignore[no-untyped-def]
    return UpdateService(make_settings(**overrides), transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# version comparison
# ---------------------------------------------------------------------------


async def test_newer_release_marks_update_available(tmp_path) -> None:  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=release_payload(tag="v0.2.0"))

    status = await make_service(handler).check()

    assert status.update_available is True
    assert status.current_version == "0.1.0"
    assert status.latest_version == "0.2.0"
    assert status.release_url == "https://github.com/santiquiroz/upflow/releases/tag/v0.2.0"
    assert status.published_at == "2026-07-16T00:00:00Z"
    assert status.error is None


async def test_equal_version_is_not_an_update() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=release_payload(tag="v0.1.0"))

    status = await make_service(handler).check()

    assert status.update_available is False
    assert status.latest_version == "0.1.0"
    assert status.error is None


async def test_older_release_is_not_an_update() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=release_payload(tag="v0.0.9"))

    status = await make_service(handler).check()

    assert status.update_available is False
    assert status.latest_version == "0.0.9"


async def test_prerelease_newer_than_current_is_an_update() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=release_payload(tag="v0.2.0-rc1"))

    status = await make_service(handler).check()

    assert status.update_available is True
    assert status.latest_version == "0.2.0-rc1"


async def test_prerelease_of_current_version_is_not_an_update(monkeypatch: pytest.MonkeyPatch) -> None:
    # current 0.2.0 stable is newer than its own release candidate.
    monkeypatch.setattr(update_service_module, "get_app_version", lambda *_a, **_k: "0.2.0")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=release_payload(tag="v0.2.0-rc2"))

    status = await make_service(handler).check()

    assert status.update_available is False


async def test_tag_with_leading_v_is_stripped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=release_payload(tag="v0.2.0"))

    status = await make_service(handler).check()

    assert status.latest_version == "0.2.0"


async def test_tag_without_leading_v_is_accepted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=release_payload(tag="0.2.0"))

    status = await make_service(handler).check()

    assert status.latest_version == "0.2.0"
    assert status.update_available is True


async def test_sends_accept_and_user_agent_headers_to_releases_api() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["accept"] = request.headers.get("accept", "")
        captured["ua"] = request.headers.get("user-agent", "")
        return httpx.Response(200, json=release_payload())

    await make_service(handler, UPDATE_REPO="santiquiroz/upflow").check()

    assert captured["url"] == "https://api.github.com/repos/santiquiroz/upflow/releases/latest"
    assert captured["accept"] == "application/vnd.github+json"
    assert captured["ua"] != ""


# ---------------------------------------------------------------------------
# failures never raise
# ---------------------------------------------------------------------------


async def test_timeout_is_caught_and_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    status = await make_service(handler).check()

    assert status.update_available is False
    assert status.error is not None
    assert status.latest_version is None


async def test_server_error_is_caught_and_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    status = await make_service(handler).check()

    assert status.update_available is False
    assert status.error is not None


async def test_rate_limit_403_is_caught_and_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "API rate limit exceeded"})

    status = await make_service(handler).check()

    assert status.update_available is False
    assert status.error is not None


async def test_invalid_json_is_caught_and_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json{")

    status = await make_service(handler).check()

    assert status.update_available is False
    assert status.error is not None


async def test_empty_releases_404_is_caught_and_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    status = await make_service(handler).check()

    assert status.update_available is False
    assert status.error is not None


async def test_missing_tag_name_is_caught_and_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"html_url": "https://x", "published_at": "2026"})

    status = await make_service(handler).check()

    assert status.update_available is False
    assert status.error is not None


async def test_unparseable_tag_is_caught_and_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=release_payload(tag="not-a-version"))

    status = await make_service(handler).check()

    assert status.update_available is False
    assert status.error is not None


# ---------------------------------------------------------------------------
# cache / TTL / force
# ---------------------------------------------------------------------------


async def test_second_check_uses_cache_without_refetching() -> None:
    handler = CountingHandler(lambda request: httpx.Response(200, json=release_payload()))
    service = UpdateService(make_settings(), transport=httpx.MockTransport(handler))

    first = await service.check()
    second = await service.check()

    assert handler.calls == 1
    assert first.latest_version == second.latest_version == "0.2.0"


async def test_force_bypasses_cache_and_refetches() -> None:
    handler = CountingHandler(lambda request: httpx.Response(200, json=release_payload()))
    service = UpdateService(make_settings(), transport=httpx.MockTransport(handler))

    await service.check()
    await service.check(force=True)

    assert handler.calls == 2


async def test_expired_ttl_triggers_refetch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Drive a controllable monotonic clock so the cache genuinely ages past the
    # TTL (a TTL of 0 is rejected by config validation, so time must advance).
    clock = {"t": 0.0}
    monkeypatch.setattr(update_service_module.time, "monotonic", lambda: clock["t"])
    handler = CountingHandler(lambda request: httpx.Response(200, json=release_payload()))
    service = UpdateService(
        make_settings(UPDATE_CHECK_TTL_SECONDS=1), transport=httpx.MockTransport(handler)
    )

    await service.check()
    clock["t"] = 100.0
    await service.check()

    assert handler.calls == 2


async def test_disabled_check_never_fetches() -> None:
    handler = CountingHandler(lambda request: httpx.Response(200, json=release_payload()))
    service = UpdateService(
        make_settings(UPDATE_CHECK_ENABLED=False), transport=httpx.MockTransport(handler)
    )

    status = await service.check()
    await service.check()

    assert handler.calls == 0
    assert status.update_available is False
    assert status.current_version == "0.1.0"
    assert status.error is None


async def test_failed_refresh_keeps_serving_last_good() -> None:
    # A transient failure must NOT hide a genuinely-available update: once a
    # good result exists, a later failed refresh keeps serving it (error None,
    # update_available True) so the banner stays visible instead of vanishing.
    responses = iter(
        [
            httpx.Response(200, json=release_payload(tag="v0.2.0")),
            httpx.Response(500, text="down"),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    service = UpdateService(make_settings(), transport=httpx.MockTransport(handler))

    good = await service.check()
    after_failure = await service.check(force=True)

    assert good.latest_version == "0.2.0"
    assert good.update_available is True
    assert after_failure.error is None
    assert after_failure.latest_version == "0.2.0"
    assert after_failure.update_available is True


async def test_first_ever_failure_returns_honest_error_status() -> None:
    # With no prior good result, a failure surfaces an honest error status
    # (all version fields null) instead of a misleading update flag.
    service = make_service(lambda request: httpx.Response(503, text="down"))

    status = await service.check()

    assert status.error is not None
    assert status.update_available is False
    assert status.latest_version is None


async def test_user_agent_carries_configured_package_name() -> None:
    # Reusability: the UA (and version lookup) derive from UPDATE_PACKAGE_NAME,
    # so pointing the checker at another project needs no code edit.
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers.get("user-agent", "")
        return httpx.Response(200, json=release_payload())

    service = UpdateService(
        make_settings(UPDATE_PACKAGE_NAME="other-project"),
        transport=httpx.MockTransport(handler),
    )
    await service.check()

    assert seen["ua"] == "other-project-update-check"


# ---------------------------------------------------------------------------
# get_app_version()
# ---------------------------------------------------------------------------


def test_get_app_version_returns_installed_version() -> None:
    assert version_module.get_app_version() == importlib.metadata.version("upflow")


def test_get_app_version_falls_back_to_pyproject(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    def raise_not_found(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(version_module.importlib.metadata, "version", raise_not_found)
    fake_pyproject = tmp_path / "pyproject.toml"
    fake_pyproject.write_text('[project]\nname = "upflow"\nversion = "9.9.9"\n', encoding="utf-8")
    monkeypatch.setattr(version_module, "PYPROJECT_PATH", fake_pyproject)

    assert version_module.get_app_version() == "9.9.9"


def test_get_app_version_final_fallback_when_nothing_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    def raise_not_found(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(version_module.importlib.metadata, "version", raise_not_found)
    monkeypatch.setattr(version_module, "PYPROJECT_PATH", tmp_path / "does-not-exist.toml")

    assert version_module.get_app_version() == "0.0.0"


# ---------------------------------------------------------------------------
# config validation for the update settings
# ---------------------------------------------------------------------------


def test_ttl_below_one_is_rejected() -> None:
    with pytest.raises(Exception):
        make_settings(UPDATE_CHECK_TTL_SECONDS=0)


def test_error_retry_below_one_is_rejected() -> None:
    with pytest.raises(Exception):
        make_settings(UPDATE_ERROR_RETRY_SECONDS=0)


def test_api_timeout_not_positive_is_rejected() -> None:
    with pytest.raises(Exception):
        make_settings(UPDATE_API_TIMEOUT_SECONDS=0)
