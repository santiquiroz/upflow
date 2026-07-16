from __future__ import annotations

import asyncio
import time

import httpx
from packaging.version import Version

from app.config import Settings
from app.core.version import get_app_version
from app.models import UpdateStatus, utc_now

# ---------------------------------------------------------------------------
# Checks the GitHub Releases API for a version newer than the installed one.
# Reusable across projects: the repo and current version are injected (via
# UPDATE_REPO + the installed package version), so nothing here is
# Upflow-specific. A failed check (offline, timeout, 403 rate-limit, 5xx,
# bad payload) never raises and never breaks the app -- it returns a status
# with `error` set and `update_available=False`, keeping the last good
# `latest_version` if one was ever fetched. Results are cached in memory with
# a TTL so anonymous GitHub rate limits (60 req/h) are never a concern.
# ---------------------------------------------------------------------------

RELEASES_LATEST_URL = "https://api.github.com/repos/{repo}/releases/latest"
GITHUB_ACCEPT_HEADER = "application/vnd.github+json"
UPDATE_USER_AGENT = "upflow-update-check"
RELEASE_REQUEST_HEADERS = {"Accept": GITHUB_ACCEPT_HEADER, "User-Agent": UPDATE_USER_AGENT}


def _normalize_tag(tag: str) -> str:
    stripped = tag.strip()
    if stripped[:1] in ("v", "V"):
        return stripped[1:]
    return stripped


def _is_newer(latest: str, current: str) -> bool:
    return Version(latest) > Version(current)


class UpdateService:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport
        self._cache: UpdateStatus | None = None
        self._cached_at_monotonic: float | None = None
        self._last_good: UpdateStatus | None = None
        self._lock = asyncio.Lock()

    async def check(self, force: bool = False) -> UpdateStatus:
        if not force and self._is_cache_fresh():
            return self._cache  # type: ignore[return-value]
        async with self._lock:
            # Re-check under the lock so concurrent callers that queued behind
            # a running fetch reuse its fresh result instead of refetching.
            if not force and self._is_cache_fresh():
                return self._cache  # type: ignore[return-value]
            status = await self._perform_check()
            self._store(status)
            return status

    def _is_cache_fresh(self) -> bool:
        if self._cache is None or self._cached_at_monotonic is None:
            return False
        age = time.monotonic() - self._cached_at_monotonic
        return age < self.settings.update_check_ttl_seconds

    def _store(self, status: UpdateStatus) -> None:
        self._cache = status
        self._cached_at_monotonic = time.monotonic()
        if status.error is None and status.latest_version is not None:
            self._last_good = status

    async def _perform_check(self) -> UpdateStatus:
        current = get_app_version()
        if not self.settings.update_check_enabled:
            return self._disabled_status(current)
        try:
            release = await self._fetch_latest_release()
            return self._status_from_release(current, release)
        except Exception as exc:  # noqa: BLE001 - any failure must degrade, never raise
            return self._error_status(current, f"{type(exc).__name__}: {exc}")

    async def _fetch_latest_release(self) -> dict:
        url = RELEASES_LATEST_URL.format(repo=self.settings.update_repo)
        async with self._build_client() as client:
            response = await client.get(url, headers=RELEASE_REQUEST_HEADERS)
            response.raise_for_status()
            return response.json()

    def _build_client(self) -> httpx.AsyncClient:
        # Built fresh per call (like HfClient): the check is low-frequency and
        # cached, so a pooled connection buys nothing and this keeps the
        # injected MockTransport trivial to swap in tests.
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=self.settings.update_api_timeout_seconds,
            follow_redirects=True,
        )

    def _status_from_release(self, current: str, release: dict) -> UpdateStatus:
        tag = release.get("tag_name")
        if not tag:
            raise ValueError("release payload has no tag_name")
        latest = _normalize_tag(tag)
        return UpdateStatus(
            current_version=current,
            latest_version=latest,
            update_available=_is_newer(latest, current),
            release_url=release.get("html_url"),
            published_at=release.get("published_at"),
            checked_at=utc_now(),
            error=None,
        )

    def _disabled_status(self, current: str) -> UpdateStatus:
        return UpdateStatus(
            current_version=current,
            latest_version=None,
            update_available=False,
            release_url=None,
            published_at=None,
            checked_at=utc_now(),
            error=None,
        )

    def _error_status(self, current: str, message: str) -> UpdateStatus:
        prior = self._last_good
        return UpdateStatus(
            current_version=current,
            latest_version=prior.latest_version if prior else None,
            update_available=False,
            release_url=prior.release_url if prior else None,
            published_at=prior.published_at if prior else None,
            checked_at=utc_now(),
            error=message,
        )
