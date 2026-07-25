from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

SESSION_COOKIE_NAME = "upflow_session"
SESSION_TTL_SECONDS = 30 * 24 * 3600


@dataclass(frozen=True, slots=True)
class SessionPayload:
    user_id: str
    session_ver: int
    expires_at: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload_b64: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()


def create_session_cookie_value(
    user_id: str, session_ver: int, secret: str, *, now: datetime | None = None
) -> str:
    expires_at = (now or _utc_now()) + timedelta(seconds=SESSION_TTL_SECONDS)
    payload = {"uid": user_id, "sv": session_ver, "exp": expires_at.isoformat()}
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = _sign(payload_b64, secret)
    return f"{payload_b64}.{signature}"


def verify_session_cookie(
    value: str, secret: str, *, now: datetime | None = None
) -> SessionPayload | None:
    try:
        payload_b64, signature = value.rsplit(".", 1)
    except ValueError:
        return None
    expected_signature = _sign(payload_b64, secret)
    if not hmac.compare_digest(signature, expected_signature):
        return None
    try:
        payload = json.loads(_b64decode(payload_b64))
        session_payload = SessionPayload(
            user_id=payload["uid"],
            session_ver=int(payload["sv"]),
            expires_at=datetime.fromisoformat(payload["exp"]),
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if session_payload.expires_at <= (now or _utc_now()):
        return None
    return session_payload
