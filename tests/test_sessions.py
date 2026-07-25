from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.services.auth.sessions import (
    SESSION_TTL_SECONDS,
    create_session_cookie_value,
    verify_session_cookie,
)

SECRET = "test-secret-32-bytes-of-entropy!"


def test_verify_accepts_freshly_created_cookie() -> None:
    value = create_session_cookie_value("user-1", 0, SECRET)

    payload = verify_session_cookie(value, SECRET)

    assert payload is not None
    assert payload.user_id == "user-1"
    assert payload.session_ver == 0


def test_verify_rejects_tampered_payload() -> None:
    value = create_session_cookie_value("user-1", 0, SECRET)
    payload_b64, signature = value.rsplit(".", 1)
    tampered = f"{payload_b64}x.{signature}"

    assert verify_session_cookie(tampered, SECRET) is None


def test_verify_rejects_tampered_signature() -> None:
    value = create_session_cookie_value("user-1", 0, SECRET)
    payload_b64, signature = value.rsplit(".", 1)
    tampered_signature = ("0" if signature[0] != "0" else "1") + signature[1:]

    assert verify_session_cookie(f"{payload_b64}.{tampered_signature}", SECRET) is None


def test_verify_rejects_wrong_secret() -> None:
    value = create_session_cookie_value("user-1", 0, SECRET)

    assert verify_session_cookie(value, "a-completely-different-secret") is None


def test_verify_rejects_expired_cookie() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value = create_session_cookie_value("user-1", 0, SECRET, now=created_at)
    past_expiry = created_at + timedelta(seconds=SESSION_TTL_SECONDS + 1)

    assert verify_session_cookie(value, SECRET, now=past_expiry) is None


def test_verify_accepts_cookie_just_before_expiry() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value = create_session_cookie_value("user-1", 0, SECRET, now=created_at)
    just_before_expiry = created_at + timedelta(seconds=SESSION_TTL_SECONDS - 1)

    assert verify_session_cookie(value, SECRET, now=just_before_expiry) is not None


def test_verify_rejects_malformed_value() -> None:
    assert verify_session_cookie("not-a-valid-token", SECRET) is None
    assert verify_session_cookie("", SECRET) is None


def test_session_ver_round_trips() -> None:
    value = create_session_cookie_value("user-1", 7, SECRET)

    payload = verify_session_cookie(value, SECRET)

    assert payload is not None
    assert payload.session_ver == 7
