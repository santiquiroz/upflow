from __future__ import annotations

from app.config import Settings

# ---------------------------------------------------------------------------
# ALLOWED_ORIGINS default must be derived from the actual APP_HOST/APP_PORT
# instead of a fixed hardcoded value, so a custom APP_PORT does not silently
# leave the CORS/origin guard pointed at the wrong port.
# ---------------------------------------------------------------------------


def test_allowed_origins_default_uses_default_app_port() -> None:
    settings = Settings()

    assert settings.allowed_origin_values == frozenset(
        {"http://127.0.0.1:8090", "http://localhost:8090"}
    )


def test_allowed_origins_default_derives_from_custom_app_port() -> None:
    settings = Settings(APP_PORT=9999)

    assert settings.allowed_origin_values == frozenset(
        {"http://127.0.0.1:9999", "http://localhost:9999"}
    )


def test_allowed_origins_default_includes_custom_app_host() -> None:
    settings = Settings(APP_HOST="192.168.1.50", APP_PORT=8090)

    assert settings.allowed_origin_values == frozenset(
        {
            "http://127.0.0.1:8090",
            "http://localhost:8090",
            "http://192.168.1.50:8090",
        }
    )


def test_allowed_origins_default_does_not_duplicate_neutral_bind_hosts() -> None:
    settings_loopback = Settings(APP_HOST="127.0.0.1", APP_PORT=8090)
    settings_any = Settings(APP_HOST="0.0.0.0", APP_PORT=8090)

    expected = frozenset({"http://127.0.0.1:8090", "http://localhost:8090"})
    assert settings_loopback.allowed_origin_values == expected
    assert settings_any.allowed_origin_values == expected


def test_allowed_origins_env_override_is_used_verbatim() -> None:
    settings = Settings(APP_PORT=9999, ALLOWED_ORIGINS="https://custom.example.com")

    assert settings.allowed_origin_values == frozenset({"https://custom.example.com"})
