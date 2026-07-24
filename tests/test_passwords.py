from __future__ import annotations

from app.services.auth.passwords import generate_salt, hash_password, verify_password


def test_generate_salt_returns_distinct_values() -> None:
    assert generate_salt() != generate_salt()


def test_hash_password_is_deterministic_for_same_salt() -> None:
    salt = generate_salt()
    assert hash_password("hunter2", salt) == hash_password("hunter2", salt)


def test_hash_password_differs_across_salts() -> None:
    assert hash_password("hunter2", generate_salt()) != hash_password("hunter2", generate_salt())


def test_verify_password_accepts_correct_password() -> None:
    salt = generate_salt()
    password_hash = hash_password("correct horse", salt)
    assert verify_password("correct horse", password_hash, salt) is True


def test_verify_password_rejects_wrong_password() -> None:
    salt = generate_salt()
    password_hash = hash_password("correct horse", salt)
    assert verify_password("wrong password", password_hash, salt) is False
