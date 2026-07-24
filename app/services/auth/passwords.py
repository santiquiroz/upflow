from __future__ import annotations

import hashlib
import hmac
import secrets

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
KEY_LENGTH = 32


def generate_salt() -> str:
    return secrets.token_hex(SALT_BYTES)


def hash_password(password: str, salt: str) -> str:
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=bytes.fromhex(salt),
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_LENGTH,
    )
    return derived.hex()


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, password_hash)
