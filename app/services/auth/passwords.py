from __future__ import annotations

import hmac
import secrets

from argon2.low_level import Type, hash_secret

ARGON2_TIME_COST = 3
ARGON2_MEMORY_COST = 65536  # 64 MiB
ARGON2_PARALLELISM = 4
ARGON2_HASH_LEN = 32
SALT_BYTES = 16


def generate_salt() -> str:
    return secrets.token_hex(SALT_BYTES)


def hash_password(password: str, salt: str) -> str:
    hashed = hash_secret(
        password.encode("utf-8"), bytes.fromhex(salt),
        time_cost=ARGON2_TIME_COST, memory_cost=ARGON2_MEMORY_COST,
        parallelism=ARGON2_PARALLELISM, hash_len=ARGON2_HASH_LEN, type=Type.ID,
    )
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, password_hash)
