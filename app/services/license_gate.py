from __future__ import annotations

from collections.abc import Set
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT
from app.services.restore_models import LICENSE_GATED_PACKS

# Dentro de app/ para que package-release.ps1 lo lleve al instalador sin una regla aparte:
# el texto se muestra ANTES de bajar el pack, cuando su LICENSE todavia no esta en disco.
GATED_LICENSES_DIR = PROJECT_ROOT / "app" / "licenses" / "gated"
LICENSE_TEXT_SUFFIX = ".txt"

LICENSE_REQUIRED_KEY = "pack.license.required"
LICENSE_UNAVAILABLE_KEY = "pack.license.unavailable"


class LicenseNotAcceptedError(PermissionError):
    def __init__(self, pack: str, key: str, reason: str) -> None:
        super().__init__(reason)
        self.pack = pack
        self.key = key
        self.reason = reason


def is_gated(pack: str, gated: Set[str] = LICENSE_GATED_PACKS) -> bool:
    return pack in gated


def gated_license_path(pack: str, directory: Path = GATED_LICENSES_DIR) -> Path:
    return directory / f"{pack}{LICENSE_TEXT_SUFFIX}"


def gated_license_text(pack: str, directory: Path = GATED_LICENSES_DIR) -> str | None:
    path = gated_license_path(pack, directory)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return text if text.strip() else None


def license_texts_on_disk(directory: Path = GATED_LICENSES_DIR) -> frozenset[str]:
    if not directory.is_dir():
        return frozenset()
    return frozenset(path.stem for path in directory.glob(f"*{LICENSE_TEXT_SUFFIX}"))


def license_gate(
    pack: str, gated: Set[str] = LICENSE_GATED_PACKS, directory: Path = GATED_LICENSES_DIR
) -> dict[str, Any]:
    if not is_gated(pack, gated):
        return {"pack": pack, "gated": False, "licenseText": None}
    return {"pack": pack, "gated": True, "licenseText": gated_license_text(pack, directory)}


def require_accepted_license(
    pack: str,
    accepted: bool,
    gated: Set[str] = LICENSE_GATED_PACKS,
    directory: Path = GATED_LICENSES_DIR,
) -> None:
    if not is_gated(pack, gated):
        return
    # Fail-closed: aceptar una licencia que nadie pudo leer no es un consentimiento.
    if gated_license_text(pack, directory) is None:
        raise LicenseNotAcceptedError(
            pack,
            LICENSE_UNAVAILABLE_KEY,
            f"The license text of the pack {pack!r} is missing, so it cannot be accepted or downloaded.",
        )
    if not accepted:
        raise LicenseNotAcceptedError(
            pack,
            LICENSE_REQUIRED_KEY,
            f"The pack {pack!r} has a restrictive license: read it and accept it before downloading.",
        )
