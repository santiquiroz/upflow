from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from app.models import utc_now


def write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp in the same directory (not the OS temp dir) so Path.replace is
    # an atomic rename on the same filesystem, never a cross-device copy.
    descriptor, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with open(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_json_atomically(path: Path, payload: Any) -> None:
    write_text_atomically(path, json.dumps(payload, indent=2))


def backup_corrupt_file(path: Path, exc: Exception, logger: logging.Logger) -> Path:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%S%f")
    backup_path = path.with_name(f"{path.name}.corrupt-{timestamp}")
    path.replace(backup_path)
    logger.warning("Corrupt JSON file at %s (%s); backed up to %s", path, exc, backup_path)
    return backup_path
