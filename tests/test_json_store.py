from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.json_store import backup_corrupt_file, write_json_atomically, write_text_atomically


def test_write_text_atomically_creates_file_with_content(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.txt"

    write_text_atomically(target, "hello")

    assert target.read_text(encoding="utf-8") == "hello"


def test_write_text_atomically_leaves_no_tmp_file_on_success(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"

    write_text_atomically(target, "hello")

    leftovers = list(tmp_path.glob(".*.tmp"))
    assert leftovers == []


def test_write_json_atomically_round_trips_payload(tmp_path: Path) -> None:
    target = tmp_path / "data.json"

    write_json_atomically(target, {"a": 1, "b": [1, 2, 3]})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}


def test_write_json_atomically_uses_temp_file_and_replace(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "data.json"
    replace_sources: list[Path] = []
    original_replace = Path.replace

    def spy_replace(self, dest):
        replace_sources.append(self)
        return original_replace(self, dest)

    monkeypatch.setattr(Path, "replace", spy_replace)

    write_json_atomically(target, {"a": 1})

    assert replace_sources[0].suffix == ".tmp"
    assert replace_sources[0].parent == target.parent


def test_backup_corrupt_file_renames_with_timestamp_suffix(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text("not json", encoding="utf-8")
    logger = logging.getLogger("test_json_store")

    backup_path = backup_corrupt_file(target, ValueError("bad"), logger)

    assert not target.exists()
    assert backup_path.exists()
    assert backup_path.name.startswith("data.json.corrupt-")
