from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

from app.config import Settings


class StorageService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ensure_directories()

    def ensure_directories(self) -> None:
        for path in (
            self.settings.runtime_path,
            self.settings.uploads_path,
            self.settings.outputs_path,
            self.settings.temp_path,
            self.settings.video_work_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload: UploadFile, destination: Path, *, max_mb: int | None = None) -> None:
        size = 0
        limit_mb = max_mb or self.settings.max_upload_mb
        with destination.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > limit_mb * 1024 * 1024:
                    raise ValueError(f"Upload exceeds limit of {limit_mb} MB")
                handle.write(chunk)
        await upload.close()
