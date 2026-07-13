from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.models import UpscaleJob


class UpscaleEngine(ABC):
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def run(self, job: UpscaleJob) -> Path:
        raise NotImplementedError
