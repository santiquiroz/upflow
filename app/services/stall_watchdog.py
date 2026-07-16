from __future__ import annotations

import time


class StallWatchdog:
    def __init__(self, stall_timeout_seconds: float) -> None:
        self._stall_timeout_seconds = stall_timeout_seconds
        self._last_value = -1
        self._last_change_time = time.monotonic()
        self.triggered = False

    def observe(self, value: int) -> bool:
        if value > self._last_value:
            self._reset(value)
            return False
        self.triggered = self._elapsed_since_change() >= self._stall_timeout_seconds
        return self.triggered

    def _reset(self, value: int) -> None:
        self._last_value = value
        self._last_change_time = time.monotonic()

    def _elapsed_since_change(self) -> float:
        return time.monotonic() - self._last_change_time
