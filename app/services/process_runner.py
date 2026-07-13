from __future__ import annotations

import asyncio
from pathlib import Path


class SubprocessTimeoutError(RuntimeError):
    pass


async def run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        await _kill_process(process)
        raise SubprocessTimeoutError(f"Process '{Path(command[0]).name}' timed out after {timeout}s") from exc
    except asyncio.CancelledError:
        await _kill_process(process)
        raise

    return stdout, stderr, _resolved_returncode(process)


def _resolved_returncode(process: asyncio.subprocess.Process) -> int:
    return process.returncode if process.returncode is not None else -1


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.wait()
