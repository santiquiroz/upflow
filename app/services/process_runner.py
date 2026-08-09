from __future__ import annotations

import asyncio
from pathlib import Path


class SubprocessTimeoutError(RuntimeError):
    pass


def is_non_empty_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


async def run_checked_process(
    command: list[str],
    timeout: float,
    failure_message: str,
    expect_output: Path | None = None,
) -> bytes:
    stdout, stderr, returncode = await run_guarded_process(command, timeout)
    if returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="ignore") or failure_message)
    if expect_output is not None and not is_non_empty_file(expect_output):
        raise RuntimeError(
            f"Process '{Path(command[0]).name}' completed but no output file was produced"
        )
    return stdout


async def run_guarded_process(
    command: list[str], timeout: float, *, env: dict[str, str] | None = None
) -> tuple[bytes, bytes, int]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
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
