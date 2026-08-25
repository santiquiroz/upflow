from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.config import Settings
from app.services import blender_service
from app.services.blender_service import (
    RESULT_SENTINEL,
    BlenderBuild,
    BlenderError,
    extract_result,
    parse_version,
    probe,
    require_build,
    run_script,
    script_path,
)
from app.services.missing_pack import MissingPack


@pytest.fixture
def blender_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    binary = tmp_path / "blender.exe"
    binary.write_bytes(b"")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    monkeypatch.setattr(
        Settings,
        "blender_scripts_dir",
        property(lambda _settings: scripts_dir),
    )
    return Settings(_env_file=None, BLENDER_BINARY=str(binary))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Blender 5.2.1 LTS", (5, 2, 1)),
        ("Blender 4.2", (4, 2, 0)),
        ("not a Blender version", None),
    ],
)
def test_parse_version(text: str, expected: tuple[int, int, int] | None) -> None:
    assert parse_version(text) == expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((4, 2, 0), True),
        ((4, 1, 9), False),
        ((5, 2, 1), True),
    ],
)
def test_build_meets_minimum(version: tuple[int, int, int], expected: bool) -> None:
    build = BlenderBuild(path=Path("blender.exe"), version=version)

    assert build.meets_minimum is expected


def test_build_formats_its_version() -> None:
    build = BlenderBuild(path=Path("blender.exe"), version=(5, 2, 1))

    assert build.version_string == "5.2.1"


def test_extract_result_returns_the_last_sentinel() -> None:
    stdout = (
        "Blender startup noise\n"
        f'{RESULT_SENTINEL}{json.dumps({"stage": "importing"})}\n'
        "more noise\n"
        f'{RESULT_SENTINEL}{json.dumps({"ok": True, "path": "mesh.glb"})}\n'
    )

    assert extract_result(stdout) == {"ok": True, "path": "mesh.glb"}


def test_extract_result_returns_none_without_a_sentinel() -> None:
    assert extract_result("Blender startup noise\nfinished\n") is None


def test_extract_result_skips_malformed_json_before_a_valid_result() -> None:
    stdout = (
        f"{RESULT_SENTINEL}{{not-json}}\n"
        f'{RESULT_SENTINEL}{json.dumps({"ok": True})}\n'
    )

    assert extract_result(stdout) == {"ok": True}


class TestScriptPath:
    def test_it_rejects_path_traversal(self, blender_settings: Settings) -> None:
        with pytest.raises(BlenderError, match="fuera"):
            script_path(blender_settings, "../escape.py")

    def test_it_rejects_a_non_python_suffix(self, blender_settings: Settings) -> None:
        with pytest.raises(BlenderError, match="fuera"):
            script_path(blender_settings, "model.txt")

    def test_it_rejects_a_missing_script(self, blender_settings: Settings) -> None:
        with pytest.raises(BlenderError, match="inexistente"):
            script_path(blender_settings, "missing.py")

    def test_it_returns_the_resolved_path_for_a_real_script(
        self, blender_settings: Settings
    ) -> None:
        script = blender_settings.blender_scripts_dir / "make.py"
        script.write_text("# test script\n", encoding="utf-8")

        assert script_path(blender_settings, "make.py") == script.resolve()


class TestProbe:
    def test_it_returns_none_when_the_binary_does_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(
            _env_file=None,
            BLENDER_BINARY=str(tmp_path / "missing-blender.exe"),
        )

        def unexpected_run(*args, **kwargs):
            raise AssertionError("subprocess.run should not be called")

        monkeypatch.setattr(blender_service.subprocess, "run", unexpected_run)

        assert probe(settings) is None

    def test_it_returns_none_when_starting_blender_raises_oserror(
        self, blender_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_run(*args, **kwargs):
            raise OSError("cannot execute")

        monkeypatch.setattr(blender_service.subprocess, "run", failing_run)

        assert probe(blender_settings) is None

    def test_it_returns_the_detected_build(
        self, blender_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Blender 5.2.1 LTS\n",
            stderr="",
        )
        monkeypatch.setattr(blender_service.subprocess, "run", lambda *args, **kwargs: completed)

        build = probe(blender_settings)

        assert build == BlenderBuild(
            path=blender_settings.blender_binary_path,
            version=(5, 2, 1),
        )


class TestRequireBuild:
    def test_it_raises_missing_pack_when_probe_finds_nothing(
        self, blender_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(blender_service, "probe", lambda _settings: None)

        with pytest.raises(MissingPack):
            require_build(blender_settings)

    def test_it_rejects_a_version_below_the_minimum(
        self, blender_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        build = BlenderBuild(blender_settings.blender_binary_path, (4, 1, 9))
        monkeypatch.setattr(blender_service, "probe", lambda _settings: build)

        with pytest.raises(BlenderError, match=r"4\.1\.9.*4\.2"):
            require_build(blender_settings)


@pytest.fixture
def runnable_script(
    blender_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> tuple[Settings, Path, BlenderBuild]:
    script = blender_settings.blender_scripts_dir / "make.py"
    script.write_text("# test script\n", encoding="utf-8")
    build = BlenderBuild(blender_settings.blender_binary_path, (5, 2, 1))
    monkeypatch.setattr(blender_service, "require_build", lambda _settings: build)
    return blender_settings, script.resolve(), build


class TestRunScript:
    def test_it_builds_the_command_and_returns_the_reported_result(
        self,
        runnable_script: tuple[Settings, Path, BlenderBuild],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings, script, build = runnable_script
        payload = {"source": "photo.png", "quality": 3}
        calls: list[tuple[list[str], dict]] = []

        def recording_run(command: list[str], **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=f'{RESULT_SENTINEL}{json.dumps({"mesh": "result.glb"})}\n',
                stderr="Blender noise\n",
            )

        monkeypatch.setattr(blender_service.subprocess, "run", recording_run)

        result = run_script(settings, "make.py", payload, timeout=42)

        assert result == {"mesh": "result.glb"}
        assert len(calls) == 1
        command, kwargs = calls[0]
        assert command == [
            str(build.path),
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            json.dumps(payload),
        ]
        assert command[-2] == "--"
        assert json.loads(command[-1]) == payload
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "timeout": 42,
            "check": False,
        }

    def test_a_failed_process_without_a_result_carries_the_combined_output(
        self,
        runnable_script: tuple[Settings, Path, BlenderBuild],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings, _, _ = runnable_script
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=7,
            stdout="stdout details\n",
            stderr="stderr details\n",
        )
        monkeypatch.setattr(blender_service.subprocess, "run", lambda *args, **kwargs: completed)

        with pytest.raises(BlenderError, match="termino en 7") as exc_info:
            run_script(settings, "make.py", {})

        assert exc_info.value.output == "stdout details\nstderr details\n"

    def test_a_reported_error_becomes_a_blender_error(
        self,
        runnable_script: tuple[Settings, Path, BlenderBuild],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings, _, _ = runnable_script
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f'{RESULT_SENTINEL}{json.dumps({"error": "mesh import failed"})}\n',
            stderr="trace details\n",
        )
        monkeypatch.setattr(blender_service.subprocess, "run", lambda *args, **kwargs: completed)

        with pytest.raises(BlenderError, match="mesh import failed") as exc_info:
            run_script(settings, "make.py", {})

        assert exc_info.value.output.endswith("trace details\n")

    def test_a_timeout_becomes_a_blender_error(
        self,
        runnable_script: tuple[Settings, Path, BlenderBuild],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings, _, _ = runnable_script

        def timing_out(command: list[str], **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        monkeypatch.setattr(blender_service.subprocess, "run", timing_out)

        with pytest.raises(BlenderError, match="paso los 12 s") as exc_info:
            run_script(settings, "make.py", {}, timeout=12)

        assert isinstance(exc_info.value.__cause__, subprocess.TimeoutExpired)
