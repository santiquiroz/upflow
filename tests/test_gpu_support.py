from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from gpu_support import GPU_TESTS_ENV, enforce_gpu_opt_in, gpu_tests_enabled

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def write_test(path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"def {name}():\n    pass\n", encoding="utf-8")


def run_pytest_with_repo_config(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(PYPROJECT), "--rootdir", str(project),
         "-p", "no:cacheprovider", *args],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def project_with_gpu_tests(tmp_path: Path) -> Path:
    write_test(tmp_path / "tests" / "test_cpu.py", "test_cpu")
    write_test(tmp_path / "tests" / "gpu" / "test_on_gpu.py", "test_on_gpu")
    return tmp_path


def test_tests_gpu_is_left_out_of_the_default_collection(project_with_gpu_tests: Path) -> None:
    completed = run_pytest_with_repo_config(project_with_gpu_tests, "--collect-only", "-q")

    assert "test_cpu.py::test_cpu" in completed.stdout
    assert "test_on_gpu" not in completed.stdout


def test_a_gpu_file_named_explicitly_is_still_collected(project_with_gpu_tests: Path) -> None:
    completed = run_pytest_with_repo_config(
        project_with_gpu_tests, "--collect-only", "-q", "tests/gpu/test_on_gpu.py"
    )

    assert "test_on_gpu.py::test_on_gpu" in completed.stdout


def test_the_gpu_marker_is_registered(tmp_path: Path) -> None:
    completed = run_pytest_with_repo_config(tmp_path, "--markers")

    assert "@pytest.mark.gpu:" in completed.stdout


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({GPU_TESTS_ENV: "1"}, True),
        ({GPU_TESTS_ENV: "0"}, False),
        ({}, False),
    ],
)
def test_gpu_tests_are_enabled_only_with_the_flag_set_to_one(environ: dict[str, str], expected: bool) -> None:
    assert gpu_tests_enabled(environ) is expected


def test_a_gpu_test_skips_without_the_opt_in() -> None:
    with pytest.raises(pytest.skip.Exception, match=f"{GPU_TESTS_ENV}=1"):
        enforce_gpu_opt_in(enabled=False)


def test_a_gpu_test_runs_with_the_opt_in() -> None:
    enforce_gpu_opt_in(enabled=True)
