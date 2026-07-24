from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.exceptions import HfDownloadTooLargeError, ModelNotFoundError, ModelProtectedError
from app.services import model_installer
from app.services.hf_client import HfFile
from app.services.model_converter import ConversionResult
from app.services.model_installer import (
    InstallStatus,
    ModelInstaller,
    _model_id_from_repo_id,
    _progress_percent,
    _validate_repo_id,
)
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus

# ---------------------------------------------------------------------------
# SP1 Task 5 - model_installer: installs .onnx models straight from a
# Hugging Face repo_id (HfClient, no real network) into ModelRegistry, on
# its own single-worker asyncio queue (deliberately separate from the GPU
# job queue). Non-.onnx weight files are rejected with a clear "not
# available yet" error -- conversion lands in Task 6.
#
# `_process_next()` is a monkeypatchable test seam: it drains exactly one
# queued job synchronously so tests can assert on the finished InstallJob
# state without spawning/polling a live background asyncio.Task. The real
# `start()`/`stop()` worker lifecycle is covered separately below.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


class FakeHfClient:
    def __init__(
        self,
        files: list[HfFile],
        *,
        download_bytes: bytes = b"fake-onnx-bytes",
        repo_files_error: Exception | None = None,
        download_error: Exception | None = None,
    ) -> None:
        self.files = files
        self.download_bytes = download_bytes
        self.repo_files_error = repo_files_error
        self.download_error = download_error
        self.repo_files_calls: list[str] = []
        self.download_calls: list[tuple[str, str, Path]] = []

    async def repo_files(self, repo_id: str) -> list[HfFile]:
        self.repo_files_calls.append(repo_id)
        if self.repo_files_error:
            raise self.repo_files_error
        return self.files

    async def download(self, repo_id, filename, dest, progress_cb=None):
        self.download_calls.append((repo_id, filename, dest))
        if self.download_error:
            raise self.download_error
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.download_bytes)
        if progress_cb is not None:
            progress_cb(len(self.download_bytes), len(self.download_bytes))
        return dest

    async def search(self, query: str, limit: int = 20):
        return []


class _FakeIoInfo:
    def __init__(self, name: str, shape: list, type_: str) -> None:
        self.name = name
        self.shape = shape
        self.type = type_


class FakeValidSession:
    def __init__(self, scale: int = 2) -> None:
        self._scale = scale
        self._input = _FakeIoInfo("input", [1, 3, "height", "width"], "tensor(float)")
        self._output = _FakeIoInfo("output", [1, 3, "height2", "width2"], "tensor(float)")

    def get_inputs(self):
        return [self._input]

    def get_outputs(self):
        return [self._output]

    def run(self, output_names, input_feed):
        array = input_feed[self._input.name]
        scaled = np.repeat(np.repeat(array, self._scale, axis=2), self._scale, axis=3)
        return [scaled]


class TwoInputSession(FakeValidSession):
    def get_inputs(self):
        return [self._input, self._input]


class ThreeDInputSession(FakeValidSession):
    def __init__(self) -> None:
        super().__init__()
        self._input = _FakeIoInfo("input", [1, 3, "width"], "tensor(float)")


class IntInputSession(FakeValidSession):
    def __init__(self) -> None:
        super().__init__()
        self._input = _FakeIoInfo("input", [1, 3, "height", "width"], "tensor(int64)")


ONNX_FILES = [HfFile(path="ONNX/4x-model-opset17.onnx", size=866_325)]
NON_ONNX_FILES = [
    HfFile(path="model.safetensors", size=4_000_000),
    HfFile(path="model.pth", size=9_000_000),
]
NO_WEIGHT_FILES = [HfFile(path="README.md", size=100)]


def make_installer(
    tmp_path: Path, files: list[HfFile], **hf_kwargs: object
) -> tuple[ModelInstaller, ModelRegistry, FakeHfClient, Settings]:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    hf = FakeHfClient(files, **hf_kwargs)
    installer = ModelInstaller(settings, registry, hf)
    return installer, registry, hf, settings


# ---------------------------------------------------------------------------
# _validate_repo_id
# ---------------------------------------------------------------------------


def test_validate_repo_id_accepts_owner_slash_name() -> None:
    assert _validate_repo_id("Kim2091/2x-AnimeSharpV4") == "Kim2091/2x-AnimeSharpV4"


@pytest.mark.parametrize(
    "bad_repo_id",
    [
        "../../etc",
        "a/b/c",
        "owner/name with space",
        "owner/%00name",
        "",
        "noSlashAtAll",
        "owner/",
        "/name",
        "owner//name",
        "owner/na..me",
    ],
)
def test_validate_repo_id_rejects_malformed_input(bad_repo_id: str) -> None:
    with pytest.raises(ValueError):
        _validate_repo_id(bad_repo_id)


def test_validate_repo_id_rejects_control_characters() -> None:
    with pytest.raises(ValueError):
        _validate_repo_id("owner/na\x00me")


# ---------------------------------------------------------------------------
# _model_id_from_repo_id / _progress_percent
# ---------------------------------------------------------------------------


def test_model_id_from_repo_id_lowercases_and_joins_with_double_dash() -> None:
    assert _model_id_from_repo_id("Kim2091/2x-AnimeSharpV4") == "kim2091--2x-animesharpv4"


def test_progress_percent_computes_rounded_percentage() -> None:
    assert _progress_percent(50, 200) == 25.0


def test_progress_percent_returns_none_without_total() -> None:
    assert _progress_percent(50, None) is None


def test_progress_percent_returns_none_for_zero_total() -> None:
    assert _progress_percent(0, 0) is None


# ---------------------------------------------------------------------------
# install_from_hf: repo_id validation happens before queueing
# ---------------------------------------------------------------------------


async def test_install_from_hf_rejects_malicious_repo_id_before_queueing(tmp_path: Path) -> None:
    installer, _, hf, _ = make_installer(tmp_path, ONNX_FILES)

    with pytest.raises(ValueError):
        await installer.install_from_hf("../../etc/passwd")

    assert hf.repo_files_calls == []
    assert installer._queue.qsize() == 0


async def test_install_from_hf_status_starts_as_downloading(tmp_path: Path) -> None:
    installer, _, _, _ = make_installer(tmp_path, ONNX_FILES)

    install_id = await installer.install_from_hf("org/repo")
    job = installer.status(install_id)

    assert job is not None
    assert job.status == InstallStatus.downloading
    assert job.model_id is None


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


async def test_install_from_hf_happy_path_registers_onnx_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, hf, settings = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession(scale=2))

    install_id = await installer.install_from_hf("Kim2091/2x-AnimeSharpV4")
    processed = await installer._process_next()

    assert processed is True
    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.installed
    assert job.model_id == "kim2091--2x-animesharpv4"
    assert job.error is None
    assert job.progress_pct == 100.0

    entry = registry.get(job.model_id)
    assert entry is not None
    assert entry.kind == ModelKind.onnx
    assert entry.status == ModelStatus.installed
    assert entry.scale == 2
    assert entry.name == "Kim2091/2x-AnimeSharpV4"
    assert entry.source == "https://huggingface.co/Kim2091/2x-AnimeSharpV4"
    assert entry.arch == "4x-model-opset17"
    assert entry.file_path == "onnx/kim2091--2x-animesharpv4.onnx"
    assert entry.size_bytes == 866_325

    onnx_path = settings.models_path / entry.file_path
    assert onnx_path.exists()
    assert onnx_path.read_bytes() == b"fake-onnx-bytes"
    staging_path = onnx_path.with_name(f"{onnx_path.name}.validating")
    assert not staging_path.exists(), "staging file must be promoted (renamed), not left behind"
    assert hf.download_calls == [
        ("Kim2091/2x-AnimeSharpV4", "ONNX/4x-model-opset17.onnx", staging_path)
    ]


# ---------------------------------------------------------------------------
# non-.onnx weight file: routed through model_converter.convert_to_onnx
# (SP1 Task 6). convert_to_onnx itself is unit-tested for real in
# test_model_converter.py (real torch.onnx.export + real onnxruntime
# validation against a fake-but-real Spandrel descriptor); these tests only
# assert the INSTALLER's wiring/routing contract, so `convert_to_onnx` is
# monkeypatched at the module level the same way `_create_validation_session`
# already is for the .onnx path above.
# ---------------------------------------------------------------------------


def _install_fake_convert_to_onnx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    arch: str = "FakeArch",
    scale: int = 4,
    onnx_bytes: bytes = b"converted-onnx-bytes",
    error: Exception | None = None,
) -> list[tuple[Path, Path]]:
    calls: list[tuple[Path, Path]] = []

    def fake_convert(weight_path: Path, out_onnx: Path, progress_cb=None) -> ConversionResult:
        calls.append((weight_path, out_onnx))
        if error is not None:
            raise error
        out_onnx.parent.mkdir(parents=True, exist_ok=True)
        out_onnx.write_bytes(onnx_bytes)
        return ConversionResult(arch=arch, scale=scale)

    monkeypatch.setattr(model_installer, "convert_to_onnx", fake_convert)
    return calls


async def test_install_routes_non_onnx_weight_through_conversion_and_registers_onnx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, hf, settings = make_installer(tmp_path, NON_ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession(scale=2))
    convert_calls = _install_fake_convert_to_onnx(monkeypatch, arch="ESRGAN", scale=4)

    install_id = await installer.install_from_hf("org/pth-repo")
    processed = await installer._process_next()

    assert processed is True
    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.installed
    assert job.error is None

    entry = registry.get(job.model_id)
    assert entry is not None
    assert entry.kind == ModelKind.onnx
    assert entry.status == ModelStatus.installed
    # arch/scale come from ConversionResult (Spandrel metadata), NOT from
    # the filename stem or from FakeValidSession's runtime-detected scale=2.
    assert entry.arch == "ESRGAN"
    assert entry.scale == 4
    assert entry.file_path == "onnx/org--pth-repo.onnx"

    onnx_path = settings.models_path / entry.file_path
    assert onnx_path.read_bytes() == b"converted-onnx-bytes"

    # pick_weight_file prefers .safetensors over .pth (NON_ONNX_FILES has both).
    assert len(convert_calls) == 1
    source_weight_path, out_onnx_arg = convert_calls[0]
    assert source_weight_path.suffix == ".safetensors"
    assert out_onnx_arg.name == "org--pth-repo.onnx.validating"


async def test_install_reports_converting_status_during_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, _, _, _ = make_installer(tmp_path, NON_ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession(scale=2))

    install_id = await installer.install_from_hf("org/pth-repo")
    job = installer.status(install_id)
    assert job is not None
    observed_statuses: list[InstallStatus] = []

    def fake_convert(weight_path: Path, out_onnx: Path, progress_cb=None) -> ConversionResult:
        observed_statuses.append(job.status)
        out_onnx.parent.mkdir(parents=True, exist_ok=True)
        out_onnx.write_bytes(b"onnx-bytes")
        return ConversionResult(arch="Arch", scale=2)

    monkeypatch.setattr(model_installer, "convert_to_onnx", fake_convert)

    await installer._process_next()

    assert observed_statuses == [InstallStatus.converting]


async def test_install_deletes_original_weight_file_after_successful_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, _, _, settings = make_installer(tmp_path, NON_ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession(scale=2))
    _install_fake_convert_to_onnx(monkeypatch)

    await installer.install_from_hf("org/pth-repo")
    await installer._process_next()

    source_weight_path = settings.temp_path / "org--pth-repo.safetensors"
    assert not source_weight_path.exists()


async def test_install_uses_converted_onnx_file_size_not_original_weight_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, _, _ = make_installer(tmp_path, NON_ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession(scale=2))
    onnx_bytes = b"x" * 123
    _install_fake_convert_to_onnx(monkeypatch, onnx_bytes=onnx_bytes)

    install_id = await installer.install_from_hf("org/pth-repo")
    await installer._process_next()

    job = installer.status(install_id)
    entry = registry.get(job.model_id)
    assert entry.size_bytes == 123
    # NON_ONNX_FILES' .safetensors entry declares 4_000_000 bytes on HF --
    # the registered size must reflect the converted .onnx, not that.
    assert entry.size_bytes != 4_000_000


async def test_install_marks_error_when_conversion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, hf, settings = make_installer(tmp_path, NON_ONNX_FILES)
    _install_fake_convert_to_onnx(monkeypatch, error=RuntimeError("conversion boom"))

    install_id = await installer.install_from_hf("org/broken-weights")
    await installer._process_next()

    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.error
    assert "conversion boom" in job.error
    assert registry.get("org--broken-weights") is None
    assert list((settings.models_path / "onnx").glob("*")) == []
    assert not (settings.temp_path / "org--broken-weights.safetensors").exists()


async def test_install_error_when_no_weight_file_present(tmp_path: Path) -> None:
    installer, registry, _, _ = make_installer(tmp_path, NO_WEIGHT_FILES)

    install_id = await installer.install_from_hf("org/empty")
    await installer._process_next()

    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.error
    assert "weight file" in job.error
    assert registry.get("org--empty") is None


# ---------------------------------------------------------------------------
# ONNX validation failures: file must be cleaned up, registry untouched
# ---------------------------------------------------------------------------


async def test_install_marks_error_when_onnx_has_multiple_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, _, settings = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: TwoInputSession())

    install_id = await installer.install_from_hf("org/bad-inputs")
    await installer._process_next()

    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.error
    assert "1 input" in job.error
    assert registry.get("org--bad-inputs") is None
    assert not (settings.models_path / "onnx" / "org--bad-inputs.onnx").exists()
    assert not (settings.models_path / "onnx" / "org--bad-inputs.onnx.validating").exists()
    assert list((settings.models_path / "onnx").glob("*")) == []


async def test_install_marks_error_when_onnx_input_not_4d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, _, _ = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: ThreeDInputSession())

    install_id = await installer.install_from_hf("org/bad-shape")
    await installer._process_next()

    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.error
    assert "4D" in job.error
    assert registry.get("org--bad-shape") is None


async def test_install_marks_error_when_onnx_input_not_float(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, _, _ = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: IntInputSession())

    install_id = await installer.install_from_hf("org/bad-dtype")
    await installer._process_next()

    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.error
    assert "float" in job.error.lower()
    assert registry.get("org--bad-dtype") is None


async def test_reinstall_failure_does_not_delete_previously_installed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the first successful install writes onnx/<model_id>.onnx.
    # A later reinstall of the SAME repo that fails validation must not wipe
    # out that previously-working file -- download+validate happens under a
    # staging name, only promoted to the final path on success.
    installer, registry, _, settings = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession(scale=2))

    first_install_id = await installer.install_from_hf("org/reinstall-me")
    await installer._process_next()
    first_job = installer.status(first_install_id)
    assert first_job is not None and first_job.status == InstallStatus.installed
    entry = registry.get(first_job.model_id)
    assert entry is not None
    onnx_path = settings.models_path / entry.file_path
    original_bytes = onnx_path.read_bytes()

    monkeypatch.setattr(installer, "_create_validation_session", lambda path: TwoInputSession())
    second_install_id = await installer.install_from_hf("org/reinstall-me")
    await installer._process_next()
    second_job = installer.status(second_install_id)

    assert second_job is not None
    assert second_job.status == InstallStatus.error
    assert onnx_path.exists()
    assert onnx_path.read_bytes() == original_bytes
    reloaded_entry = registry.get(first_job.model_id)
    assert reloaded_entry is not None
    assert reloaded_entry.status == ModelStatus.installed


# ---------------------------------------------------------------------------
# download-time failures
# ---------------------------------------------------------------------------


async def test_install_marks_error_on_download_too_large(tmp_path: Path) -> None:
    installer, registry, _, _ = make_installer(
        tmp_path, ONNX_FILES, download_error=HfDownloadTooLargeError("too big")
    )

    install_id = await installer.install_from_hf("org/huge")
    await installer._process_next()

    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.error
    assert "too big" in job.error
    assert registry.get("org--huge") is None


async def test_install_marks_error_when_repo_files_fails(tmp_path: Path) -> None:
    installer, registry, _, _ = make_installer(tmp_path, [], repo_files_error=RuntimeError("hub down"))

    install_id = await installer.install_from_hf("org/down")
    await installer._process_next()

    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.error
    assert "hub down" in job.error
    assert registry.get("org--down") is None


# ---------------------------------------------------------------------------
# M1 fast-follow: promote (staging_dest.replace(final_dest)) retries a
# transient Windows PermissionError instead of failing the whole install --
# this was the source of the flake noted in the SP1 Task 8 smoke report (an
# ORT validation session or a warm cached session on reinstall can hold the
# file handle a moment longer than the replace() call).
# ---------------------------------------------------------------------------


def _patch_staging_replace(monkeypatch: pytest.MonkeyPatch, *, fail_times: int | None) -> list[int]:
    """Patches Path.replace so only staging (`*.onnx.validating`) targets are
    affected -- ModelRegistry._persist's own tmp-then-replace write must keep
    working normally, or these tests would spuriously fail on registry
    persistence instead of exercising the promote retry path.
    """
    original_replace = Path.replace
    call_count = [0]

    def fake_replace(self: Path, target: Path) -> Path:
        if self.name.endswith(".onnx.validating"):
            call_count[0] += 1
            if fail_times is None or call_count[0] <= fail_times:
                raise PermissionError("[WinError 5] Access is denied")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fake_replace)
    return call_count


async def test_install_promote_retries_permission_error_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, _, settings = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession(scale=2))
    call_count = _patch_staging_replace(monkeypatch, fail_times=2)

    install_id = await installer.install_from_hf("org/flaky-replace")
    await installer._process_next()

    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.installed
    assert job.error is None
    assert call_count[0] == 3, "expected 2 failures then a successful 3rd attempt"

    entry = registry.get(job.model_id)
    assert entry is not None
    onnx_path = settings.models_path / entry.file_path
    assert onnx_path.exists()
    staging_path = onnx_path.with_name(f"{onnx_path.name}.validating")
    assert not staging_path.exists()


async def test_install_promote_surfaces_clean_error_when_always_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, _, settings = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession(scale=2))
    _patch_staging_replace(monkeypatch, fail_times=None)

    install_id = await installer.install_from_hf("org/permanently-locked")
    await installer._process_next()

    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.error
    assert job.error is not None
    assert "locked" in job.error.lower() or "permission" in job.error.lower()
    assert registry.get("org--permanently-locked") is None

    staging_path = settings.models_path / "onnx" / "org--permanently-locked.onnx.validating"
    assert not staging_path.exists(), "staging file must be cleaned up after retries are exhausted"
    assert not (settings.models_path / "onnx" / "org--permanently-locked.onnx").exists()


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


def test_status_returns_none_for_unknown_install_id(tmp_path: Path) -> None:
    installer, _, _, _ = make_installer(tmp_path, ONNX_FILES)

    assert installer.status("does-not-exist") is None


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


async def test_delete_raises_model_not_found_for_unknown_id(tmp_path: Path) -> None:
    installer, _, _, _ = make_installer(tmp_path, ONNX_FILES)

    with pytest.raises(ModelNotFoundError):
        await installer.delete("does-not-exist")


async def test_delete_raises_model_protected_for_builtin(tmp_path: Path) -> None:
    installer, registry, _, _ = make_installer(tmp_path, ONNX_FILES)

    with pytest.raises(ModelProtectedError):
        await installer.delete("realesrgan-x4plus")

    assert registry.get("realesrgan-x4plus") is not None


async def test_delete_removes_installed_onnx_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    installer, registry, _, _ = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession())
    install_id = await installer.install_from_hf("org/removable")
    await installer._process_next()
    job = installer.status(install_id)
    assert job is not None and job.model_id is not None

    await installer.delete(job.model_id)

    assert registry.get(job.model_id) is None


async def test_delete_removes_onnx_file_from_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    installer, registry, _, settings = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession())
    install_id = await installer.install_from_hf("org/removable")
    await installer._process_next()
    job = installer.status(install_id)
    assert job is not None and job.model_id is not None
    entry = registry.get(job.model_id)
    assert entry is not None
    onnx_path = settings.models_path / entry.file_path
    assert onnx_path.exists()

    await installer.delete(job.model_id)

    assert not onnx_path.exists()
    assert registry.get(job.model_id) is None


async def test_delete_builtin_does_not_touch_any_onnx_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, _, settings = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession())
    install_id = await installer.install_from_hf("org/keepme")
    await installer._process_next()
    job = installer.status(install_id)
    assert job is not None
    onnx_path = settings.models_path / registry.get(job.model_id).file_path
    assert onnx_path.exists()

    with pytest.raises(ModelProtectedError):
        await installer.delete("realesrgan-x4plus")

    assert onnx_path.exists(), "deleting a builtin must not touch other models' files"


async def test_delete_does_not_remove_file_outside_models_dir(tmp_path: Path) -> None:
    # A manipulated/corrupt entry whose file_path escapes the models dir must
    # never let delete() unlink an arbitrary file on disk. The registry entry
    # is still removed; only the out-of-dir file deletion is refused.
    installer, registry, _, settings = make_installer(tmp_path, ONNX_FILES)
    outside_file = tmp_path / "outside.onnx"
    outside_file.write_bytes(b"do-not-delete-me")
    registry.register(
        ModelEntry(
            id="evil-entry",
            name="evil",
            kind=ModelKind.onnx,
            source="https://huggingface.co/evil/evil",
            size_bytes=10,
            scale=4,
            file_path=str(outside_file),
        )
    )

    await installer.delete("evil-entry")

    assert outside_file.exists(), "delete() must not unlink files outside the models dir"
    assert registry.get("evil-entry") is None


async def test_delete_diffusion_model_removes_directory_recursively(tmp_path: Path) -> None:
    installer, registry, _, settings = make_installer(tmp_path, files=[])
    model_dir = settings.models_path / "generation" / "gen--amd--sd15"
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (model_dir / "unet" / "model.onnx").write_bytes(b"onnx")
    registry.register(
        ModelEntry(
            id="gen--amd--sd15",
            name="amd/sd15",
            kind=ModelKind.diffusion_onnx,
            source="hf:amd/sd15",
            size_bytes=4,
            scale=None,
            file_path="generation/gen--amd--sd15",
        )
    )

    await installer.delete("gen--amd--sd15")

    assert not model_dir.exists()
    assert registry.get("gen--amd--sd15") is None


async def test_delete_diffusion_model_rejects_path_escaping_models_root(tmp_path: Path) -> None:
    # A manipulated/corrupt entry whose file_path escapes the models dir must
    # never let delete() remove an arbitrary directory on disk. The registry entry
    # is still removed; only the out-of-dir directory deletion is refused.
    installer, registry, _, settings = make_installer(tmp_path, files=[])
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (outside / "nested").mkdir()
    (outside / "nested" / "file.txt").write_text("do-not-delete-me")
    registry.register(
        ModelEntry(
            id="gen--evil",
            name="evil",
            kind=ModelKind.diffusion_onnx,
            source="hf:evil/evil",
            size_bytes=1,
            scale=None,
            file_path=str(outside),
        )
    )

    await installer.delete("gen--evil")

    assert outside.exists(), "delete() must not rmtree directories outside the models dir"
    assert (outside / "nested" / "file.txt").exists(), "nested content must survive"
    assert registry.get("gen--evil") is None


async def test_delete_diffusion_model_logs_and_swallows_locked_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Item 2 (final whole-branch review): shutil.rmtree(..., ignore_errors=True)
    # swallowed the error INSIDE rmtree itself, so the surrounding `except
    # OSError` (which logs) never fired for a locked directory -- unlike the
    # sibling file-unlink path a few lines up, which does log. A multi-GB
    # install could silently fail to delete with zero trace.
    installer, registry, _, settings = make_installer(tmp_path, files=[])
    model_dir = settings.models_path / "generation" / "gen--amd--sd15"
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    registry.register(
        ModelEntry(
            id="gen--amd--sd15",
            name="amd/sd15",
            kind=ModelKind.diffusion_onnx,
            source="hf:amd/sd15",
            size_bytes=4,
            scale=None,
            file_path="generation/gen--amd--sd15",
        )
    )

    def flaky_rmtree(path, *args, ignore_errors: bool = False, **kwargs: object) -> None:
        if ignore_errors:
            return  # mirrors real shutil.rmtree(ignore_errors=True): swallows silently
        raise OSError("[WinError 5] Access is denied")

    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)

    with caplog.at_level(logging.ERROR):
        await installer.delete("gen--amd--sd15")  # must not raise

    assert any(record.levelno >= logging.ERROR for record in caplog.records), (
        "a locked directory delete failure must be logged, matching the file-unlink path"
    )
    assert registry.get("gen--amd--sd15") is None


# ---------------------------------------------------------------------------
# worker lifecycle (own single-worker queue, separate from the GPU queue)
# ---------------------------------------------------------------------------


async def test_start_spawns_single_worker_task(tmp_path: Path) -> None:
    installer, _, _, _ = make_installer(tmp_path, ONNX_FILES)

    await installer.start()
    try:
        assert installer._worker_task is not None
        assert not installer._worker_task.done()
    finally:
        await installer.stop()

    assert installer._worker_task is None


async def test_start_is_idempotent(tmp_path: Path) -> None:
    installer, _, _, _ = make_installer(tmp_path, ONNX_FILES)

    await installer.start()
    first_task = installer._worker_task
    await installer.start()

    assert installer._worker_task is first_task
    await installer.stop()


async def test_worker_processes_queued_install_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, _, _ = make_installer(tmp_path, ONNX_FILES)
    monkeypatch.setattr(installer, "_create_validation_session", lambda path: FakeValidSession())

    await installer.start()
    try:
        install_id = await installer.install_from_hf("org/via-worker")
        await installer._queue.join()
        job = installer.status(install_id)
        assert job is not None
        assert job.status == InstallStatus.installed
        assert registry.get(job.model_id) is not None
    finally:
        await installer.stop()
