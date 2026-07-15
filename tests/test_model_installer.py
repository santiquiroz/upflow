from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.exceptions import HfDownloadTooLargeError, ModelNotFoundError, ModelProtectedError
from app.services.hf_client import HfFile
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
# non-.onnx weight file: rejected without downloading (Task 6 will replace
# this branch with real .pth/.safetensors conversion)
# ---------------------------------------------------------------------------


async def test_install_rejects_non_onnx_weight_file_without_downloading(tmp_path: Path) -> None:
    installer, registry, hf, _ = make_installer(tmp_path, NON_ONNX_FILES)

    install_id = await installer.install_from_hf("org/pth-repo")
    await installer._process_next()

    job = installer.status(install_id)
    assert job is not None
    assert job.status == InstallStatus.error
    assert ".onnx" in job.error
    assert hf.download_calls == []
    assert registry.get("org--pth-repo") is None


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
