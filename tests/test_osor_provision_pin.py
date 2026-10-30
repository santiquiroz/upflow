from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from app.config import Settings
from app.services.license_gate import LICENSE_REQUIRED_KEY, LicenseNotAcceptedError
from app.services.pack_provisioner import PackProvisioner
from app.services.restore_models import LICENSE_GATED_PACKS


ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = ROOT / "scripts" / "fetch_osor_sources.py"
BASE_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
BASE_REVISION = "115134f363124c53c7d878647567d04daf26e41e"
WEIGHTS_REVISION = "9288156766b1df44e5f50fb697876a6f2fd8c062"
PINNED_FILES = {
    "unet/diffusion_pytorch_model.fp16.safetensors": (
        5135178560,
        "6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f",
    ),
    "vae/diffusion_pytorch_model.fp16.safetensors": (
        167335338,
        "4ad62825e5c8b31eefb77355ba6693785619bf7d669d9b1a6fd9f19dec6d65b3",
    ),
    "text_encoder/model.fp16.safetensors": (
        246144867,
        "fc83cf401d930147807e7c44021c164bcc5508c9d4cc0ff35f4e354685ca9cd0",
    ),
    "text_encoder_2/model.fp16.safetensors": (
        1389382884,
        "a8622bd41f8d359df484fdf9de091edb9337a2fd747f0a5c9c320a62e24d1fd3",
    ),
    "osor-sdxlinpainting/weights/sdxlinpainting_phase2.pth": (
        3112296806,
        "4f6abee0f152919fa8faca1895820ee531adb2b42b5de25db2e9a311aa47face",
    ),
}


def load_fetch_module(monkeypatch, download, list_files=None):
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    hub.list_repo_files = list_files or (lambda repo_id, revision: [])
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    spec = importlib.util.spec_from_file_location("fetch_osor_sources", FETCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_script_contains_pinned_revisions_files_sizes_and_hashes():
    source = FETCH_SCRIPT.read_text(encoding="utf-8")

    assert BASE_REVISION in source
    assert WEIGHTS_REVISION in source
    assert "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" in source
    assert "QinmingZhou/OSOR" in source
    for relative, (size, sha256) in PINNED_FILES.items():
        assert relative in source
        assert str(size) in source
        assert sha256 in source
    assert "hf_hub_download" in source
    assert "list_repo_files" in source


def test_hash_mismatch_removes_downloaded_file_and_fails(tmp_path, monkeypatch):
    downloaded = []

    def fake_download(repo_id, filename, *, revision, local_dir):
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bad")
        downloaded.append(destination)
        return str(destination)

    module = load_fetch_module(monkeypatch, fake_download)
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(total=20 * 1024**3, used=0, free=20 * 1024**3),
    )
    relative = next(iter(module.PINNED_BASE_FILES))
    module.PINNED_BASE_FILES[relative] = (3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    with pytest.raises(SystemExit) as caught:
        module.main(["--dest", str(tmp_path / "sources")])

    assert caught.value.code != 0
    assert downloaded
    assert not downloaded[0].exists()


def writing_download(downloaded: list[Path], content: bytes = b"abc"):
    def fake_download(repo_id, filename, *, revision, local_dir):
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        downloaded.append(destination)
        return str(destination)

    return fake_download


def with_plenty_of_space(module, monkeypatch):
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(total=20 * 1024**3, used=0, free=20 * 1024**3),
    )


def test_small_base_files_keeps_only_configs_scheduler_and_tokenizers(monkeypatch):
    repo_files = [
        ".gitattributes",
        "README.md",
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "scheduler/scheduler_config.json",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer_2/merges.txt",
        "tokenizer_2/vocab.json",
        "tokenizers/vocab.json",
        "inpaint.png",
    ]
    calls = []

    def fake_list(repo_id, revision):
        calls.append((repo_id, revision))
        return repo_files

    module = load_fetch_module(monkeypatch, lambda *args, **kwargs: None, fake_list)

    assert module.small_base_files() == sorted(
        [
            "unet/config.json",
            "vae/config.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "scheduler/scheduler_config.json",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/vocab.json",
        ]
    )
    assert calls == [(BASE_REPO, BASE_REVISION)]


def test_size_mismatch_removes_downloaded_file_and_fails(tmp_path, monkeypatch):
    downloaded = []
    module = load_fetch_module(monkeypatch, writing_download(downloaded))
    with_plenty_of_space(module, monkeypatch)
    relative = next(iter(module.PINNED_BASE_FILES))
    module.PINNED_BASE_FILES[relative] = (4, hashlib.sha256(b"abc").hexdigest())

    with pytest.raises(SystemExit) as caught:
        module.main(["--dest", str(tmp_path / "sources")])

    assert caught.value.code != 0
    assert "bytes" in str(caught.value.code)
    assert len(downloaded) == 1
    assert not downloaded[0].exists()


def test_weights_hash_mismatch_removes_the_weights_file_and_fails(tmp_path, monkeypatch):
    downloaded = []
    module = load_fetch_module(monkeypatch, writing_download(downloaded))
    with_plenty_of_space(module, monkeypatch)
    good = (3, hashlib.sha256(b"abc").hexdigest())
    for relative in module.PINNED_BASE_FILES:
        module.PINNED_BASE_FILES[relative] = good
    module.PINNED_WEIGHTS_FILES[module.WEIGHTS_FILE] = (3, "0" * 64)

    with pytest.raises(SystemExit) as caught:
        module.main(["--dest", str(tmp_path / "sources")])

    assert caught.value.code != 0
    weights = tmp_path / "sources" / module.WEIGHTS_SUBDIR / module.WEIGHTS_FILE
    assert downloaded[-1] == weights
    assert not weights.exists()
    assert all(path.exists() for path in downloaded[:-1])


def test_insufficient_disk_space_fails_before_download(tmp_path, monkeypatch):
    downloads = []
    module = load_fetch_module(
        monkeypatch, lambda *args, **kwargs: downloads.append((args, kwargs))
    )
    monkeypatch.setattr(
        module.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(total=20 * 1024**3, used=10 * 1024**3, free=10 * 1024**3),
    )

    with pytest.raises(SystemExit) as caught:
        module.main(["--dest", str(tmp_path / "sources")])

    assert caught.value.code != 0
    assert downloads == []


@pytest.mark.asyncio
async def test_osor_provisioning_requires_license_acceptance(tmp_path):
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    licenses = tmp_path / "gated"
    licenses.mkdir()
    (licenses / "osor.txt").write_text("CreativeML Open RAIL++-M\n", encoding="utf-8")
    provisioner = PackProvisioner(
        settings,
        gated_packs=LICENSE_GATED_PACKS,
        gated_licenses_dir=licenses,
    )

    with pytest.raises(LicenseNotAcceptedError) as caught:
        await provisioner.provision("osor")

    assert caught.value.key == LICENSE_REQUIRED_KEY
    assert "osor" in LICENSE_GATED_PACKS
