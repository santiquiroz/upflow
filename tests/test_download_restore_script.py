"""scripts/download-restore.ps1 corrido de verdad, sin red.

La descarga se sustituye definiendo Invoke-WebRequest en la sesion que invoca al
script: en PowerShell la funcion gana al cmdlet. El stub escribe la URL como
contenido, asi cada archivo tiene un sha256 y un tamano conocidos.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from app.services.restore_models import (
    BUNDLE_NAMES,
    RestoreArtifact,
    RestoreBundle,
    RestoreLicenseFile,
    RestoreModelSpec,
    bundle_installed,
    catalog_problems,
    release_license_asset,
)
from restore_script_catalog import (
    SCRIPT,
    render_catalog,
    requires_powershell,
    run_powershell,
    script_text,
    with_catalog,
)

pytestmark = requires_powershell

RELEASE = "https://github.com/santiquiroz/port-restore-onnx/releases/download/v1.0.0"
STUB_DOWNLOAD = (
    "function Invoke-WebRequest { param([string]$Uri, [string]$OutFile, [switch]$UseBasicParsing) "
    "[IO.File]::WriteAllText($OutFile, $Uri) }"
)
STUB_OFFLINE = "function Invoke-WebRequest { throw 'sin red: no deberia bajar nada' }"
WRONG_SHA = "0" * 64

MODELS = {
    "drunet-color": RestoreModelSpec(
        id="drunet-color",
        name="DRUNet color",
        bundle="core",
        filename="drunet-color.onnx",
        fp16_filename="drunet-color-fp16.onnx",
        license_spdx="MIT",
        license_url="https://github.com/cszn/KAIR/blob/master/LICENSE",
        copyright="Copyright (c) 2021 Kai Zhang",
        attribution="DRUNet from KAIR by Kai Zhang",
        data_lineage="D1a",
        commercial_use="yes",
        source_url="https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth",
        source_revision="v1.0",
        source_sha256="a" * 64,
        modifications=("exported to ONNX opset 17",),
        tile_min=128,
    )
}


def served(url: str) -> tuple[str, int]:
    data = url.encode("ascii")
    return hashlib.sha256(data).hexdigest(), len(data)


def artifact(filename: str, precision: str = "fp32", **overrides) -> RestoreArtifact:
    url = f"{RELEASE}/{filename}"
    sha256, size = served(url)
    fields = dict(
        model_id="drunet-color", precision=precision, filename=filename, url=url, sha256=sha256, size=size
    )
    return RestoreArtifact(**{**fields, **overrides})


def license_file(filename: str, **overrides) -> RestoreLicenseFile:
    url = f"{RELEASE}/{release_license_asset('drunet-color', filename)}"
    sha256, size = served(url)
    fields = dict(model_id="drunet-color", filename=filename, url=url, sha256=sha256, size=size)
    return RestoreLicenseFile(**{**fields, **overrides})


def published(
    fp16: RestoreArtifact | None = None, notice: RestoreLicenseFile | None = None
) -> dict[str, RestoreBundle]:
    core = RestoreBundle(
        "core",
        (artifact("drunet-color.onnx"), fp16 or artifact("drunet-color-fp16.onnx", "fp16")),
        (license_file("LICENSE"), notice or license_file("NOTICE.txt")),
    )
    bundles = {"core": core, "faces": RestoreBundle("faces", ()), "colorize": RestoreBundle("colorize", ())}
    assert catalog_problems(MODELS, bundles) == []
    return bundles


def prepare_root(root: Path, bundles: dict[str, RestoreBundle] | None = None) -> Path:
    text = script_text() if bundles is None else with_catalog(script_text(), render_catalog(bundles, MODELS))
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / SCRIPT.name).write_text(text, encoding="utf-8")
    return root


def run(root: Path, bundle: str | None, stub: str = STUB_DOWNLOAD) -> subprocess.CompletedProcess:
    arguments = f" -Bundle {bundle}" if bundle else ""
    return run_powershell(f"{stub}\n& '{root / 'scripts' / SCRIPT.name}'{arguments}", timeout=180)


def output(result: subprocess.CompletedProcess) -> str:
    return (result.stderr + result.stdout).decode("utf-8", "replace")


def restore_dir(root: Path) -> Path:
    return root / "vendor" / "restore"


def leftovers(root: Path) -> list[Path]:
    return sorted(restore_dir(root).rglob("*.download")) if restore_dir(root).exists() else []


@pytest.mark.parametrize("bundle", BUNDLE_NAMES)
def test_while_nothing_is_published_the_script_fails_loudly(tmp_path: Path, bundle: str) -> None:
    result = run(prepare_root(tmp_path), bundle)

    assert result.returncode != 0
    assert "not published yet" in output(result)
    assert not (restore_dir(tmp_path) / f"{bundle}.installed.json").exists()


def test_without_a_bundle_the_script_fails_and_says_which_to_pass(tmp_path: Path) -> None:
    result = run(prepare_root(tmp_path), None)

    assert result.returncode != 0
    assert "-Bundle" in output(result)


def test_an_unknown_bundle_is_rejected(tmp_path: Path) -> None:
    result = run(prepare_root(tmp_path, published()), "everything")

    assert result.returncode != 0
    assert not restore_dir(tmp_path).exists()


def test_it_downloads_verifies_copies_the_licenses_and_writes_the_manifest(tmp_path: Path) -> None:
    bundles = published()

    result = run(prepare_root(tmp_path, bundles), "core")

    assert result.returncode == 0, output(result)
    licenses = restore_dir(tmp_path) / "licenses" / "drunet-color"
    assert (licenses / "LICENSE").read_text(encoding="ascii").endswith("licenses--drunet-color--LICENSE")
    assert (licenses / "NOTICE.txt").is_file()
    assert bundle_installed(restore_dir(tmp_path), bundles["core"]) is True
    assert leftovers(tmp_path) == []


def test_the_manifest_lists_files_hashes_sizes_and_licenses(tmp_path: Path) -> None:
    bundles = published()
    run(prepare_root(tmp_path, bundles), "core")

    manifest = json.loads((restore_dir(tmp_path) / "core.installed.json").read_text(encoding="utf-8"))

    assert manifest["pack"] == "restore-core"
    fp32 = bundles["core"].artifacts[0]
    assert {"model": "drunet-color", "precision": "fp32", "file": fp32.filename, "sha256": fp32.sha256,
            "size": fp32.size, "license": "MIT"} in manifest["files"]
    assert len(manifest["files"]) == 2
    assert {entry["file"] for entry in manifest["licenses"]} == {
        "licenses/drunet-color/LICENSE",
        "licenses/drunet-color/NOTICE.txt",
    }


def test_a_model_that_does_not_match_its_sha256_fails_and_leaves_nothing(tmp_path: Path) -> None:
    bundles = published(fp16=artifact("drunet-color-fp16.onnx", "fp16", sha256=WRONG_SHA))

    result = run(prepare_root(tmp_path, bundles), "core")

    assert result.returncode != 0
    assert "SHA-256" in output(result)
    assert not (restore_dir(tmp_path) / "drunet-color-fp16.onnx").exists()
    assert not (restore_dir(tmp_path) / "core.installed.json").exists()
    assert leftovers(tmp_path) == []


def test_a_model_with_the_wrong_size_fails(tmp_path: Path) -> None:
    good = artifact("drunet-color-fp16.onnx", "fp16")
    bundles = published(fp16=artifact("drunet-color-fp16.onnx", "fp16", size=good.size + 1))

    result = run(prepare_root(tmp_path, bundles), "core")

    assert result.returncode != 0
    assert "bytes" in output(result)
    assert not (restore_dir(tmp_path) / "core.installed.json").exists()


def test_a_license_file_that_does_not_verify_fails_the_install(tmp_path: Path) -> None:
    bundles = published(notice=license_file("NOTICE.txt", sha256=WRONG_SHA))

    result = run(prepare_root(tmp_path, bundles), "core")

    assert result.returncode != 0
    assert not (restore_dir(tmp_path) / "licenses" / "drunet-color" / "NOTICE.txt").exists()
    assert not (restore_dir(tmp_path) / "core.installed.json").exists()


def test_verified_files_are_not_downloaded_again(tmp_path: Path) -> None:
    root = prepare_root(tmp_path, published())
    assert run(root, "core").returncode == 0

    result = run(root, "core", stub=STUB_OFFLINE)

    assert result.returncode == 0, output(result)
    assert (restore_dir(tmp_path) / "core.installed.json").is_file()


def test_a_corrupt_file_on_disk_is_downloaded_again(tmp_path: Path) -> None:
    bundles = published()
    root = prepare_root(tmp_path, bundles)
    assert run(root, "core").returncode == 0
    model = restore_dir(tmp_path) / "drunet-color.onnx"
    model.write_bytes(b"x" * bundles["core"].artifacts[0].size)

    result = run(root, "core")

    assert result.returncode == 0, output(result)
    assert model.read_text(encoding="ascii") == bundles["core"].artifacts[0].url


def test_a_failed_reinstall_does_not_leave_the_old_manifest_behind(tmp_path: Path) -> None:
    root = prepare_root(tmp_path, published())
    assert run(root, "core").returncode == 0
    (restore_dir(tmp_path) / "drunet-color.onnx").write_bytes(b"corrupt")

    result = run(root, "core", stub=STUB_OFFLINE)

    assert result.returncode != 0
    assert not (restore_dir(tmp_path) / "core.installed.json").exists()
