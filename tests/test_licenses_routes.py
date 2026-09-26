from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import licenses_routes
from app.config import Settings, get_settings
from app.services.licenses_view import (
    NOTICES_PATH,
    LicenseCatalog,
    licenses_view,
    notice_entries,
)
from app.services.restore_models import (
    RestoreArtifact,
    RestoreBundle,
    RestoreLicenseFile,
    RestoreModelSpec,
    VendoredModel,
)

RELEASE = "https://github.com/santiquiroz/port-restore-onnx/releases/download/v1.0.0"
SHA = "a" * 64
LICENSE_TEXT = "MIT License\n\nCopyright (c) 2021 Kai Zhang\n"
NOTICE_TEXT = "Modified by port-restore-onnx on 2026-09-20: exported to ONNX.\n"

NOTICES = """# Third-party notices

Intro text.

## Components bundled in the release

### Real-ESRGAN ONNX exports

- Component: `vendor/realesrgan-onnx/`
- License: BSD-3-Clause
- Copyright: Copyright (c) 2021, Xintao Wang

```text
BSD 3-Clause License
- Fake: this line lives inside the license text
## Not a heading either
```

## Ported source code

### RetinaFace priors and decoding

- Ported into: app/services/engines/face_detect.py
- License: MIT
- Source: https://github.com/yakhyo/retinaface-pytorch (commit 7601e1c)
- Source: https://github.com/biubug6/Pytorch_Retinaface (commit b984b4b)
"""


def make_spec(**overrides) -> RestoreModelSpec:
    fields = dict(
        id="drunet-color",
        name="DRUNet color",
        bundle="core",
        filename="drunet-color.onnx",
        license_spdx="MIT",
        license_url="https://github.com/cszn/KAIR/blob/master/LICENSE",
        copyright="Copyright (c) 2021 Kai Zhang",
        attribution="DRUNet from KAIR by Kai Zhang",
        data_lineage="D1a",
        commercial_use="yes",
        source_url="https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth",
        source_revision="v1.0",
        source_sha256=SHA,
        modifications=("exported to ONNX opset 17",),
        tile_min=128,
    )
    return RestoreModelSpec(**{**fields, **overrides})


def license_file(filename: str, text: str) -> RestoreLicenseFile:
    return RestoreLicenseFile(
        model_id="drunet-color",
        filename=filename,
        url=f"{RELEASE}/licenses--drunet-color--{filename}",
        sha256=SHA,
        size=len(text.encode("utf-8")),
    )


def core_bundle() -> RestoreBundle:
    artifact = RestoreArtifact(
        model_id="drunet-color",
        precision="fp32",
        filename="drunet-color.onnx",
        url=f"{RELEASE}/drunet-color.onnx",
        sha256=SHA,
        size=10,
    )
    files = (license_file("LICENSE", LICENSE_TEXT), license_file("NOTICE.txt", NOTICE_TEXT))
    return RestoreBundle(name="core", artifacts=(artifact,), license_files=files)


def install(model_dir: Path, bundle: RestoreBundle) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for artifact in bundle.artifacts:
        (model_dir / artifact.filename).write_bytes(b"x" * artifact.size)
    texts = {"LICENSE": LICENSE_TEXT, "NOTICE.txt": NOTICE_TEXT}
    for entry in bundle.license_files:
        path = model_dir / entry.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(texts[entry.filename].encode("utf-8"))
    (model_dir / bundle.manifest_name).write_text(json.dumps({"bundle": bundle.name}), encoding="utf-8")


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        RESTORE_MODEL_DIR=str(tmp_path / "restore"),
    )


def vendored_migan(path: Path) -> dict[str, VendoredModel]:
    spec = make_spec(id="migan", name="MI-GAN", bundle="migan", filename="migan.onnx")
    return {"migan": VendoredModel(spec, pack="migan", path_of=lambda settings: path)}


def empty_catalog(**overrides) -> LicenseCatalog:
    fields = dict(models={}, bundles={name: RestoreBundle(name, ()) for name in ("core", "faces")}, vendored={})
    return LicenseCatalog(**{**fields, **overrides})


def test_an_installed_pack_lists_its_models_with_license_lineage_and_files(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    bundle = core_bundle()
    install(settings.restore_model_dir_path, bundle)
    catalog = empty_catalog(models={"drunet-color": make_spec()}, bundles={"core": bundle})

    view = licenses_view(settings, catalog, notices_path=tmp_path / "missing.md")

    assert [pack["pack"] for pack in view["packs"]] == ["restore-core"]
    model = view["packs"][0]["models"][0]
    assert model["id"] == "drunet-color"
    assert model["licenseSpdx"] == "MIT"
    assert model["dataLineage"] == "D1a"
    assert model["commercialUse"] == "yes"
    assert [(f["name"], f["text"]) for f in model["files"]] == [("LICENSE", LICENSE_TEXT), ("NOTICE.txt", NOTICE_TEXT)]
    assert view["thirdParty"] == []


def test_a_pack_that_is_not_installed_is_not_listed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    catalog = empty_catalog(models={"drunet-color": make_spec()}, bundles={"core": core_bundle()})

    assert licenses_view(settings, catalog, notices_path=tmp_path / "missing.md")["packs"] == []


def test_a_vendored_model_lists_the_license_files_next_to_it(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    model = tmp_path / "vendor" / "migan" / "migan.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx")
    (model.parent / "LICENSE").write_text("MIT", encoding="utf-8")
    (model.parent / "LICENSE-WEIGHTS").write_text("MIT weights", encoding="utf-8")
    (model.parent / "README.md").write_text("not a license", encoding="utf-8")

    view = licenses_view(settings, empty_catalog(vendored=vendored_migan(model)), notices_path=tmp_path / "x.md")

    assert [pack["pack"] for pack in view["packs"]] == ["migan"]
    files = view["packs"][0]["models"][0]["files"]
    assert [entry["name"] for entry in files] == ["LICENSE", "LICENSE-WEIGHTS"]


def test_a_missing_vendored_model_is_not_listed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    catalog = empty_catalog(vendored=vendored_migan(tmp_path / "nowhere" / "migan.onnx"))

    assert licenses_view(settings, catalog, notices_path=tmp_path / "x.md")["packs"] == []


def test_notice_entries_keep_section_fields_and_license_text() -> None:
    entries = notice_entries(NOTICES)

    assert [(entry["section"], entry["title"]) for entry in entries] == [
        ("Components bundled in the release", "Real-ESRGAN ONNX exports"),
        ("Ported source code", "RetinaFace priors and decoding"),
    ]
    esrgan, retina = entries
    assert esrgan["fields"] == {
        "Component": ["vendor/realesrgan-onnx/"],
        "License": ["BSD-3-Clause"],
        "Copyright": ["Copyright (c) 2021, Xintao Wang"],
    }
    assert esrgan["licenseText"].startswith("BSD 3-Clause License")
    assert "## Not a heading either" in esrgan["licenseText"]
    assert retina["fields"]["Source"] == [
        "https://github.com/yakhyo/retinaface-pytorch (commit 7601e1c)",
        "https://github.com/biubug6/Pytorch_Retinaface (commit b984b4b)",
    ]
    assert retina["licenseText"] is None


def test_the_shipped_notices_parse_into_entries_with_a_license() -> None:
    entries = notice_entries(NOTICES_PATH.read_text(encoding="utf-8"))

    assert entries
    assert all(entry["fields"].get("License") for entry in entries)
    assert any(entry["title"] == "RetinaFace priors and decoding" for entry in entries)


def test_get_licenses_coroutine_returns_packs_and_third_party(tmp_path: Path) -> None:
    response = asyncio.run(licenses_routes.get_licenses(settings=make_settings(tmp_path)))

    assert response.packs == []
    assert any(entry.title == "Real-ESRGAN ONNX exports" for entry in response.third_party)


def test_licenses_endpoint_serializes_camel_case(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(licenses_routes.router)
    app.dependency_overrides[get_settings] = lambda: make_settings(tmp_path)

    body = TestClient(app).get("/api/v1/licenses").json()

    assert set(body) == {"packs", "thirdParty"}
    first = body["thirdParty"][0]
    assert set(first) == {"title", "section", "fields", "licenseText"}


def license_gate_client(monkeypatch, tmp_path: Path, gated: frozenset[str]) -> TestClient:
    import functools

    from app.services.license_gate import license_gate

    monkeypatch.setattr(
        licenses_routes, "license_gate", functools.partial(license_gate, gated=gated, directory=tmp_path)
    )
    app = FastAPI()
    app.include_router(licenses_routes.router)
    return TestClient(app)


def test_the_license_of_an_ungated_pack_says_there_is_no_gate(monkeypatch, tmp_path: Path) -> None:
    body = license_gate_client(monkeypatch, tmp_path, frozenset()).get("/api/v1/packs/rife/license").json()

    assert body == {"pack": "rife", "gated": False, "licenseText": None}


def test_the_license_of_a_gated_pack_carries_its_full_text(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "rife.txt").write_text("S-Lab License 1.0\n\nNon-commercial use only.\n", encoding="utf-8")

    body = license_gate_client(monkeypatch, tmp_path, frozenset({"rife"})).get("/api/v1/packs/rife/license").json()

    assert body == {"pack": "rife", "gated": True, "licenseText": "S-Lab License 1.0\n\nNon-commercial use only.\n"}


def test_the_license_of_an_unknown_pack_is_a_404(monkeypatch, tmp_path: Path) -> None:
    response = license_gate_client(monkeypatch, tmp_path, frozenset()).get("/api/v1/packs/no-such-pack/license")

    assert response.status_code == 404
