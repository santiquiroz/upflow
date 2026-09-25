from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from app.services import restore_models as rm
from app.services.restore_models import (
    BUNDLE_NAMES,
    LICENSE_FIELDS,
    LICENSE_GATED_PACKS,
    RESTORE_BUNDLES,
    RESTORE_MODELS,
    RestoreArtifact,
    RestoreBundle,
    RestoreLicenseFile,
    RestoreModelSpec,
    bundle_installed,
    catalog_problems,
    gated_packs,
    installed_manifest,
    model_path,
)
from restore_script_catalog import (
    catalog_block,
    comparable,
    declared_sha256,
    evaluate_catalog,
    expected_catalog,
    render_catalog,
    requires_powershell,
    script_text,
    validate_set,
)

RELEASE = "https://github.com/santiquiroz/port-restore-onnx/releases/download/v1.0.0"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

VALID_SPEC = dict(
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
    source_sha256=SHA_A,
    modifications=("exported to ONNX opset 17",),
    tile_min=128,
    tile_candidates=(256, 384, 512),
)


def make_spec(**overrides) -> RestoreModelSpec:
    return RestoreModelSpec(**{**VALID_SPEC, **overrides})


def make_artifact(**overrides) -> RestoreArtifact:
    fields = dict(
        model_id="drunet-color",
        precision="fp32",
        filename="drunet-color.onnx",
        url="https://github.com/santiquiroz/port-restore-onnx/releases/download/v1.0.0/drunet-color.onnx",
        sha256=SHA_B,
        size=10,
    )
    return RestoreArtifact(**{**fields, **overrides})


def fp16_artifact() -> RestoreArtifact:
    return make_artifact(
        precision="fp16",
        filename="drunet-color-fp16.onnx",
        url="https://github.com/santiquiroz/port-restore-onnx/releases/download/v1.0.0/drunet-color-fp16.onnx",
        sha256=SHA_C,
        size=5,
    )


def make_license_file(**overrides) -> RestoreLicenseFile:
    fields = dict(
        model_id="drunet-color",
        filename="LICENSE",
        url=f"{RELEASE}/licenses--drunet-color--LICENSE",
        sha256=SHA_A,
        size=7,
    )
    return RestoreLicenseFile(**{**fields, **overrides})


def notice_file() -> RestoreLicenseFile:
    return make_license_file(
        filename="NOTICE.txt", url=f"{RELEASE}/licenses--drunet-color--NOTICE.txt", sha256=SHA_C, size=4
    )


def core_bundle(
    *artifacts: RestoreArtifact, license_files: tuple[RestoreLicenseFile, ...] | None = None
) -> RestoreBundle:
    return RestoreBundle(
        name="core",
        artifacts=artifacts or (make_artifact(), fp16_artifact()),
        license_files=(make_license_file(), notice_file()) if license_files is None else license_files,
    )


def install(model_dir: Path, bundle: RestoreBundle, sizes: dict[str, int] | None = None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for artifact in bundle.artifacts:
        size = (sizes or {}).get(artifact.filename, artifact.size)
        (model_dir / artifact.filename).write_bytes(b"x" * size)
    for license_file in bundle.license_files:
        path = model_dir / license_file.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * (sizes or {}).get(license_file.filename, license_file.size))
    manifest = {"bundle": bundle.name, "files": [a.filename for a in bundle.artifacts]}
    (model_dir / f"{bundle.name}.installed.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_license_fields_are_the_ones_the_spec_makes_mandatory() -> None:
    assert LICENSE_FIELDS == (
        "license_spdx",
        "license_url",
        "copyright",
        "attribution",
        "data_lineage",
        "commercial_use",
        "source_url",
        "source_revision",
        "source_sha256",
        "modifications",
    )


@pytest.mark.parametrize("field", LICENSE_FIELDS)
def test_a_spec_without_a_license_field_cannot_be_built(field: str) -> None:
    kwargs = {key: value for key, value in VALID_SPEC.items() if key != field}

    with pytest.raises(TypeError):
        RestoreModelSpec(**kwargs)


@pytest.mark.parametrize("field", [f for f in LICENSE_FIELDS if f != "modifications"])
def test_a_spec_with_a_blank_license_field_fails(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_spec(**{field: "  "})


def test_a_spec_without_license_spdx_fails() -> None:
    with pytest.raises(ValueError, match="license_spdx"):
        make_spec(license_spdx="")


def test_commercial_use_only_accepts_yes_no_or_unclear() -> None:
    for value in ("yes", "no", "unclear"):
        assert make_spec(commercial_use=value).commercial_use == value

    with pytest.raises(ValueError, match="commercial_use"):
        make_spec(commercial_use="maybe")


def test_source_sha256_must_be_64_lowercase_hex() -> None:
    with pytest.raises(ValueError, match="source_sha256"):
        make_spec(source_sha256="ABC")


@pytest.mark.parametrize("modifications", [(), ("",), ("exported", " ")])
def test_modifications_must_list_at_least_one_real_change(modifications) -> None:
    with pytest.raises(ValueError, match="modifications"):
        make_spec(modifications=modifications)


@pytest.mark.parametrize("filename", ["../escape.onnx", "sub/model.onnx", "model.pth", ""])
def test_model_files_must_be_plain_onnx_names(filename: str) -> None:
    with pytest.raises(ValueError, match="filename"):
        make_spec(filename=filename)
    with pytest.raises(ValueError, match="fp16_filename"):
        make_spec(fp16_filename=filename or "x")


def test_runtime_defaults_are_safe_until_p0_gpu_measures_them() -> None:
    spec = make_spec()

    assert spec.vram_factor == 3.0
    assert spec.ort_disable_all is False
    assert dict(spec.tile_by_precision) == {}
    assert spec.fixed_shape is False


def test_tile_by_precision_is_a_read_only_copy() -> None:
    source = {"fp32": 256}
    spec = make_spec(tile_by_precision=source)
    source["fp32"] = 1024

    assert spec.tile_by_precision["fp32"] == 256
    with pytest.raises(TypeError):
        spec.tile_by_precision["fp32"] = 512  # type: ignore[index]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tile_by_precision": {"int8": 256}}, "tile_by_precision"),
        ({"tile_by_precision": {"fp32": 64}}, "tile_by_precision"),
        ({"tile_by_precision": {"fp16": 256}, "fp16_filename": None}, "fp16"),
        ({"vram_factor": 0.0}, "vram_factor"),
        ({"tile_min": 0}, "tile_min"),
        ({"tile_candidates": (0, 256)}, "tile_candidates"),
        ({"overlap": -1}, "overlap"),
        ({"overlap": 128}, "overlap"),
        ({"channels": 2}, "channels"),
        ({"bundle": "Core Pack"}, "bundle"),
        ({"id": "DRUNet color"}, "id"),
    ],
)
def test_runtime_fields_are_validated(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_spec(**overrides)


def test_pack_id_prefixes_the_bundle_name() -> None:
    assert make_spec().pack_id == "restore-core"
    assert core_bundle().pack_id == "restore-core"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"sha256": "123"}, "sha256"),
        ({"size": 0}, "size"),
        ({"url": "http://example.com/drunet-color.onnx"}, "url"),
        ({"filename": "../drunet-color.onnx"}, "filename"),
        ({"precision": "int8"}, "precision"),
    ],
)
def test_artifacts_are_validated(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_artifact(**overrides)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"filename": "README.md"}, "filename"),
        ({"filename": "../LICENSE"}, "filename"),
        ({"model_id": "DRUNet color"}, "model_id"),
        ({"url": "http://example.com/licenses--drunet-color--LICENSE"}, "url"),
        ({"url": f"{RELEASE}/LICENSE"}, "licenses--drunet-color--LICENSE"),
        ({"sha256": "123"}, "sha256"),
        ({"size": 0}, "size"),
    ],
)
def test_license_files_are_validated(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_license_file(**overrides)


@pytest.mark.parametrize("filename", ["LICENSE", "LICENSE.txt", "LICENSE-weights.md", "NOTICE.txt"])
def test_license_files_are_the_license_and_the_notice(filename: str) -> None:
    license_file = make_license_file(filename=filename, url=f"{RELEASE}/licenses--drunet-color--{filename}")

    assert license_file.relative_path == Path("licenses", "drunet-color", filename)


def test_bundles_ship_no_license_files_unless_declared() -> None:
    assert RestoreBundle("core", ()).license_files == ()


def test_bundle_names_are_the_three_download_bundles() -> None:
    assert BUNDLE_NAMES == ("core", "faces", "colorize")
    assert tuple(RESTORE_BUNDLES) == BUNDLE_NAMES
    assert all(RESTORE_BUNDLES[name].name == name for name in BUNDLE_NAMES)


def test_every_catalog_spec_carries_every_license_field() -> None:
    spec_fields = {f.name for f in dataclasses.fields(RestoreModelSpec)}

    assert set(LICENSE_FIELDS) <= spec_fields
    for spec in RESTORE_MODELS.values():
        assert all(getattr(spec, field) for field in LICENSE_FIELDS), spec.id


def test_catalog_keys_are_the_spec_ids() -> None:
    assert all(model_id == spec.id for model_id, spec in RESTORE_MODELS.items())


def test_the_shipped_catalog_is_consistent() -> None:
    assert catalog_problems(RESTORE_MODELS, RESTORE_BUNDLES) == []


def test_license_gated_packs_are_exactly_the_non_commercial_ones() -> None:
    assert LICENSE_GATED_PACKS == gated_packs(RESTORE_MODELS)


def test_gated_packs_only_counts_commercial_use_no() -> None:
    models = {
        "a": make_spec(id="a", bundle="faces-nc", commercial_use="no"),
        "b": make_spec(id="b", bundle="faces", commercial_use="unclear"),
        "c": make_spec(id="c", bundle="core", commercial_use="yes"),
    }

    assert gated_packs(models) == frozenset({"restore-faces-nc"})


def test_a_consistent_catalog_has_no_problems() -> None:
    models = {"drunet-color": make_spec()}

    assert catalog_problems(models, {"core": core_bundle()}) == []


def test_catalog_flags_an_artifact_of_an_unknown_model() -> None:
    bundle = core_bundle(make_artifact(model_id="ghost"))

    assert any("ghost" in p for p in catalog_problems({}, {"core": bundle}))


def test_catalog_flags_an_artifact_in_the_wrong_bundle() -> None:
    models = {"drunet-color": make_spec(bundle="faces")}

    problems = catalog_problems(models, {"core": core_bundle(), "faces": RestoreBundle("faces", ())})

    assert any("bundle" in p for p in problems)


def test_catalog_flags_a_model_file_missing_from_its_bundle() -> None:
    models = {"drunet-color": make_spec()}

    problems = catalog_problems(models, {"core": core_bundle(make_artifact())})

    assert any("drunet-color-fp16.onnx" in p for p in problems)


def test_catalog_flags_an_artifact_whose_file_is_not_the_spec_file() -> None:
    models = {"drunet-color": make_spec(fp16_filename=None)}

    problems = catalog_problems(models, {"core": core_bundle(make_artifact(), fp16_artifact())})

    assert any("fp16" in p for p in problems)


def test_catalog_flags_duplicated_file_names() -> None:
    models = {"drunet-color": make_spec(fp16_filename=None)}

    problems = catalog_problems(models, {"core": core_bundle(make_artifact(), make_artifact())})

    assert any("duplicated" in p for p in problems)


def test_catalog_flags_a_bundle_key_that_is_not_its_name() -> None:
    assert any("key" in p for p in catalog_problems({}, {"faces": RestoreBundle("core", ())}))


def test_non_commercial_models_never_enter_a_default_bundle() -> None:
    models = {"drunet-color": make_spec(commercial_use="no")}

    problems = catalog_problems(models, {"core": core_bundle()})

    assert any("commercial" in p for p in problems)


def test_catalog_flags_a_license_file_of_an_unknown_model() -> None:
    ghost = make_license_file(model_id="ghost", url=f"{RELEASE}/licenses--ghost--LICENSE")
    bundle = core_bundle(license_files=(make_license_file(), notice_file(), ghost))

    assert any("ghost" in p for p in catalog_problems({"drunet-color": make_spec()}, {"core": bundle}))


def test_catalog_flags_a_license_file_of_a_model_in_another_bundle() -> None:
    gfpgan = make_spec(id="gfpgan", bundle="faces", filename="gfpgan.onnx", fp16_filename=None)
    stray = make_license_file(model_id="gfpgan", url=f"{RELEASE}/licenses--gfpgan--LICENSE")
    bundles = {
        "core": core_bundle(license_files=(make_license_file(), notice_file(), stray)),
        "faces": RestoreBundle("faces", (make_artifact(model_id="gfpgan", filename="gfpgan.onnx"),)),
    }

    problems = catalog_problems({"drunet-color": make_spec(), "gfpgan": gfpgan}, bundles)

    assert any("gfpgan" in p and p.startswith("core") for p in problems)


@pytest.mark.parametrize(
    "license_files, missing",
    [((make_license_file(),), "NOTICE.txt"), ((notice_file(),), "LICENSE")],
)
def test_catalog_flags_a_model_without_its_license_or_notice(license_files, missing: str) -> None:
    bundle = core_bundle(license_files=license_files)

    problems = catalog_problems({"drunet-color": make_spec()}, {"core": bundle})

    assert any(missing in p for p in problems)


def test_catalog_flags_duplicated_license_files() -> None:
    bundle = core_bundle(license_files=(make_license_file(), make_license_file(), notice_file()))

    problems = catalog_problems({"drunet-color": make_spec()}, {"core": bundle})

    assert any("duplicated" in p for p in problems)


def test_bundle_installed_needs_the_manifest(tmp_path: Path) -> None:
    bundle = core_bundle()
    install(tmp_path, bundle)
    (tmp_path / "core.installed.json").unlink()

    assert bundle_installed(tmp_path, bundle) is False


def test_bundle_installed_needs_every_file(tmp_path: Path) -> None:
    bundle = core_bundle()
    install(tmp_path, bundle)
    (tmp_path / "drunet-color-fp16.onnx").unlink()

    assert bundle_installed(tmp_path, bundle) is False


def test_bundle_installed_needs_every_declared_size(tmp_path: Path) -> None:
    bundle = core_bundle()
    install(tmp_path, bundle, sizes={"drunet-color.onnx": 9})

    assert bundle_installed(tmp_path, bundle) is False


def test_bundle_installed_needs_every_license_file_at_its_size(tmp_path: Path) -> None:
    bundle = core_bundle()
    install(tmp_path, bundle)
    (tmp_path / "licenses" / "drunet-color" / "NOTICE.txt").unlink()

    assert bundle_installed(tmp_path, bundle) is False

    install(tmp_path, bundle, sizes={"NOTICE.txt": 3})

    assert bundle_installed(tmp_path, bundle) is False


def test_bundle_installed_with_manifest_and_every_file_at_its_size(tmp_path: Path) -> None:
    bundle = core_bundle()
    install(tmp_path, bundle)

    assert bundle_installed(tmp_path, bundle) is True
    assert installed_manifest(tmp_path, bundle) == tmp_path / "core.installed.json"


def test_an_unpublished_bundle_is_never_installed(tmp_path: Path) -> None:
    bundle = RestoreBundle("core", ())
    install(tmp_path, bundle)

    assert bundle_installed(tmp_path, bundle) is False
    assert installed_manifest(tmp_path, bundle) is None


def test_the_shipped_bundles_are_not_installed_in_an_empty_folder(tmp_path: Path) -> None:
    assert not any(bundle_installed(tmp_path, bundle) for bundle in RESTORE_BUNDLES.values())


def test_model_path_by_precision(tmp_path: Path) -> None:
    models = {"drunet-color": make_spec()}

    assert model_path(tmp_path, "drunet-color", "fp32", models) == tmp_path / "drunet-color.onnx"
    assert model_path(tmp_path, "drunet-color", "fp16", models) == tmp_path / "drunet-color-fp16.onnx"


def test_model_path_is_none_without_an_fp16_variant(tmp_path: Path) -> None:
    models = {"drunet-color": make_spec(fp16_filename=None)}

    assert model_path(tmp_path, "drunet-color", "fp16", models) is None


def test_model_path_rejects_unknown_precisions_and_models(tmp_path: Path) -> None:
    models = {"drunet-color": make_spec()}

    with pytest.raises(ValueError, match="precision"):
        model_path(tmp_path, "drunet-color", "int8", models)
    with pytest.raises(KeyError):
        model_path(tmp_path, "ghost", "fp32", models)


def test_model_path_defaults_to_the_shipped_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(rm.RESTORE_MODELS, "drunet-color", make_spec())

    assert model_path(tmp_path, "drunet-color", "fp32") == tmp_path / "drunet-color.onnx"


# ---------------------------------------------------------------------------
# Anti-deriva: scripts/download-restore.ps1 baja exactamente lo que dice el catalogo
# ---------------------------------------------------------------------------


def test_the_script_offers_exactly_the_catalog_bundles() -> None:
    assert validate_set(script_text()) == BUNDLE_NAMES


def test_every_sha256_in_the_script_belongs_to_the_catalog() -> None:
    catalog = {a.sha256 for b in RESTORE_BUNDLES.values() for a in b.artifacts}
    catalog |= {f.sha256 for b in RESTORE_BUNDLES.values() for f in b.license_files}

    assert declared_sha256(script_text()) - catalog == set()


@requires_powershell
def test_the_script_table_is_the_catalog() -> None:
    table = evaluate_catalog(catalog_block(script_text()))

    assert comparable(table) == expected_catalog(RESTORE_BUNDLES, RESTORE_MODELS)
    assert all(row.get("Label") for tables in table.values() for row in tables["Files"])


@requires_powershell
def test_the_anti_drift_reads_a_published_table_as_powershell_does() -> None:
    # Un solo archivo por bundle ejercita el caso en que PowerShell desarma un @() de un elemento.
    models = {"drunet-color": make_spec(fp16_filename=None)}
    published = {
        "core": core_bundle(make_artifact()),
        "faces": RestoreBundle("faces", ()),
        "colorize": RestoreBundle("colorize", ()),
    }
    assert catalog_problems(models, published) == []

    table = evaluate_catalog(render_catalog(published, models))

    assert comparable(table) == expected_catalog(published, models)
    assert comparable(table) != expected_catalog(RESTORE_BUNDLES, RESTORE_MODELS)
