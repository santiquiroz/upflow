"""THIRD_PARTY_NOTICES.md tiene que cubrir todo lo de terceros que Upflow redistribuye.

Se cruzan cuatro fuentes con las entradas del archivo: lo que `package-release.ps1`
mete en el zip y en el instalador, los modelos de `RESTORE_BUNDLES`, los archivos
con linea "Adapted from" (codigo portado, sala limpia de la spec §6.1) y las
fuentes tipograficas bundleadas. Cada verificador es una funcion pura que devuelve
la lista de problemas, para poder probarla con textos sinteticos ademas del repo.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from app.services.restore_models import (
    RESTORE_BUNDLES,
    RESTORE_MODELS,
    RestoreArtifact,
    RestoreBundle,
    RestoreModelSpec,
)

ROOT = Path(__file__).resolve().parent.parent
NOTICES_PATH = ROOT / "THIRD_PARTY_NOTICES.md"
PACKAGE_RELEASE = ROOT / "scripts" / "package-release.ps1"
FRONTEND_PACKAGE = ROOT / "frontend" / "package.json"

REQUIRED_FIELDS = ("License", "Copyright", "Source")
SCANNED_DIRS = ("app", "scripts", "tools", "frontend/src")
SCANNED_SUFFIXES = frozenset({".py", ".ps1", ".ts", ".tsx", ".js", ".mjs"})
FONT_SUFFIXES = frozenset({".ttf", ".otf", ".woff", ".woff2"})
FRONTEND_BUNDLE = "frontend/dist/"
EMBEDDED_PYTHON = "python/"

_ENTRY_HEADING = re.compile(r"^### (?P<title>.+?)\s*$")
_FIELD = re.compile(r"^- (?P<key>[A-Za-z][A-Za-z ()]*?): (?P<value>.+?)\s*$")
_STAGED_IN_APP = re.compile(r"Join-Path \$(?:installerAppDir|Destination) '(?P<path>[^']+)'")
_INCLUDE_FILES = re.compile(r"\$includeFiles = @\((?P<items>[^)]*)\)")
_QUOTED = re.compile(r"'([^']+)'")
PROVENANCE_FORMAT = "Adapted from <repo>@<sha> (<license>, © <author>)"
ADAPTED_FROM = re.compile(
    r"Adapted from (?P<repo>[\w.-]+(?:/[\w.-]+)+)@(?P<rev>[0-9a-f]{7,40}) "
    r"\((?P<license>[^,()]+), © (?P<author>[^()]+)\)"
)


def _clean(value: str) -> str:
    return value.replace("`", "").strip()


def _entry_title(line: str, current: str) -> str:
    heading = _ENTRY_HEADING.match(line)
    if heading:
        return heading["title"]
    return "" if line.startswith("#") else current


def _with_field(fields: Mapping[str, tuple[str, ...]], key: str, value: str) -> dict[str, tuple[str, ...]]:
    return {**fields, key: fields.get(key, ()) + (value,)}


def notice_entries(text: str) -> dict[str, dict[str, tuple[str, ...]]]:
    entries: dict[str, dict[str, tuple[str, ...]]] = {}
    title = ""
    for line in text.splitlines():
        title = _entry_title(line, title)
        field = _FIELD.match(line)
        if _ENTRY_HEADING.match(line):
            entries[title] = {}
        elif title and field:
            entries[title] = _with_field(entries[title], field["key"], _clean(field["value"]))
    return entries


def entries_with(entries: Mapping[str, Mapping[str, tuple[str, ...]]], key: str, value: str) -> list[str]:
    return [title for title, fields in entries.items() if value in fields.get(key, ())]


def missing_field_problems(entries: Mapping[str, Mapping[str, tuple[str, ...]]]) -> list[str]:
    return [
        f"{title!r} has no {name!r}"
        for title, fields in entries.items()
        for name in REQUIRED_FIELDS
        if not any(value.strip() for value in fields.get(name, ()))
    ]


def _release_path(ps1_path: str) -> str:
    return ps1_path.replace("\\", "/").strip("/") + "/"


def staged_release_components(ps1_text: str) -> set[str]:
    staged = {_release_path(match["path"]) for match in _STAGED_IN_APP.finditer(ps1_text)}
    if "$pythonEmbedUrl" in ps1_text:
        staged.add(EMBEDDED_PYTHON)
    return staged


def release_include_files(ps1_text: str) -> tuple[str, ...]:
    match = _INCLUDE_FILES.search(ps1_text)
    return tuple(_QUOTED.findall(match["items"])) if match else ()


def bundled_component_problems(entries, ps1_text: str) -> list[str]:
    return [
        f"package-release.ps1 ships {path!r} but no entry has 'Component: {path}'"
        for path in sorted(staged_release_components(ps1_text))
        if not entries_with(entries, "Component", path)
    ]


def _bundle_model_ids(bundles: Mapping[str, RestoreBundle]) -> list[str]:
    return sorted({artifact.model_id for bundle in bundles.values() for artifact in bundle.artifacts})


def _restore_model_problem(entries, model_id: str, models: Mapping[str, RestoreModelSpec]) -> list[str]:
    titles = entries_with(entries, "Restore model", model_id)
    if not titles:
        return [f"restore model {model_id!r} has no entry with 'Restore model: {model_id}'"]
    spdx = models[model_id].license_spdx if model_id in models else None
    if spdx is not None and all(spdx not in entries[title].get("License", ()) for title in titles):
        return [f"restore model {model_id!r}: the entry does not say 'License: {spdx}'"]
    return []


def restore_model_problems(entries, bundles: Mapping[str, RestoreBundle], models) -> list[str]:
    return [
        problem
        for model_id in _bundle_model_ids(bundles)
        for problem in _restore_model_problem(entries, model_id, models)
    ]


def _scanned_files(root: Path) -> Iterable[Path]:
    for folder in SCANNED_DIRS:
        base = root / folder
        if base.is_dir():
            yield from (
                path
                for path in base.rglob("*")
                if path.suffix in SCANNED_SUFFIXES and "node_modules" not in path.parts
            )


def adapted_from_lines(root: Path) -> list[tuple[str, str]]:
    found = []
    for path in sorted(_scanned_files(root)):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Adapted from" in line:
                found.append((path.relative_to(root).as_posix(), line.strip()))
    return found


def _source_matches(entries, title: str, repo: str) -> bool:
    return any(repo in source for source in entries[title].get("Source", ()))


def _adapted_from_problem(entries, relpath: str, line: str) -> list[str]:
    match = ADAPTED_FROM.search(line)
    if match is None:
        return [f"{relpath}: malformed provenance line {line!r} (expected {PROVENANCE_FORMAT!r})"]
    titles = entries_with(entries, "Ported into", relpath)
    if not titles:
        return [f"{relpath}: ported code has no entry with 'Ported into: {relpath}'"]
    if not any(_source_matches(entries, title, match["repo"]) for title in titles):
        return [f"{relpath}: no entry for it cites {match['repo']!r} as its Source"]
    if not any(match["license"].strip() in entries[title].get("License", ()) for title in titles):
        return [f"{relpath}: no entry for it says 'License: {match['license'].strip()}'"]
    return []


def ported_code_problems(entries, lines: Iterable[tuple[str, str]]) -> list[str]:
    return [problem for relpath, line in lines for problem in _adapted_from_problem(entries, relpath, line)]


def stale_path_problems(entries, root: Path) -> list[str]:
    ported = [(title, path) for title, fields in entries.items() for path in fields.get("Ported into", ())]
    return [f"{title!r}: 'Ported into: {path}' does not exist" for title, path in ported if not (root / path).is_file()]


def bundled_fonts(root: Path) -> list[str]:
    assets = root / "app" / "assets"
    if not assets.is_dir():
        return []
    return sorted(path.relative_to(root).as_posix() for path in assets.rglob("*") if path.suffix in FONT_SUFFIXES)


def font_problems(entries, fonts: Iterable[str]) -> list[str]:
    return [
        f"font {font!r} has no entry with 'Component: {font}'"
        for font in fonts
        if not entries_with(entries, "Component", font)
    ]


def frontend_package_problems(entries, dependencies: Iterable[str]) -> list[str]:
    in_bundle = set(entries_with(entries, "Component", FRONTEND_BUNDLE))
    return [
        f"frontend dependency {name!r} has no '{FRONTEND_BUNDLE}' entry with 'Package: {name}'"
        for name in sorted(dependencies)
        if not in_bundle.intersection(entries_with(entries, "Package", name))
    ]


def _repo_entries() -> dict[str, dict[str, tuple[str, ...]]]:
    return notice_entries(NOTICES_PATH.read_text(encoding="utf-8"))


def _ps1_text() -> str:
    return PACKAGE_RELEASE.read_text(encoding="utf-8")


def _entry(title: str, **fields: str) -> str:
    body = "\n".join(f"- {key.replace('_', ' ')}: {value}" for key, value in fields.items())
    return f"### {title}\n{body}\n"


def _complete(title: str, **fields: str) -> str:
    return _entry(title, License="MIT", Copyright="(c) Someone", Source="https://example.org/x", **fields)


def _spec(model_id: str, license_spdx: str = "MIT") -> RestoreModelSpec:
    return RestoreModelSpec(
        id=model_id,
        name="Example",
        bundle="core",
        filename=f"{model_id}.onnx",
        license_spdx=license_spdx,
        license_url="https://example.org/LICENSE",
        copyright="(c) Someone",
        attribution="Someone et al.",
        data_lineage="D1a",
        commercial_use="yes",
        source_url="https://example.org/repo",
        source_revision="abc1234",
        source_sha256="0" * 64,
        modifications=("exported to ONNX",),
        tile_min=64,
    )


def _bundle_of(model_id: str) -> dict[str, RestoreBundle]:
    artifact = RestoreArtifact(
        model_id=model_id,
        precision="fp32",
        filename=f"{model_id}.onnx",
        url="https://example.org/a.onnx",
        sha256="0" * 64,
        size=1,
    )
    return {"core": RestoreBundle("core", (artifact,))}


# ---------------------------------------------------------------------------
# El repo real
# ---------------------------------------------------------------------------


def test_the_notices_file_exists_and_has_entries() -> None:
    assert NOTICES_PATH.is_file()
    assert _repo_entries()


def test_every_entry_names_its_license_copyright_and_source() -> None:
    assert missing_field_problems(_repo_entries()) == []


def test_package_release_ships_the_notices_file_in_every_artifact() -> None:
    assert "THIRD_PARTY_NOTICES.md" in release_include_files(_ps1_text())


def test_package_release_is_still_parsed_as_bundling_third_party_parts() -> None:
    # Si alguien reescribe el script y el regex deja de ver lo que stagea, el
    # cruce de abajo pasaria en vacio.
    assert {"vendor/apollo/", "vendor/realesrgan-onnx/", "vendor/wheels/", FRONTEND_BUNDLE, EMBEDDED_PYTHON} <= (
        staged_release_components(_ps1_text())
    )


def test_every_component_package_release_bundles_has_an_entry() -> None:
    assert bundled_component_problems(_repo_entries(), _ps1_text()) == []


def test_every_restore_bundle_model_has_an_entry_with_its_license() -> None:
    assert restore_model_problems(_repo_entries(), RESTORE_BUNDLES, RESTORE_MODELS) == []


def test_every_file_with_ported_code_has_an_entry() -> None:
    assert ported_code_problems(_repo_entries(), adapted_from_lines(ROOT)) == []


def test_ported_code_entries_point_to_files_that_exist() -> None:
    assert stale_path_problems(_repo_entries(), ROOT) == []


def test_every_bundled_font_has_an_entry() -> None:
    assert font_problems(_repo_entries(), bundled_fonts(ROOT)) == []


def test_every_frontend_runtime_dependency_has_an_entry() -> None:
    dependencies = json.loads(FRONTEND_PACKAGE.read_text(encoding="utf-8")).get("dependencies", {})
    assert frontend_package_problems(_repo_entries(), dependencies) == []


def test_apollo_entry_carries_the_share_alike_license_of_its_source() -> None:
    entries = _repo_entries()
    (title,) = entries_with(entries, "Component", "vendor/apollo/")
    assert entries[title]["License"] == ("CC-BY-SA-4.0",)


def test_real_esrgan_entry_reproduces_the_bsd_notice() -> None:
    text = NOTICES_PATH.read_text(encoding="utf-8")
    entries = notice_entries(text)
    (title,) = entries_with(entries, "Component", "vendor/realesrgan-onnx/")
    assert entries[title]["License"] == ("BSD-3-Clause",)
    # La clausula 2 de BSD-3 exige reproducir el aviso completo en la redistribucion binaria.
    assert "Copyright (c) 2021, Xintao Wang" in text
    assert "Redistributions in binary form must reproduce the above copyright notice" in text


# ---------------------------------------------------------------------------
# Los verificadores detectan lo que tienen que detectar
# ---------------------------------------------------------------------------


def test_notice_entries_collects_repeated_fields_per_entry() -> None:
    text = "# Top\n\n" + _complete("A", Component="`vendor/a/`", Ported_into="x.py") + "- Ported into: y.py\n"
    entries = notice_entries(text)
    assert entries["A"]["Component"] == ("vendor/a/",)
    assert entries["A"]["Ported into"] == ("x.py", "y.py")


def test_fields_outside_an_entry_are_ignored() -> None:
    text = "## Section\n- License: MIT\n" + _complete("A")
    assert set(notice_entries(text)) == {"A"}
    assert notice_entries(text)["A"]["License"] == ("MIT",)


def test_an_entry_without_copyright_is_reported() -> None:
    entries = notice_entries(_entry("A", License="MIT", Source="https://example.org"))
    assert missing_field_problems(entries) == ["'A' has no 'Copyright'"]


def test_a_component_staged_by_the_script_without_entry_is_reported() -> None:
    ps1 = "$dst = Join-Path $installerAppDir 'vendor\\newthing'\n$x = Join-Path $installerAppDir 'vendor\\apollo'"
    entries = notice_entries(_complete("Apollo", Component="vendor/apollo/"))
    assert bundled_component_problems(entries, ps1) == [
        "package-release.ps1 ships 'vendor/newthing/' but no entry has 'Component: vendor/newthing/'"
    ]


def test_the_embedded_python_counts_as_a_bundled_component() -> None:
    assert staged_release_components("Invoke-WebRequest -Uri $pythonEmbedUrl") == {EMBEDDED_PYTHON}


def test_include_files_are_read_from_the_allowlist() -> None:
    ps1 = "$includeFiles = @('pyproject.toml', 'LICENSE', 'THIRD_PARTY_NOTICES.md')"
    assert release_include_files(ps1) == ("pyproject.toml", "LICENSE", "THIRD_PARTY_NOTICES.md")


def test_a_restore_model_without_entry_is_reported() -> None:
    models = {"drunet": _spec("drunet")}
    assert restore_model_problems({}, _bundle_of("drunet"), models) == [
        "restore model 'drunet' has no entry with 'Restore model: drunet'"
    ]


def test_a_restore_model_entry_with_another_license_is_reported() -> None:
    models = {"drunet": _spec("drunet", "Apache-2.0")}
    entries = notice_entries(_complete("DRUNet", Restore_model="drunet"))
    assert restore_model_problems(entries, _bundle_of("drunet"), models) == [
        "restore model 'drunet': the entry does not say 'License: Apache-2.0'"
    ]


def test_a_restore_model_with_matching_entry_passes() -> None:
    models = {"drunet": _spec("drunet")}
    entries = notice_entries(_complete("DRUNet", Restore_model="drunet"))
    assert restore_model_problems(entries, _bundle_of("drunet"), models) == []


@pytest.fixture
def ported_tree(tmp_path: Path) -> Path:
    module = tmp_path / "app" / "services" / "restore"
    module.mkdir(parents=True)
    (module / "priors.py").write_text(
        "# Adapted from yakhyo/retinaface@0123abc (MIT, © Yakhyokhuja Valikhujaev)\nPRIORS = 1\n",
        encoding="utf-8",
    )
    return tmp_path


def test_adapted_from_lines_are_found_in_shipped_folders(ported_tree: Path) -> None:
    assert adapted_from_lines(ported_tree) == [
        ("app/services/restore/priors.py", "# Adapted from yakhyo/retinaface@0123abc (MIT, © Yakhyokhuja Valikhujaev)")
    ]


def test_ported_code_without_entry_is_reported(ported_tree: Path) -> None:
    assert ported_code_problems({}, adapted_from_lines(ported_tree)) == [
        "app/services/restore/priors.py: ported code has no entry with 'Ported into: app/services/restore/priors.py'"
    ]


def test_ported_code_whose_entry_cites_another_repo_is_reported(ported_tree: Path) -> None:
    entries = notice_entries(_complete("RetinaFace", Ported_into="app/services/restore/priors.py"))
    assert ported_code_problems(entries, adapted_from_lines(ported_tree)) == [
        "app/services/restore/priors.py: no entry for it cites 'yakhyo/retinaface' as its Source"
    ]


def test_ported_code_with_a_matching_entry_passes(ported_tree: Path) -> None:
    entries = notice_entries(
        _entry(
            "RetinaFace priors",
            License="MIT",
            Copyright="(c) 2024 Yakhyokhuja Valikhujaev",
            Source="https://github.com/yakhyo/retinaface",
            Ported_into="app/services/restore/priors.py",
        )
    )
    assert ported_code_problems(entries, adapted_from_lines(ported_tree)) == []
    assert stale_path_problems(entries, ported_tree) == []


def test_a_malformed_provenance_line_is_reported() -> None:
    (problem,) = ported_code_problems({}, [("app/x.py", "# Adapted from somewhere on the internet")])
    assert problem.startswith("app/x.py: malformed provenance line")


def test_an_entry_pointing_to_a_missing_file_is_reported(tmp_path: Path) -> None:
    entries = notice_entries(_complete("Gone", Ported_into="app/gone.py"))
    assert stale_path_problems(entries, tmp_path) == ["'Gone': 'Ported into: app/gone.py' does not exist"]


def test_a_bundled_font_without_entry_is_reported(tmp_path: Path) -> None:
    fonts = tmp_path / "app" / "assets" / "fonts"
    fonts.mkdir(parents=True)
    (fonts / "SourceCodePro-Regular.ttf").write_bytes(b"\0")
    assert font_problems({}, bundled_fonts(tmp_path)) == [
        "font 'app/assets/fonts/SourceCodePro-Regular.ttf' has no entry with "
        "'Component: app/assets/fonts/SourceCodePro-Regular.ttf'"
    ]


def test_a_frontend_dependency_without_entry_is_reported() -> None:
    entries = notice_entries(_complete("React", Component=FRONTEND_BUNDLE, Package="react"))
    assert frontend_package_problems(entries, ["react", "left-pad"]) == [
        "frontend dependency 'left-pad' has no 'frontend/dist/' entry with 'Package: left-pad'"
    ]
