from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, Settings
from app.services.restore_models import (
    RESTORE_BUNDLES,
    RESTORE_MODELS,
    VENDORED_MODELS,
    RestoreBundle,
    RestoreModelSpec,
    VendoredModel,
    bundle_installed,
    is_license_file_name,
)

NOTICES_PATH = PROJECT_ROOT / "THIRD_PARTY_NOTICES.md"
HEADING = re.compile(r"^(?P<level>#{2,3}) (?P<title>.+?)\s*$", re.MULTILINE)
FIELD = re.compile(r"^- (?P<key>[A-Za-z][A-Za-z ()]*?): (?P<value>.+?)\s*$", re.MULTILINE)
FENCE = re.compile(r"^```text\n(?P<body>.*?)^```", re.MULTILINE | re.DOTALL)
SECTION_LEVEL = "##"


@dataclass(frozen=True, slots=True)
class LicenseCatalog:
    models: Mapping[str, RestoreModelSpec] = field(default_factory=lambda: RESTORE_MODELS)
    bundles: Mapping[str, RestoreBundle] = field(default_factory=lambda: RESTORE_BUNDLES)
    vendored: Mapping[str, VendoredModel] = field(default_factory=lambda: VENDORED_MODELS)


@dataclass(frozen=True, slots=True)
class NoticeBlock:
    section: str
    title: str
    body: str


def licenses_view(
    settings: Settings, catalog: LicenseCatalog | None = None, notices_path: Path = NOTICES_PATH
) -> dict[str, Any]:
    chosen = catalog or LicenseCatalog()
    packs = installed_bundle_packs(settings.restore_model_dir_path, chosen) + vendored_packs(settings, chosen)
    notices = notices_path.read_text(encoding="utf-8") if notices_path.is_file() else ""
    return {"packs": packs, "thirdParty": notice_entries(notices)}


def installed_bundle_packs(model_dir: Path, catalog: LicenseCatalog) -> list[dict[str, Any]]:
    return [
        {"pack": bundle.pack_id, "models": _bundle_models(model_dir, bundle, catalog.models)}
        for bundle in catalog.bundles.values()
        if bundle_installed(model_dir, bundle)
    ]


def _bundle_models(
    model_dir: Path, bundle: RestoreBundle, models: Mapping[str, RestoreModelSpec]
) -> list[dict[str, Any]]:
    return [
        model_entry(spec, [model_dir / f.relative_path for f in bundle.license_files if f.model_id == spec.id])
        for spec in models.values()
        if spec.bundle == bundle.name
    ]


def vendored_packs(settings: Any, catalog: LicenseCatalog) -> list[dict[str, Any]]:
    packs = []
    for vendored in catalog.vendored.values():
        path = vendored.path_of(settings)
        if path.is_file():
            packs.append({"pack": vendored.pack, "models": [model_entry(vendored.spec, sibling_licenses(path))]})
    return packs


def sibling_licenses(model_file: Path) -> list[Path]:
    return sorted(path for path in model_file.parent.iterdir() if path.is_file() and is_license_file_name(path.name))


def model_entry(spec: RestoreModelSpec, license_files: Sequence[Path]) -> dict[str, Any]:
    return {
        "id": spec.id,
        "name": spec.name,
        "licenseSpdx": spec.license_spdx,
        "licenseUrl": spec.license_url,
        "copyright": spec.copyright,
        "attribution": spec.attribution,
        "dataLineage": spec.data_lineage,
        "commercialUse": spec.commercial_use,
        "sourceUrl": spec.source_url,
        "sourceRevision": spec.source_revision,
        "modifications": list(spec.modifications),
        "files": [license_file_entry(path) for path in license_files if path.is_file()],
    }


def license_file_entry(path: Path) -> dict[str, str]:
    return {"name": path.name, "text": path.read_text(encoding="utf-8", errors="replace")}


def notice_entries(text: str) -> list[dict[str, Any]]:
    return [notice_entry(block) for block in notice_blocks(text)]


def notice_blocks(text: str) -> list[NoticeBlock]:
    headings = list(HEADING.finditer(_without_fences(text)))
    blocks = []
    section = ""
    for position, heading in enumerate(headings):
        if heading["level"] == SECTION_LEVEL:
            section = heading["title"]
            continue
        end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
        blocks.append(NoticeBlock(section, heading["title"], text[heading.end() : end]))
    return blocks


def notice_entry(block: NoticeBlock) -> dict[str, Any]:
    fence = FENCE.search(block.body)
    return {
        "title": block.title,
        "section": block.section,
        "fields": notice_fields(_without_fences(block.body)),
        "licenseText": None if fence is None else fence["body"].rstrip("\n"),
    }


def notice_fields(body: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for match in FIELD.finditer(body):
        fields = {**fields, match["key"]: [*fields.get(match["key"], []), match["value"].replace("`", "")]}
    return fields


def _without_fences(text: str) -> str:
    # Blanquea los bloques de licencia sin cambiar los offsets: un "## " o un "- X: y" dentro
    # del texto de una licencia no es un encabezado ni un campo.
    return FENCE.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), text)
