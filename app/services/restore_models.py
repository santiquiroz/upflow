"""Catalogo de modelos de restauracion de fotos y de los bundles que los bajan.

Modulo de datos puro (sin imports de app.*): lo consumen config, los motores,
el provisionador de packs y el anti-deriva de scripts/download-restore.ps1.

Los campos de licencia son obligatorios (spec §3.7) y usan los mismos nombres
que `ModelLicense` del repo de export (port-restore-onnx/common/manifest.py),
para que P1-15 copie models.lock.json sin traducir nada.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

CommercialUse = Literal["yes", "no", "unclear"]
Precision = Literal["fp32", "fp16"]

COMMERCIAL_USE: tuple[str, ...] = ("yes", "no", "unclear")
PRECISIONS: tuple[str, ...] = ("fp32", "fp16")
PACK_PREFIX = "restore-"
MANIFEST_SUFFIX = ".installed.json"
DEFAULT_VRAM_FACTOR = 3.0
DEFAULT_OVERLAP = 32
LICENSES_DIRNAME = "licenses"
NOTICE_NAME = "NOTICE.txt"

LICENSE_FIELDS: tuple[str, ...] = (
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

LINEAGE_KEYS: tuple[str, ...] = ("D1a", "D1b", "D1c")
NO_DATA_DEBT = "none (permissive training data)"
LINEAGE_SEPARATOR = " + "

# Tipo de clausula de la tabla de linaje de §3.7 -> decision que la cubre; None = sin deuda.
CLAUSE_LINEAGE: Mapping[str, str | None] = MappingProxyType(
    {
        "permissive": None,
        "research-only": "D1a",
        "no-license": "D1a",
        # Licencia declarada sin decir si cubre las imagenes: D1a hasta que el autor lo confirme.
        "unclear-scope": "D1a",
        "share-alike": "D1b",
        "no-derivatives": "D1b",
        "undeclared": "D1c",
    }
)

_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SAFE_ONNX_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.onnx")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LICENSE_NAME = re.compile(r"LICENSE[A-Za-z0-9._-]*")


def is_safe_id(value: str) -> bool:
    return bool(_SAFE_ID.fullmatch(value))


def is_safe_onnx_file(name: str) -> bool:
    return bool(_SAFE_ONNX_FILE.fullmatch(name)) and ".." not in name


def is_sha256(value: str) -> bool:
    return bool(_SHA256.fullmatch(value))


def is_license_name(name: str) -> bool:
    return bool(_LICENSE_NAME.fullmatch(name)) and ".." not in name


def is_license_file_name(name: str) -> bool:
    return name == NOTICE_NAME or is_license_name(name)


def lineage_parts(value: str) -> list[str]:
    return [part.strip() for part in value.split("+")]


def is_data_lineage(value: str) -> bool:
    if value == NO_DATA_DEBT:
        return True
    return all(part in LINEAGE_KEYS for part in lineage_parts(value))


def pack_id(bundle: str) -> str:
    return f"{PACK_PREFIX}{bundle}"


def release_license_asset(model_id: str, filename: str) -> str:
    # Los assets de GitHub no tienen carpetas: port-restore-onnx aplana licenses/<modelo>/<archivo>.
    return f"{LICENSES_DIRNAME}--{model_id}--{filename}"


def _raise_on(owner: str, problems: list[str]) -> None:
    if problems:
        raise ValueError(f"{owner}: " + "; ".join(problems))


@dataclass(frozen=True, slots=True)
class RestoreModelSpec:
    id: str
    # Nombre propio del modelo: se muestra tal cual, no se traduce.
    name: str
    # Bundle de descarga ("core", "faces", "colorize"); los packs no comerciales
    # usan un nombre propio ("faces-nc") que nunca entra en RESTORE_BUNDLES.
    bundle: str
    filename: str
    license_spdx: str
    license_url: str
    copyright: str
    attribution: str
    # Clave de la tabla de linaje de datos de §3.7 (D1a, D1b, D1c o combinaciones).
    data_lineage: str
    commercial_use: CommercialUse
    source_url: str
    source_revision: str
    source_sha256: str
    modifications: tuple[str, ...]
    tile_min: int
    # Solo con validacion fp16 del repo de export Y de P0-GPU; si no, None.
    fp16_filename: str | None = field(default=None, kw_only=True)
    tile_candidates: tuple[int, ...] = field(default=(), kw_only=True)
    fixed_shape: bool = field(default=False, kw_only=True)
    overlap: int = field(default=DEFAULT_OVERLAP, kw_only=True)
    channels: int = field(default=3, kw_only=True)
    # Techo del tile por precision medido en P0-GPU; vacio = sin techo todavia.
    tile_by_precision: Mapping[str, int] = field(default_factory=dict, kw_only=True)
    # Activaciones y arena de DML ocupan 2-4x el archivo; P0-GPU lo mide por modelo.
    vram_factor: float = field(default=DEFAULT_VRAM_FACTOR, kw_only=True)
    ort_disable_all: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_by_precision", MappingProxyType(dict(self.tile_by_precision)))
        _raise_on(repr(self.id), license_problems(self) + identity_problems(self) + runtime_problems(self))

    @property
    def pack_id(self) -> str:
        return pack_id(self.bundle)

    def files_by_precision(self) -> dict[str, str]:
        files = {"fp32": self.filename}
        if self.fp16_filename is not None:
            files["fp16"] = self.fp16_filename
        return files


def _blank_license_fields(spec: RestoreModelSpec) -> list[str]:
    text_fields = (name for name in LICENSE_FIELDS if name != "modifications")
    return [f"{name} is empty" for name in text_fields if not str(getattr(spec, name)).strip()]


def _modification_problems(modifications: tuple[str, ...]) -> list[str]:
    if modifications and all(str(change).strip() for change in modifications):
        return []
    return ["modifications must list at least one change"]


def license_problems(spec: RestoreModelSpec) -> list[str]:
    problems = _blank_license_fields(spec) + _modification_problems(spec.modifications)
    if spec.commercial_use not in COMMERCIAL_USE:
        problems.append(f"commercial_use must be one of {COMMERCIAL_USE}")
    if not is_sha256(spec.source_sha256):
        problems.append("source_sha256 must be 64 lowercase hex characters")
    if spec.data_lineage.strip() and not is_data_lineage(spec.data_lineage):
        problems.append(f"data_lineage must combine {LINEAGE_KEYS} or be {NO_DATA_DEBT!r}")
    return problems


def identity_problems(spec: RestoreModelSpec) -> list[str]:
    problems = []
    if not is_safe_id(spec.id):
        problems.append(f"id {spec.id!r} is not a safe identifier")
    if not is_safe_id(spec.bundle):
        problems.append(f"bundle {spec.bundle!r} is not a safe identifier")
    if not str(spec.name).strip():
        problems.append("name is empty")
    if not is_safe_onnx_file(spec.filename):
        problems.append(f"filename {spec.filename!r} is not a plain .onnx name")
    if spec.fp16_filename is not None and not is_safe_onnx_file(spec.fp16_filename):
        problems.append(f"fp16_filename {spec.fp16_filename!r} is not a plain .onnx name")
    return problems


def _tile_problems(spec: RestoreModelSpec) -> list[str]:
    problems = []
    if spec.tile_min <= 0:
        problems.append("tile_min must be positive")
    if any(tile <= 0 for tile in spec.tile_candidates):
        problems.append("tile_candidates must be positive")
    if not 0 <= spec.overlap < max(spec.tile_min, 1):
        problems.append("overlap must be within [0, tile_min)")
    return problems


def _ceiling_problems(spec: RestoreModelSpec) -> list[str]:
    problems = []
    unknown = sorted(set(spec.tile_by_precision) - set(PRECISIONS))
    if unknown:
        problems.append(f"tile_by_precision has unknown precisions {unknown}")
    if any(tile < spec.tile_min for tile in spec.tile_by_precision.values()):
        problems.append("tile_by_precision is below tile_min")
    if "fp16" in spec.tile_by_precision and spec.fp16_filename is None:
        problems.append("tile_by_precision has fp16 but the model has no fp16 file")
    return problems


def runtime_problems(spec: RestoreModelSpec) -> list[str]:
    problems = _tile_problems(spec) + _ceiling_problems(spec)
    if spec.channels not in (1, 3):
        problems.append("channels must be 1 or 3")
    if spec.vram_factor <= 0:
        problems.append("vram_factor must be positive")
    return problems


@dataclass(frozen=True, slots=True)
class RestoreArtifact:
    model_id: str
    precision: Precision
    filename: str
    url: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _raise_on(self.filename, artifact_problems(self))


def artifact_problems(artifact: RestoreArtifact) -> list[str]:
    problems = []
    if artifact.precision not in PRECISIONS:
        problems.append(f"precision must be one of {PRECISIONS}")
    if not is_safe_onnx_file(artifact.filename):
        problems.append(f"filename {artifact.filename!r} is not a plain .onnx name")
    if not artifact.url.startswith("https://"):
        problems.append("url must be https")
    if not is_sha256(artifact.sha256):
        problems.append("sha256 must be 64 lowercase hex characters")
    if artifact.size <= 0:
        problems.append("size must be positive")
    return problems


@dataclass(frozen=True, slots=True)
class RestoreLicenseFile:
    model_id: str
    filename: str
    url: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _raise_on(self.filename, license_file_problems(self))

    @property
    def relative_path(self) -> Path:
        return Path(LICENSES_DIRNAME, self.model_id, self.filename)


def license_file_problems(license_file: RestoreLicenseFile) -> list[str]:
    problems = []
    if not is_safe_id(license_file.model_id):
        problems.append(f"model_id {license_file.model_id!r} is not a safe identifier")
    if not is_license_file_name(license_file.filename):
        problems.append(f"filename {license_file.filename!r} is neither LICENSE* nor {NOTICE_NAME}")
    if not license_file.url.startswith("https://"):
        problems.append("url must be https")
    asset = release_license_asset(license_file.model_id, license_file.filename)
    if not license_file.url.endswith(f"/{asset}"):
        problems.append(f"url must point to the release asset {asset!r}")
    if not is_sha256(license_file.sha256):
        problems.append("sha256 must be 64 lowercase hex characters")
    if license_file.size <= 0:
        problems.append("size must be positive")
    return problems


@dataclass(frozen=True, slots=True)
class RestoreBundle:
    name: str
    artifacts: tuple[RestoreArtifact, ...]
    # licenses/<modelo>/{LICENSE*, NOTICE.txt} del Release, copiados junto a los modelos.
    license_files: tuple[RestoreLicenseFile, ...] = ()

    @property
    def pack_id(self) -> str:
        return pack_id(self.name)

    @property
    def manifest_name(self) -> str:
        return f"{self.name}{MANIFEST_SUFFIX}"


BUNDLE_NAMES: tuple[str, ...] = ("core", "faces", "colorize")

# Vacio hasta M-06/P1-15: los ONNX todavia no estan publicados. Llenarlo exige
# a la vez los artefactos de cada bundle y el anti-deriva de download-restore.ps1.
RESTORE_MODELS: dict[str, RestoreModelSpec] = {}

RESTORE_BUNDLES: dict[str, RestoreBundle] = {name: RestoreBundle(name, ()) for name in BUNDLE_NAMES}

# Declarado a mano y no derivado: el test lo compara con gated_packs(RESTORE_MODELS),
# asi un modelo no comercial nuevo no puede colarse sin pasar por la compuerta.
LICENSE_GATED_PACKS: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class VendoredModel:
    spec: RestoreModelSpec
    pack: str
    # Recibe Settings; queda sin tipar para no importar app.config desde este modulo de datos.
    path_of: Callable[[Any], Path]


MIGAN_MODEL_ID = "migan"
MIGAN_REVISION = "406830d0fa60666da0071c342ad2fbc8f30c5c64"

# Ya vendorizado por su propio pack (download-migan.ps1), fuera de los bundles restore-*:
# el dueno unico abre su sesion, pero la ruta la resuelve Settings.
VENDORED_MODELS: dict[str, VendoredModel] = {
    MIGAN_MODEL_ID: VendoredModel(
        RestoreModelSpec(
            id=MIGAN_MODEL_ID,
            name="MI-GAN",
            bundle="migan",
            filename="migan_pipeline_v2.onnx",
            license_spdx="MIT",
            license_url=f"https://huggingface.co/andraniksargsyan/migan/blob/{MIGAN_REVISION}/LICENSE",
            copyright="Copyright (c) 2024 Picsart AI Research",
            attribution="MI-GAN (Sargsyan et al., ICCV 2023), Picsart AI Research",
            data_lineage="D1a",
            commercial_use="unclear",
            source_url="https://huggingface.co/andraniksargsyan/migan",
            source_revision=MIGAN_REVISION,
            source_sha256="6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b",
            modifications=("none: the published migan_pipeline_v2.onnx is used unchanged",),
            tile_min=512,
            overlap=128,
        ),
        pack="migan",
        path_of=lambda settings: settings.migan_model_path,
    ),
}


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    key: str
    name: str
    # Lo que dice la fuente primaria, citado o resumido sin interpretar.
    terms: str
    clause: str
    primary_source: str
    verified_on: str
    excluded_uses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _raise_on(repr(self.key), dataset_problems(self))

    @property
    def lineage(self) -> str | None:
        return CLAUSE_LINEAGE[self.clause]


def dataset_problems(dataset: TrainingDataset) -> list[str]:
    problems = []
    if not is_safe_id(dataset.key):
        problems.append(f"key {dataset.key!r} is not a safe identifier")
    if not dataset.name.strip():
        problems.append("name is empty")
    if not dataset.terms.strip():
        problems.append("terms is empty")
    if dataset.clause not in CLAUSE_LINEAGE:
        problems.append(f"clause must be one of {tuple(CLAUSE_LINEAGE)}")
    if not dataset.primary_source.startswith("https://"):
        problems.append("primary_source must be https")
    if not _ISO_DATE.fullmatch(dataset.verified_on):
        problems.append("verified_on must be an ISO date")
    return problems


def _dataset(key: str, name: str, terms: str, clause: str, source: str, **extra: Any) -> TrainingDataset:
    return TrainingDataset(key, name, terms, clause, source, verified_on="2026-09-26", **extra)


# Tabla de linaje de §3.7 verificada en P4-LIC; el detalle y las citas estan en
# docs/superpowers/specs/2026-09-26-restore-data-lineage.md.
TRAINING_DATASETS: dict[str, TrainingDataset] = {
    dataset.key: dataset
    for dataset in (
        _dataset(
            "gopro",
            "GoPro (Nah et al., CVPR 2017)",
            "GOPRO dataset is released under CC BY 4.0 license.",
            "permissive",
            "https://seungjunnah.github.io/Datasets/gopro.html",
        ),
        _dataset(
            "sidd",
            "SIDD (Abdelhamed et al., CVPR 2018)",
            "The dataset and the associated code repositories are under the MIT License.",
            "permissive",
            "https://abdokamel.github.io/sidd/",
        ),
        _dataset(
            "dpdd",
            "DPDD (Abuolaim and Brown, ECCV 2020)",
            "The README states no dataset license; the repository LICENSE is MIT (software wording).",
            "unclear-scope",
            "https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel",
        ),
        _dataset(
            "davis-2017",
            "DAVIS 2017 (Pont-Tuset et al.)",
            "The download page states no terms; the toolkit README says 'DAVIS is released under the BSD "
            "License' next to a software BSD-3 LICENSE.",
            "unclear-scope",
            "https://davischallenge.org/davis2017/code.html",
        ),
        _dataset(
            "reds",
            "REDS (Nah et al., CVPRW 2019)",
            "REDS dataset is released under CC BY 4.0 license.",
            "permissive",
            "https://seungjunnah.github.io/Datasets/reds.html",
        ),
        _dataset(
            "lol",
            "LOL (Wei et al., BMVC 2018)",
            "The project page offers the download with no license or terms.",
            "no-license",
            "https://daooshee.github.io/BMVC2018website/",
        ),
        _dataset(
            "sa-1b",
            "SA-1B (Kirillov et al., 2023)",
            "SA-1B Dataset Research License: Research Purposes only, on a non-commercial basis; no use for "
            "surveillance, biometric processing or identifying individuals.",
            "research-only",
            "https://ai.meta.com/datasets/segment-anything/",
            excluded_uses=("surveillance", "biometric processing", "identifying individuals"),
        ),
        _dataset(
            "mit-adobe-fivek",
            "MIT-Adobe FiveK (Bychkovsky et al., CVPR 2011)",
            "Adobe research license: solely for your own research purposes, not directed toward commercial "
            "advantage or monetary compensation.",
            "research-only",
            "https://data.csail.mit.edu/graphics/fivek/",
        ),
        _dataset(
            "imagenet",
            "ImageNet",
            "Researcher shall use the Database only for non-commercial research and educational purposes.",
            "research-only",
            "https://image-net.org/download.php",
        ),
        _dataset(
            "sid",
            "See-in-the-Dark (Chen et al., CVPR 2018)",
            "The code repository README says 'License: MIT License' without saying whether it covers "
            "the images.",
            "unclear-scope",
            "https://github.com/cchen156/Learning-to-See-in-the-Dark",
        ),
    )
}

# Checkpoints candidatos de P4 -> datos con que se entrenaron (incluidas las redes de la perdida).
P4_MODEL_DATASETS: dict[str, tuple[str, ...]] = {
    "nafnet-gopro": ("gopro",),
    "nafnet-sidd": ("sidd",),
    "restormer-motion-deblur": ("gopro",),
    "restormer-defocus-deblur": ("dpdd",),
    "restormer-real-denoise": ("sidd",),
    "fastdvdnet": ("davis-2017",),
    # REDS + la perdida perceptual VGG19 preentrenada en ImageNet.
    "realbasicvsr": ("reds", "imagenet"),
    "retinexformer-lol-v1": ("lol",),
    "retinexformer-fivek": ("mit-adobe-fivek",),
    "retinexformer-sid": ("sid",),
    "mobilesam": ("sa-1b",),
}


def lineage_of(
    dataset_keys: tuple[str, ...], datasets: Mapping[str, TrainingDataset] = TRAINING_DATASETS
) -> str:
    if not dataset_keys:
        raise ValueError("lineage_of needs at least one dataset")
    found = {datasets[key].lineage for key in dataset_keys}
    keys = [key for key in LINEAGE_KEYS if key in found]
    return LINEAGE_SEPARATOR.join(keys) if keys else NO_DATA_DEBT


def excluded_uses_of(
    dataset_keys: tuple[str, ...], datasets: Mapping[str, TrainingDataset] = TRAINING_DATASETS
) -> tuple[str, ...]:
    uses = (use for key in dataset_keys for use in datasets[key].excluded_uses)
    return tuple(dict.fromkeys(uses))


def gated_packs(models: Mapping[str, RestoreModelSpec]) -> frozenset[str]:
    return frozenset(spec.pack_id for spec in models.values() if spec.commercial_use == "no")


def _bundle_key_problems(key: str, bundle: RestoreBundle) -> list[str]:
    return [] if key == bundle.name else [f"bundle key {key!r} does not match its name {bundle.name!r}"]


def _duplicate_file_problems(bundle: RestoreBundle) -> list[str]:
    names = [artifact.filename for artifact in bundle.artifacts]
    repeated = sorted({name for name in names if names.count(name) > 1})
    return [f"{bundle.name}: file {name!r} is duplicated" for name in repeated]


def _artifact_catalog_problems(
    bundle: RestoreBundle, artifact: RestoreArtifact, models: Mapping[str, RestoreModelSpec]
) -> list[str]:
    spec = models.get(artifact.model_id)
    if spec is None:
        return [f"{bundle.name}: {artifact.filename!r} belongs to unknown model {artifact.model_id!r}"]
    problems = []
    if spec.bundle != bundle.name:
        problems.append(f"{bundle.name}: {artifact.model_id!r} is declared in bundle {spec.bundle!r}")
    if spec.files_by_precision().get(artifact.precision) != artifact.filename:
        problems.append(
            f"{bundle.name}: {artifact.filename!r} is not the {artifact.precision} file of {spec.id!r}"
        )
    return problems


def _missing_file_problems(bundle: RestoreBundle, spec: RestoreModelSpec) -> list[str]:
    shipped = {artifact.filename for artifact in bundle.artifacts}
    missing = sorted(set(spec.files_by_precision().values()) - shipped)
    return [f"{bundle.name}: {spec.id!r} needs {name!r}, which the bundle does not ship" for name in missing]


def _non_commercial_problems(bundle: RestoreBundle, spec: RestoreModelSpec) -> list[str]:
    if spec.commercial_use != "no":
        return []
    return [f"{bundle.name}: non-commercial model {spec.id!r} cannot be in a default bundle"]


def _duplicate_license_problems(bundle: RestoreBundle) -> list[str]:
    paths = [license_file.relative_path.as_posix() for license_file in bundle.license_files]
    repeated = sorted({path for path in paths if paths.count(path) > 1})
    return [f"{bundle.name}: license file {path!r} is duplicated" for path in repeated]


def _license_catalog_problems(
    bundle: RestoreBundle, license_file: RestoreLicenseFile, models: Mapping[str, RestoreModelSpec]
) -> list[str]:
    owner = f"{bundle.name}: license {license_file.filename!r}"
    spec = models.get(license_file.model_id)
    if spec is None:
        return [f"{owner} belongs to unknown model {license_file.model_id!r}"]
    if spec.bundle != bundle.name:
        return [f"{owner} of {spec.id!r}, which is in bundle {spec.bundle!r}"]
    return []


def _missing_license_problems(bundle: RestoreBundle, spec: RestoreModelSpec) -> list[str]:
    names = [f.filename for f in bundle.license_files if f.model_id == spec.id]
    problems = []
    if NOTICE_NAME not in names:
        problems.append(f"{bundle.name}: {spec.id!r} ships without its {NOTICE_NAME}")
    if not any(is_license_name(name) for name in names):
        problems.append(f"{bundle.name}: {spec.id!r} ships without a LICENSE file")
    return problems


def _model_problems(bundle: RestoreBundle, spec: RestoreModelSpec) -> list[str]:
    return (
        _missing_file_problems(bundle, spec)
        + _missing_license_problems(bundle, spec)
        + _non_commercial_problems(bundle, spec)
    )


def _bundle_problems(key: str, bundle: RestoreBundle, models: Mapping[str, RestoreModelSpec]) -> list[str]:
    problems = _bundle_key_problems(key, bundle) + _duplicate_file_problems(bundle)
    problems += _duplicate_license_problems(bundle)
    for artifact in bundle.artifacts:
        problems += _artifact_catalog_problems(bundle, artifact, models)
    for license_file in bundle.license_files:
        problems += _license_catalog_problems(bundle, license_file, models)
    for spec in models.values():
        if spec.bundle == bundle.name:
            problems += _model_problems(bundle, spec)
    return problems


def catalog_problems(
    models: Mapping[str, RestoreModelSpec], bundles: Mapping[str, RestoreBundle]
) -> list[str]:
    problems = [
        f"model key {key!r} does not match its id {spec.id!r}" for key, spec in models.items() if key != spec.id
    ]
    for key, bundle in bundles.items():
        problems += _bundle_problems(key, bundle, models)
    return problems


def _file_present(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size


def bundle_installed(model_dir: Path, bundle: RestoreBundle) -> bool:
    # Un bundle sin artefactos no esta publicado: un manifiesto suelto no lo instala.
    if not bundle.artifacts or not (model_dir / bundle.manifest_name).is_file():
        return False
    models_present = all(_file_present(model_dir / a.filename, a.size) for a in bundle.artifacts)
    return models_present and all(
        _file_present(model_dir / f.relative_path, f.size) for f in bundle.license_files
    )


def installed_manifest(model_dir: Path, bundle: RestoreBundle) -> Path | None:
    return model_dir / bundle.manifest_name if bundle_installed(model_dir, bundle) else None


def model_path(
    model_dir: Path,
    model_id: str,
    precision: str,
    models: Mapping[str, RestoreModelSpec] = RESTORE_MODELS,
) -> Path | None:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    filename = models[model_id].files_by_precision().get(precision)
    return model_dir / filename if filename is not None else None
