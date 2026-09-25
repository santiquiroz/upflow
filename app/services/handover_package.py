"""Paquete de entrega CCTV y "Check files are unchanged" (spec §4.11 y §4.13).

El directorio del job ya tiene la forma del paquete (`01_original` ...
`04_stills` y los archivos sueltos de la raiz), asi que las rutas de
`SHA256SUMS.txt` valen igual en el disco y dentro del ZIP. El ZIP va sin
compresion (`ZIP_STORED`, ZIP64) bajo una sola carpeta raiz y solo lleva los
archivos del esquema: nunca `work.mkv` ni restos de la sesion.
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from app.services.cctv_clarify_runner import (
    REPRODUCE_DIR,
    ReproduceCheck,
    ReproduceStep,
    build_reproduce_script,
)
from app.services.cctv_ingest import (
    FRAME_INDEX_NAME,
    ORIGINAL_DIRNAME,
    WORK_COPY_NAME,
    RemuxAttempt,
    SourceRecord,
    WorkingCopy,
    build_remux_command,
    sha256_file,
)
from app.services.cctv_report import (
    DISCLAIMER_TEXT,
    GUIDELINES_TEXT,
    HASH_SCOPE,
    REPORT_HTML_NAME,
    REPORT_JSON_NAME,
    SHA256SUMS_NAME,
    parse_sha256sums,
    write_sha256sums,
)
from app.services.cctv_report_model import ReportMode
from app.services.ffmpeg_capabilities import FfmpegCapabilities

PROCESSED_DIRNAME = "02_processed"
COMPARISONS_DIRNAME = "03_comparisons"
STILLS_DIRNAME = "04_stills"
MEMBER_DIRS = (ORIGINAL_DIRNAME, PROCESSED_DIRNAME, COMPARISONS_DIRNAME, STILLS_DIRNAME)
REPRODUCE_NAME = "reproduce.cmd"
README_NAME = "README.txt"
ROOT_FILES = (FRAME_INDEX_NAME, REPORT_JSON_NAME, REPORT_HTML_NAME, SHA256SUMS_NAME, REPRODUCE_NAME, README_NAME)

PROCESSED_TAG = "__upflow-clarify__"
ANALYSIS_EXTENSION = ".mkv"
VIEWING_EXTENSION = ".mp4"
FALLBACK_STEM = "clip"
PACKAGE_SUFFIX = "upflow"
MAX_LABEL_CHARS = 60
REMUX_LABEL = "working copy (remux without re-encoding)"
FFMPEG_PLACEHOLDER = Path("ffmpeg.exe")

NO_DISK_ROOM_WARNING = "cctv.package.noDiskRoom"
DISK_HEADROOM_FRACTION = 0.90
ZIP_ENTRY_OVERHEAD_BYTES = 1024
PART_SUFFIX = ".part"

_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_NOT_PROCESSED_CHAR = re.compile(r"[^A-Za-z0-9 ._-]")
_UNSAFE_LABEL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# --- Nombres ---


def check_job_id(job_id: str) -> str:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError(f"Unsafe job id for a file name: {job_id!r}")
    return job_id


def ascii_stem(original_name: str) -> str:
    # ASCII a proposito: reproduce.cmd compara con findstr, que no encuentra nombres UTF-8.
    folded = unicodedata.normalize("NFKD", PureWindowsPath(original_name).stem)
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    stem = _NOT_PROCESSED_CHAR.sub("_", ascii_only).strip(" .")
    return stem if any(char.isalnum() for char in stem) else FALLBACK_STEM


def processed_name(original_name: str, job_id: str, extension: str) -> str:
    return f"{ascii_stem(original_name)}{PROCESSED_TAG}{check_job_id(job_id)}{extension}"


def safe_label(raw: str | None) -> str:
    cleaned = _UNSAFE_LABEL_CHARS.sub("_", raw or "").strip(" .")
    return cleaned[:MAX_LABEL_CHARS].rstrip(" .")


def case_folder_label(case_label: str | None, original_name: str) -> str:
    return safe_label(case_label) or safe_label(PureWindowsPath(original_name).stem) or FALLBACK_STEM


def package_root_name(case_label: str | None, original_name: str, day: date) -> str:
    return f"{case_folder_label(case_label, original_name)}_{day.isoformat()}_{PACKAGE_SUFFIX}"


# --- Ubicacion de las salidas ---


@dataclass(frozen=True, slots=True)
class ProcessedFiles:
    analysis: Path
    viewing: Path


def place_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    return destination


def processed_path(job_dir: Path, original_name: str, job_id: str, extension: str) -> Path:
    return job_dir / PROCESSED_DIRNAME / processed_name(original_name, job_id, extension)


def place_processed(job_dir: Path, original_name: str, job_id: str, analysis: Path, viewing: Path) -> ProcessedFiles:
    return ProcessedFiles(
        analysis=place_file(analysis, processed_path(job_dir, original_name, job_id, ANALYSIS_EXTENSION)),
        viewing=place_file(viewing, processed_path(job_dir, original_name, job_id, VIEWING_EXTENSION)),
    )


# --- Miembros del paquete ---


@dataclass(frozen=True, slots=True)
class PackageMember:
    arcname: str
    path: Path


def directory_members(job_dir: Path, dirname: str) -> list[PackageMember]:
    folder = job_dir / dirname
    if not folder.is_dir():
        return []
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    return [PackageMember(path.relative_to(job_dir).as_posix(), path) for path in files]


def root_members(job_dir: Path) -> list[PackageMember]:
    return [PackageMember(name, job_dir / name) for name in ROOT_FILES if (job_dir / name).is_file()]


def package_members(job_dir: Path) -> tuple[PackageMember, ...]:
    nested = [member for dirname in MEMBER_DIRS for member in directory_members(job_dir, dirname)]
    return (*nested, *root_members(job_dir))


def checksum_paths(job_dir: Path) -> tuple[str, ...]:
    return tuple(member.arcname for member in package_members(job_dir) if member.arcname != SHA256SUMS_NAME)


# --- reproduce.cmd del paquete ---


@dataclass(frozen=True, slots=True)
class ReproduceSources:
    verified_copy: Path
    working_copy: WorkingCopy


@dataclass(frozen=True, slots=True)
class ReproducedOutput:
    produced_at: Path
    listed_as: str

    def __post_init__(self) -> None:
        # findstr compara bytes: un nombre no ASCII nunca coincidiria con la linea de SHA256SUMS.txt.
        if not self.listed_as.isascii():
            raise ValueError(f"reproduce.cmd can only check ASCII names: {self.listed_as!r}")

    @property
    def reproduced_as(self) -> str:
        return f"{REPRODUCE_DIR}/{PurePosixPath(self.listed_as).name}"


def successful_attempt(working: WorkingCopy) -> RemuxAttempt:
    attempt = next((a for a in working.attempts if a.ok and a.label == working.method), None)
    if attempt is None:
        raise ValueError(f"The working copy records no successful {working.method!r} remux")
    return attempt


def remux_step(sources: ReproduceSources) -> ReproduceStep:
    working = sources.working_copy
    argv = build_remux_command(
        FFMPEG_PLACEHOLDER, sources.verified_copy, working.path, successful_attempt(working).input_args, working.copytb
    )
    return ReproduceStep(REMUX_LABEL, tuple(argv))


def reproduce_path_map(sources: ReproduceSources, outputs: Sequence[ReproducedOutput]) -> dict[str, str]:
    return {
        str(sources.verified_copy): f"{ORIGINAL_DIRNAME}/{sources.verified_copy.name}",
        str(sources.working_copy.path): f"{REPRODUCE_DIR}/{WORK_COPY_NAME}",
        **{str(output.produced_at): output.reproduced_as for output in outputs},
    }


def package_reproduce_script(
    sources: ReproduceSources,
    steps: Sequence[ReproduceStep],
    outputs: Sequence[ReproducedOutput],
    caps: FfmpegCapabilities,
) -> str:
    checks = [ReproduceCheck(output.reproduced_as, output.listed_as) for output in outputs]
    return build_reproduce_script((remux_step(sources), *steps), checks, reproduce_path_map(sources, outputs), caps)


# --- README.txt ---


@dataclass(frozen=True, slots=True)
class ReadmeFacts:
    original_name: str
    original_sha256: str
    mode: ReportMode
    ffmpeg_version: str
    ffmpeg_sha256: str
    cpu_extensions: tuple[str, ...]


def readme_facts(source: SourceRecord, mode: ReportMode, caps: FfmpegCapabilities) -> ReadmeFacts:
    return ReadmeFacts(
        original_name=source.original_name,
        original_sha256=source.sha256,
        mode=mode,
        ffmpeg_version=caps.version,
        ffmpeg_sha256=caps.binary_sha256,
        cpu_extensions=caps.cpu_extensions,
    )


HASH_SCOPE_ES = (
    "Los valores SHA-256 se calcularon cuando Upflow recibió el archivo. Detectan cambios accidentales "
    "desde ese momento; no prueban que la grabación sea auténtica ni dicen qué pasó antes de cargarla."
)
REWRITE_EN = "Anyone who changes the files can also rewrite SHA256SUMS.txt."
REWRITE_ES = "Quien altere los archivos también puede regenerar SHA256SUMS.txt."
PROCESSED_LABEL_EN = "Processed version — not the original recording"
PROCESSED_LABEL_ES = "Versión procesada — no es la grabación original"
GUIDELINES_ES = (
    "El informe registra lo que las guías de SWGDE y ENFSI piden a un informe de procesamiento: "
    "hashes, cada paso en orden con su configuración y las versiones del software."
)
DISCLAIMER_ES = (
    "Upflow no es una herramienta forense certificada y ningún laboratorio forense la ha validado. "
    "Este informe documenta lo que hizo Upflow; no certifica autenticidad ni admisibilidad. "
    "Conserve y entregue siempre la grabación original."
)


def powershell_path(facts: ReadmeFacts) -> str:
    escaped = facts.original_name.replace("'", "''")
    return f"'01_original\\{escaped}'"


def cpu_text(facts: ReadmeFacts) -> str:
    return " ".join(facts.cpu_extensions) or "none detected"


def classic_only(facts: ReadmeFacts, lines: list[str]) -> list[str]:
    return lines if facts.mode == "classic" else []


def contents_en(facts: ReadmeFacts) -> list[str]:
    return [
        "WHAT IS IN THIS PACKAGE",
        "  01_original       The verified copy of the recording, byte for byte as Upflow received it, with its own name.",
        f"                    The original recording is the file in 01_original: {facts.original_name}",
        f"                    SHA-256: {facts.original_sha256}",
        "  02_processed      Processed copies. Their names never repeat the name of the original.",
        *classic_only(facts, [
            "                    The .mkv is lossless (use it for examination); the .mp4 is a lossy viewing copy.",
        ]),
        "  03_comparisons    Side-by-side video: original | processed.",
        "  04_stills         Exported frames (PNG) of the original and of the processed copy.",
        "  frame_index.csv   Frame index of the original: frame number, time, key frame and picture type.",
        "  report.json / report.html   Processing report: every step in order, its settings and software versions.",
        "  SHA256SUMS.txt    SHA-256 of the original and of every file Upflow created.",
        *classic_only(facts, [
            "  reproduce.cmd     Rebuilds the copies of 02_processed in reproduced\\ and compares them with SHA256SUMS.txt.",
        ]),
    ]  # fmt: skip


def contents_es(facts: ReadmeFacts) -> list[str]:
    return [
        "QUÉ TRAE ESTE PAQUETE",
        "  01_original       La copia verificada de la grabación, idéntica byte a byte a la que recibió Upflow y con su nombre.",
        f"                    La grabación original es el archivo de 01_original: {facts.original_name}",
        f"                    SHA-256: {facts.original_sha256}",
        "  02_processed      Copias procesadas. Sus nombres nunca repiten el del original.",
        *classic_only(facts, [
            "                    El .mkv es sin pérdida (para examinar); el .mp4 es una copia de visualización con pérdida.",
        ]),
        "  03_comparisons    Video lado a lado: original | procesado.",
        "  04_stills         Cuadros exportados (PNG) del original y de la copia procesada.",
        "  frame_index.csv   Índice de cuadros del original: número, tiempo, cuadro clave y tipo de imagen.",
        "  report.json / report.html   Informe de procesamiento: cada paso en orden, su configuración y las versiones.",
        "  SHA256SUMS.txt    SHA-256 del original y de cada archivo que creó Upflow.",
        *classic_only(facts, [
            "  reproduce.cmd     Vuelve a generar en reproduced\\ las copias de 02_processed y las compara con SHA256SUMS.txt.",
        ]),
    ]  # fmt: skip


def hashes_en(facts: ReadmeFacts) -> list[str]:
    return [
        "CHECKING THE HASHES",
        f"  PowerShell:           Get-FileHash -Algorithm SHA256 -LiteralPath {powershell_path(facts)}",
        "  Linux, macOS, Git Bash:  sha256sum -c SHA256SUMS.txt",
        f"  {HASH_SCOPE.text}",
        f"  {REWRITE_EN}",
    ]


def hashes_es(facts: ReadmeFacts) -> list[str]:
    return [
        "CÓMO COMPROBAR LOS HASHES",
        f"  PowerShell:           Get-FileHash -Algorithm SHA256 -LiteralPath {powershell_path(facts)}",
        "  Linux, macOS, Git Bash:  sha256sum -c SHA256SUMS.txt",
        f"  {HASH_SCOPE_ES}",
        f"  {REWRITE_ES}",
    ]


def reproduce_en(facts: ReadmeFacts) -> list[str]:
    if facts.mode != "classic":
        return [
            "REPRODUCING",
            "  The processed copies were made with an AI model: they are a visualization and cannot be reproduced bit for bit.",
        ]
    return [
        "REPRODUCING THE PROCESSED COPIES",
        "  Open a command prompt in this folder and run reproduce.cmd. If ffmpeg.exe is not on PATH, set it first:",
        '    set "FFMPEG=C:\\path\\to\\ffmpeg.exe"',
        "    reproduce.cmd",
        "  It rebuilds the copies in reproduced\\ from 01_original and prints MATCH or DIFFERENT for each one.",
        f"  It needs the same ffmpeg build: {facts.ffmpeg_version}",
        f"  ffmpeg.exe SHA-256: {facts.ffmpeg_sha256}",
        f"  and the same CPU features ({cpu_text(facts)}). Another build or CPU can give different bytes.",
    ]


def reproduce_es(facts: ReadmeFacts) -> list[str]:
    if facts.mode != "classic":
        return [
            "REPRODUCCIÓN",
            "  Las copias procesadas se hicieron con un modelo de IA: son una visualización y no se pueden reproducir bit a bit.",
        ]
    return [
        "CÓMO REPRODUCIR LAS COPIAS PROCESADAS",
        "  Abra una consola en esta carpeta y ejecute reproduce.cmd. Si ffmpeg.exe no está en el PATH, indíquelo antes:",
        '    set "FFMPEG=C:\\ruta\\a\\ffmpeg.exe"',
        "    reproduce.cmd",
        "  Vuelve a generar las copias en reproduced\\ desde 01_original e imprime MATCH o DIFFERENT para cada una.",
        f"  Necesita la misma build de ffmpeg: {facts.ffmpeg_version}",
        f"  SHA-256 de ffmpeg.exe: {facts.ffmpeg_sha256}",
        f"  y las mismas extensiones de CPU ({cpu_text(facts)}). Otra build u otra CPU pueden dar bytes distintos.",
    ]


CHECKLIST_EN = [
    "HANDING OVER A RECORDING",
    "  1. Act quickly: the recorder overwrites old footage. Check its oldest recording and lock the segment if it allows it.",
    "  2. Measure the clock offset: photograph the recorder screen showing its time next to the official time and",
    "     write down the difference in seconds.",
    "  3. Export in the native format from the recorder itself, with a margin before and after the event, from every",
    "     relevant camera.",
    "  4. Hand over the verified copy in 01_original as it is: do not trim, re-compress or enhance it.",
    f'  5. Hand over processed material separately, labelled "{PROCESSED_LABEL_EN}",',
    "     together with report.html and SHA256SUMS.txt. Never instead of the original.",
    "  6. Do not share the footage publicly before handing it over.",
]
CHECKLIST_ES = [
    "CÓMO ENTREGAR UNA GRABACIÓN",
    "  1. Actúe rápido: el grabador sobrescribe lo viejo. Revise la grabación más antigua y bloquee el tramo si se puede.",
    "  2. Mida el desfase del reloj: fotografíe la pantalla del grabador con su hora junto a la hora legal y anote",
    "     la diferencia en segundos.",
    "  3. Exporte en el formato nativo desde el propio grabador, con margen antes y después del hecho, de todas las",
    "     cámaras relevantes.",
    "  4. Entregue la copia verificada de 01_original tal cual: sin recortarla, recomprimirla ni mejorarla.",
    f'  5. Entregue el material procesado aparte, rotulado "{PROCESSED_LABEL_ES}",',
    "     junto con report.html y SHA256SUMS.txt. Nunca en lugar del original.",
    "  6. No difunda la grabación antes de entregarla.",
]


def english_section(facts: ReadmeFacts) -> list[str]:
    parts = (contents_en(facts), hashes_en(facts), reproduce_en(facts), CHECKLIST_EN, [GUIDELINES_TEXT, DISCLAIMER_TEXT])
    return [line for part in parts for line in (*part, "")]


def spanish_section(facts: ReadmeFacts) -> list[str]:
    parts = (contents_es(facts), hashes_es(facts), reproduce_es(facts), CHECKLIST_ES, [GUIDELINES_ES, DISCLAIMER_ES])
    return [line for part in parts for line in (*part, "")]


def readme_text(facts: ReadmeFacts, root_name: str) -> str:
    title = f"Upflow handover package / Paquete de entrega de Upflow — {root_name}"
    lines = [
        title,
        "=" * len(title),
        "",
        "ENGLISH",
        "",
        *english_section(facts),
        "ESPAÑOL",
        "",
        *spanish_section(facts),
    ]
    return "\r\n".join(lines).rstrip() + "\r\n"


# --- Cierre del directorio del job ---


def write_text_file(path: Path, text: str) -> Path:
    path.write_bytes(text.encode("utf-8"))
    return path


def finalize_package_files(
    job_dir: Path, root_name: str, facts: ReadmeFacts, reproduce_script: str | None
) -> Path:
    if reproduce_script is not None:
        write_text_file(job_dir / REPRODUCE_NAME, reproduce_script)
    write_text_file(job_dir / README_NAME, readme_text(facts, root_name))
    return write_sha256sums(job_dir, checksum_paths(job_dir))


# --- ZIP ---


@dataclass(frozen=True, slots=True)
class PackageOutcome:
    path: Path | None
    warnings: tuple[str, ...] = ()


def free_disk_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def needed_bytes(members: Sequence[PackageMember]) -> int:
    return sum(member.path.stat().st_size + ZIP_ENTRY_OVERHEAD_BYTES for member in members)


def has_room(folder: Path, needed: int, free_bytes: Callable[[Path], int]) -> bool:
    return needed <= int(free_bytes(folder) * DISK_HEADROOM_FRACTION)


def write_zip(members: Sequence[PackageMember], destination: Path, root_name: str) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True, strict_timestamps=False) as archive:
        for member in members:
            archive.write(member.path, f"{root_name}/{member.arcname}")


def write_zip_atomically(members: Sequence[PackageMember], destination: Path, root_name: str) -> Path:
    partial = destination.with_name(destination.name + PART_SUFFIX)
    try:
        write_zip(members, partial, root_name)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def build_handover_package(
    job_dir: Path,
    destination: Path,
    root_name: str,
    free_bytes: Callable[[Path], int] = free_disk_bytes,
) -> PackageOutcome:
    members = package_members(job_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not has_room(destination.parent, needed_bytes(members), free_bytes):
        return PackageOutcome(None, (NO_DISK_ROOM_WARNING,))
    return PackageOutcome(write_zip_atomically(members, destination, safe_label(root_name) or FALLBACK_STEM))


# --- Check files are unchanged ---


@dataclass(frozen=True, slots=True)
class UnchangedCheck:
    checked: int
    mismatches: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.missing

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "checked": self.checked, "mismatches": list(self.mismatches), "missing": list(self.missing)}


LIST_MISSING = UnchangedCheck(0, (), (SHA256SUMS_NAME,))
LIST_TAMPERED = UnchangedCheck(0, (SHA256SUMS_NAME,), ())


def read_checksum_list(sums: Path) -> tuple[tuple[str, str], ...]:
    entries = parse_sha256sums(sums.read_text(encoding="utf-8"))
    if not entries:
        raise ValueError("SHA256SUMS.txt lists no files")
    return tuple((entry.path, entry.sha256) for entry in entries)


def compare_entries(base: Path, entries: Sequence[tuple[str, str]], hash_file: Callable[[Path], str]) -> UnchangedCheck:
    present = [(path, digest) for path, digest in entries if (base / path).is_file()]
    missing = tuple(path for path, _ in entries if not (base / path).is_file())
    mismatches = tuple(path for path, digest in present if hash_file(base / path) != digest)
    return UnchangedCheck(len(present), mismatches, missing)


def check_files_unchanged(base: Path, hash_file: Callable[[Path], str] = sha256_file) -> UnchangedCheck:
    sums = base / SHA256SUMS_NAME
    if not sums.is_file():
        return LIST_MISSING
    try:
        entries = read_checksum_list(sums)
    except (ValueError, UnicodeDecodeError):
        return LIST_TAMPERED
    return compare_entries(base, entries, hash_file)
