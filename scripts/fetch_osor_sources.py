from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

GIB = 1024**3
REQUIRED_FREE_BYTES = 16 * GIB
HASH_CHUNK_BYTES = 8 * 1024 * 1024

BASE_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
BASE_REVISION = "115134f363124c53c7d878647567d04daf26e41e"
BASE_SUBDIR = "base"
WEIGHTS_REPO = "QinmingZhou/OSOR"
WEIGHTS_REVISION = "9288156766b1df44e5f50fb697876a6f2fd8c062"
WEIGHTS_SUBDIR = "weights"
WEIGHTS_FILE = "osor-sdxlinpainting/weights/sdxlinpainting_phase2.pth"

PINNED_BASE_FILES: dict[str, tuple[int, str]] = {
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
}
PINNED_WEIGHTS_FILES: dict[str, tuple[int, str]] = {
    WEIGHTS_FILE: (
        3112296806,
        "4f6abee0f152919fa8faca1895820ee531adb2b42b5de25db2e9a311aa47face",
    ),
}

SMALL_BASE_FILES = frozenset(
    {
        "unet/config.json",
        "vae/config.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json",
        "scheduler/scheduler_config.json",
    }
)
SMALL_BASE_FOLDERS = ("tokenizer/", "tokenizer_2/")


def gib_text(size: int) -> str:
    return f"{size / GIB:.1f} GB"


def pinned_local_files(dest: Path) -> list[tuple[Path, str]]:
    base = dest / BASE_SUBDIR
    weights = dest / WEIGHTS_SUBDIR
    return [(base, relative) for relative in PINNED_BASE_FILES] + [
        (weights, relative) for relative in PINNED_WEIGHTS_FILES
    ]


def partial_paths(local_dir: Path, relative: str) -> tuple[Path, ...]:
    final = local_dir / relative
    cache_partial = local_dir / ".cache" / "huggingface" / "download" / f"{relative}.incomplete"
    return (final, final.with_name(final.name + ".incomplete"), cache_partial)


def bytes_present(local_dir: Path, relative: str) -> int:
    return sum(path.stat().st_size for path in partial_paths(local_dir, relative) if path.is_file())


def already_downloaded_bytes(dest: Path) -> int:
    return sum(bytes_present(local_dir, relative) for local_dir, relative in pinned_local_files(dest))


def required_free_bytes(dest: Path) -> int:
    return max(0, REQUIRED_FREE_BYTES - already_downloaded_bytes(dest))


def ensure_free_space(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    required = required_free_bytes(dest)
    if free < required:
        raise SystemExit(
            f"Hacen falta {gib_text(required)} libres en {dest} para bajar OSOR; "
            f"hay {gib_text(free)} libres."
        )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integrity_problem(path: Path, size: int, sha256: str) -> str | None:
    actual_size = path.stat().st_size
    if actual_size != size:
        return f"{actual_size} bytes, se esperaban {size}"
    actual_sha256 = sha256_of(path)
    if actual_sha256 != sha256:
        return f"sha256 {actual_sha256}, se esperaba {sha256}"
    return None


def download(repo: str, revision: str, relative: str, local_dir: Path) -> Path:
    return Path(hf_hub_download(repo, relative, revision=revision, local_dir=local_dir))


def fetch_verified(
    repo: str, revision: str, relative: str, local_dir: Path, size: int, sha256: str
) -> None:
    path = download(repo, revision, relative, local_dir)
    problem = integrity_problem(path, size, sha256)
    if problem is None:
        return
    path.unlink(missing_ok=True)
    raise SystemExit(
        f"{relative} no verifica ({problem}). Se borro el archivo; vuelve a correr la descarga."
    )


def is_small_base_file(name: str) -> bool:
    return name in SMALL_BASE_FILES or name.startswith(SMALL_BASE_FOLDERS)


def small_base_files() -> list[str]:
    return sorted(
        name for name in list_repo_files(BASE_REPO, revision=BASE_REVISION) if is_small_base_file(name)
    )


def fetch_base(dest: Path) -> Path:
    local_dir = dest / BASE_SUBDIR
    for relative, (size, sha256) in PINNED_BASE_FILES.items():
        fetch_verified(BASE_REPO, BASE_REVISION, relative, local_dir, size, sha256)
    for relative in small_base_files():
        download(BASE_REPO, BASE_REVISION, relative, local_dir)
    return local_dir


def fetch_weights(dest: Path) -> Path:
    local_dir = dest / WEIGHTS_SUBDIR
    for relative, (size, sha256) in PINNED_WEIGHTS_FILES.items():
        fetch_verified(WEIGHTS_REPO, WEIGHTS_REVISION, relative, local_dir, size, sha256)
    return local_dir / WEIGHTS_FILE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baja la base SDXL-inpainting y los pesos OSOR en revisiones fijas.")
    parser.add_argument("--dest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dest = args.dest
    ensure_free_space(dest)
    base = fetch_base(dest)
    weights = fetch_weights(dest)
    print(f"Base lista en: {base}", flush=True)
    print(f"Pesos listos en: {weights}", flush=True)


if __name__ == "__main__":
    main()
