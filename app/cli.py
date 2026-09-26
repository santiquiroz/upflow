"""CLI headless `upflow`: reescalado y restauracion de fotos sin servidor ni UI, pensada para agentes.

    upflow upscale --in a.png --out b.webp --model realesrgan-x4plus --scale 2 --json
    upflow restore --in scan.tif --out foto.png --steps repair,denoise,tone --preset gentle --json
    upflow models --json | upflow health --json | upflow preflight --repo X | upflow install --repo X --yes

`--json` imprime UNA sola linea JSON en stdout (contrato en app/headless.py). Codigos
de salida: 0 ok, 2 argumentos, 3 modelo no instalado, 4 dispositivo, 5 fallo.
Nunca pregunta nada por consola: las descargas exigen --yes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app import headless
from app.services.photo_restore_chain import RESTORE_CHAIN, step_ids
from app.services.photo_restore_presets import PHOTO_PRESETS

Handler = Callable[[argparse.Namespace], Awaitable[dict[str, Any]]]
FORMATS = ("png", "jpg", "jpeg", "webp", "jxl", "avif")
RESTORE_FORMATS = ("png", "jpg", "jpeg", "webp")
UPSCALE_MODES = ("none", "classic", "ai")
ROTATIONS = (0, 90, 180, 270)
QUARTER_TURN = 90


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="upflow", description="Upflow headless (sin servidor)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upscale = subparsers.add_parser("upscale", help="reescala una imagen")
    upscale.add_argument("--in", dest="input_path", required=True)
    upscale.add_argument("--out", dest="output_path", required=True)
    upscale.add_argument("--model", default="realesrgan-x4plus")
    upscale.add_argument("--scale", type=int, choices=(2, 3, 4), default=4)
    upscale.add_argument("--tile", type=int, default=None, help="omitido=auto, 0=sin tiling, N>=32")
    upscale.add_argument("--tile-overlap", type=int, default=None, help="solo motor ONNX")
    upscale.add_argument("--device", default=None, help="cpu, dml:0... (omitido = DEFAULT_DEVICE)")
    upscale.add_argument("--format", choices=FORMATS, default=None, help="omitido = extension de --out")
    _add_json_flag(upscale)
    upscale.set_defaults(handler=run_upscale)

    _add_json_flag(subparsers.add_parser("models", help="modelos instalados")).set_defaults(handler=run_models)
    _add_json_flag(subparsers.add_parser("health", help="version, GPUs, packs, modelos")).set_defaults(handler=run_health)

    preflight = subparsers.add_parser("preflight", help="compatibilidad y capacidad antes de instalar")
    preflight.add_argument("--repo", required=True)
    _add_json_flag(preflight).set_defaults(handler=run_preflight)

    add_restore_parser(subparsers)

    install = subparsers.add_parser("install", help="instala un upscaler desde Hugging Face")
    install.add_argument("--repo", required=True)
    install.add_argument("--yes", action="store_true", help="obligatorio: autoriza la descarga")
    _add_json_flag(install).set_defaults(handler=run_install)
    return parser


def add_restore_parser(subparsers: Any) -> None:
    restore = subparsers.add_parser("restore", help="restaura una foto (sin servidor)")
    restore.add_argument("--in", dest="input_path", required=True)
    restore.add_argument("--out", dest="output_path", required=True)
    steps_help = f"CSV de {','.join(step_ids(RESTORE_CHAIN))}; omitido = lo que proponga el analisis"
    restore.add_argument("--steps", default=None, help=steps_help)
    restore.add_argument("--preset", choices=[preset.id for preset in PHOTO_PRESETS], default=None)
    restore.add_argument("--scale", type=int, default=1)
    restore.add_argument("--upscale", choices=UPSCALE_MODES, default=None, help="omitido: none con --scale 1, si no ai")
    restore.add_argument("--model", default=headless.DEFAULT_SR_MODEL, help="modelo SR para --upscale ai")
    restore.add_argument("--face-blend", type=float, default=None)
    restore.add_argument("--rotate", type=int, choices=ROTATIONS, default=0)
    restore.add_argument("--crop", default=None, help="x,y,w,h en pixeles, despues de girar")
    restore.add_argument("--device", default=None, help="cpu, dml:0... (omitido = DEFAULT_DEVICE)")
    restore.add_argument("--format", choices=RESTORE_FORMATS, default=None, help="omitido = extension de --out")
    _add_json_flag(restore).set_defaults(handler=run_restore)


def _add_json_flag(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--json", action="store_true", help="una sola linea JSON en stdout")
    return parser


# ---------------------------------------------------------------- handlers


async def run_upscale(args: argparse.Namespace) -> dict[str, Any]:
    return await headless.upscale_image(
        headless.build_context(),
        Path(args.input_path),
        Path(args.output_path),
        model=args.model,
        scale=args.scale,
        tile_size=args.tile,
        tile_overlap=args.tile_overlap,
        device=args.device,
        output_format=args.format,
    )


async def run_restore(args: argparse.Namespace) -> dict[str, Any]:
    return await headless.restore_image(
        headless.build_context(),
        Path(args.output_path),
        restore_spec_from_args(args),
        source=Path(args.input_path),
    )


def restore_spec_from_args(args: argparse.Namespace) -> headless.RestoreSpec:
    return headless.RestoreSpec(
        steps=parse_steps(args.steps),
        options=restore_options_from_args(args),
        preset=args.preset,
        scale=args.scale,
        model=args.model,
        device=args.device,
        output_format=args.format,
    )


def parse_steps(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(step.strip() for step in raw.split(",") if step.strip())


def parse_box(raw: str) -> list[int]:
    parts = raw.split(",")
    try:
        box = [int(part) for part in parts]
    except ValueError as exc:
        raise headless.UsageError(f"--crop needs four integers x,y,w,h, got {raw!r}") from exc
    if len(box) != 4:
        raise headless.UsageError(f"--crop needs four integers x,y,w,h, got {raw!r}")
    return box


def geometry_from_args(rotate: int, crop: str | None) -> dict[str, Any]:
    geometry: dict[str, Any] = {}
    if rotate:
        geometry["rotate90"] = rotate // QUARTER_TURN
    if crop is not None:
        geometry["crop"] = parse_box(crop)
    return geometry


def restore_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    geometry = geometry_from_args(args.rotate, args.crop)
    if geometry:
        options["geometry"] = geometry
    if args.upscale is not None:
        options["upscale_mode"] = args.upscale
    if args.face_blend is not None:
        options["faces"] = {"blend": args.face_blend}
    return options


async def run_models(args: argparse.Namespace) -> dict[str, Any]:
    return headless.list_models(headless.build_context())


async def run_health(args: argparse.Namespace) -> dict[str, Any]:
    return headless.health(headless.build_context())


async def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    return await headless.preflight_model(headless.build_context(), args.repo)


async def run_install(args: argparse.Namespace) -> dict[str, Any]:
    if not args.yes:
        raise headless.UsageError("downloads need --yes")
    return await headless.install_model(headless.build_context(), args.repo)


# ---------------------------------------------------------------- salida


def print_human(command: str, payload: dict[str, Any]) -> None:
    if command == "upscale":
        tile = payload.get("tile") or {}
        print(
            f"wrote {payload.get('output')} ({payload.get('width')}x{payload.get('height')}) "
            f"model={payload.get('model')} device={payload.get('device')} "
            f"tile={tile.get('size')} {payload.get('seconds')}s"
        )
        return
    if command == "restore":
        print(
            f"wrote {payload.get('output')} ({payload.get('width')}x{payload.get('height')}) "
            f"steps={','.join(payload.get('steps', []))} device={payload.get('device')} {payload.get('seconds')}s"
        )
        return
    if command == "models":
        for model in payload.get("models", []):
            scales = ",".join(str(scale) for scale in model.get("scales", []))
            print(f"{model.get('id')}  {model.get('kind')}  {scales}")
        return
    if command == "health":
        devices = ", ".join(f"{d.get('id')} freeVramMb={d.get('freeVramMb')}" for d in payload.get("devices", []))
        print(f"version={payload.get('version')} devices={devices} modelsInstalled={payload.get('modelsInstalled')}")
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def print_error(message: str, code: int, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps({"ok": False, "error": message, "code": code}, ensure_ascii=False))
        return
    print(f"error: {message}", file=sys.stderr)


def run_command(args: argparse.Namespace, handler: Handler) -> int:
    try:
        payload = asyncio.run(handler(args))
    except headless.HeadlessError as exc:
        print_error(str(exc), exc.exit_code, args.json)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 - la CLI nunca vuelca un traceback a un agente
        print_error(str(exc), headless.EXIT_FAILED, args.json)
        return headless.EXIT_FAILED
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print_human(args.command, payload)
    return headless.EXIT_OK if payload.get("ok", True) else headless.EXIT_FAILED


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_command(args, args.handler)


if __name__ == "__main__":
    raise SystemExit(main())
