"""CLI headless `upflow`: reescalado sin servidor ni UI, pensada para agentes.

    upflow upscale --in a.png --out b.webp --model realesrgan-x4plus --scale 2 --json
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

Handler = Callable[[argparse.Namespace], Awaitable[dict[str, Any]]]
FORMATS = ("png", "jpg", "jpeg", "webp", "jxl", "avif")


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

    install = subparsers.add_parser("install", help="instala un upscaler desde Hugging Face")
    install.add_argument("--repo", required=True)
    install.add_argument("--yes", action="store_true", help="obligatorio: autoriza la descarga")
    _add_json_flag(install).set_defaults(handler=run_install)
    return parser


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
