"""CLI headless `upflow`: reescalado sin servidor ni UI, pensada para agentes.

    upflow upscale --in a.png --out b.webp --model realesrgan-x4plus --scale 2 --json
    upflow models --json | upflow health --json | upflow preflight --repo X | upflow install --repo X --yes
    upflow cctv probe --in clip.mp4 --json
    upflow cctv clarify --in clip.mp4 --out-dir caso --preset day --no-osd --json
    upflow cctv verify --dir caso/<jobId>.cctv --json

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
from app.services.cctv_presets import CCTV_PRESETS

Handler = Callable[[argparse.Namespace], Awaitable[dict[str, Any]]]
FORMATS = ("png", "jpg", "jpeg", "webp", "jxl", "avif")
CCTV_PRESET_IDS = tuple(preset.id for preset in CCTV_PRESETS)
OSD_FLAGS_HINT = "Pass --osd x,y,w,h (one per box) with --osd-confirmed, or --no-osd."


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

    add_cctv_commands(subparsers)
    return parser


def add_cctv_commands(subparsers: argparse._SubParsersAction) -> None:
    cctv = subparsers.add_parser("cctv", help="video de camaras de seguridad, carril clasico en CPU")
    commands = cctv.add_subparsers(dest="cctv_command", required=True)

    probe = commands.add_parser("probe", help="hash, contenedor, indice de cuadros y diagnostico")
    probe.add_argument("--in", dest="input_path", required=True)
    _add_json_flag(probe).set_defaults(handler=run_cctv_probe)

    clarify = commands.add_parser("clarify", help="Clarify video: filtros clasicos deterministas + paquete")
    clarify.add_argument("--in", dest="input_path", required=True)
    clarify.add_argument("--out-dir", dest="out_dir", required=True)
    clarify.add_argument("--preset", choices=CCTV_PRESET_IDS, default=None, help="omitido = el sugerido")
    clarify.add_argument("--osd", dest="osd_boxes", action="append", type=parse_box, default=[], metavar="X,Y,W,H")
    decision = clarify.add_mutually_exclusive_group()
    decision.add_argument("--osd-confirmed", action="store_true", help="las cajas --osd tapan la hora y la camara")
    decision.add_argument("--no-osd", action="store_true", help="el video no tiene texto en pantalla")
    clarify.add_argument("--trim", type=parse_trim, default=None, metavar="A:B", help="primer y ultimo cuadro")
    clarify.add_argument("--frames", type=parse_frames, default=(), metavar="N,M", help="cuadros a exportar")
    _add_json_flag(clarify).set_defaults(handler=run_cctv_clarify)

    verify = commands.add_parser("verify", help="Check files are unchanged (SHA256SUMS.txt)")
    verify.add_argument("--dir", dest="result_dir", required=True)
    _add_json_flag(verify).set_defaults(handler=run_cctv_verify)


def parse_ints(raw: str, separator: str, count: int | None = None) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in raw.split(separator))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integers separated by {separator!r}: {raw!r}") from exc
    if count is not None and len(values) != count:
        raise argparse.ArgumentTypeError(f"expected {count} integers separated by {separator!r}: {raw!r}")
    return values


def parse_box(raw: str) -> tuple[int, int, int, int]:
    x, y, width, height = parse_ints(raw, ",", 4)
    return x, y, width, height


def parse_trim(raw: str) -> tuple[int, int]:
    first, last = parse_ints(raw, ":", 2)
    return first, last


def parse_frames(raw: str) -> tuple[int, ...]:
    return parse_ints(raw, ",")


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


async def run_cctv_probe(args: argparse.Namespace) -> dict[str, Any]:
    return await headless.cctv_probe(headless.build_context(), Path(args.input_path))


def clarify_choices(args: argparse.Namespace) -> headless.CctvClarifyChoices:
    return headless.CctvClarifyChoices(
        preset=args.preset,
        osd_boxes=tuple(args.osd_boxes),
        osd_confirmed=args.osd_confirmed,
        no_osd=args.no_osd,
        trim=args.trim,
        still_frames=tuple(args.frames),
    )


def check_osd_flags(choices: headless.CctvClarifyChoices) -> None:
    try:
        headless.check_osd_decision(choices)
    except headless.UsageError as exc:
        raise headless.UsageError(f"{exc} {OSD_FLAGS_HINT}", key=exc.key) from exc


async def run_cctv_clarify(args: argparse.Namespace) -> dict[str, Any]:
    choices = clarify_choices(args)
    check_osd_flags(choices)
    return await headless.cctv_clarify_file(
        headless.build_context(), Path(args.input_path), Path(args.out_dir), choices
    )


async def run_cctv_verify(args: argparse.Namespace) -> dict[str, Any]:
    return headless.cctv_check_unchanged(Path(args.result_dir))


# ---------------------------------------------------------------- salida


def print_upscale(payload: dict[str, Any]) -> None:
    tile = payload.get("tile") or {}
    print(
        f"wrote {payload.get('output')} ({payload.get('width')}x{payload.get('height')}) "
        f"model={payload.get('model')} device={payload.get('device')} "
        f"tile={tile.get('size')} {payload.get('seconds')}s"
    )


def print_models(payload: dict[str, Any]) -> None:
    for model in payload.get("models", []):
        scales = ",".join(str(scale) for scale in model.get("scales", []))
        print(f"{model.get('id')}  {model.get('kind')}  {scales}")


def print_health(payload: dict[str, Any]) -> None:
    devices = ", ".join(f"{d.get('id')} freeVramMb={d.get('freeVramMb')}" for d in payload.get("devices", []))
    print(f"version={payload.get('version')} devices={devices} modelsInstalled={payload.get('modelsInstalled')}")


def listed(values: list[str]) -> str:
    return ",".join(values) or "-"


def print_cctv_probe(payload: dict[str, Any]) -> None:
    container = (payload.get("container") or {}).get("label")
    print(
        f"sha256={payload.get('sourceSha256')} container={container} "
        f"suggestedPreset={payload.get('suggestedPreset')} decodeFailed={payload.get('decodeFailed')} "
        f"warnings={listed(payload.get('warnings', []))}"
    )


def print_cctv_clarify(payload: dict[str, Any]) -> None:
    print(
        f"wrote {payload.get('outputDir')} job={payload.get('jobId')} preset={payload.get('preset')} "
        f"frames={payload.get('framesIn')}->{payload.get('framesOut')} sha256={payload.get('sourceSha256')} "
        f"{payload.get('seconds')}s"
    )


def print_cctv_verify(payload: dict[str, Any]) -> None:
    if payload.get("ok"):
        print(f"unchanged: {payload.get('checked')} files in {payload.get('directory')}")
        return
    print(f"CHANGED: {listed(payload.get('mismatches', []))} MISSING: {listed(payload.get('missing', []))}")


HUMAN_PRINTERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "upscale": print_upscale,
    "models": print_models,
    "health": print_health,
    "cctv probe": print_cctv_probe,
    "cctv clarify": print_cctv_clarify,
    "cctv verify": print_cctv_verify,
}


def command_name(args: argparse.Namespace) -> str:
    sub = getattr(args, "cctv_command", None)
    return args.command if sub is None else f"{args.command} {sub}"


def print_human(command: str, payload: dict[str, Any]) -> None:
    printer = HUMAN_PRINTERS.get(command)
    if printer is None:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    printer(payload)


def print_error(message: str, code: int, json_mode: bool, key: str | None = None) -> None:
    if json_mode:
        keyed = {"key": key} if key else {}
        print(json.dumps({"ok": False, "error": message, "code": code, **keyed}, ensure_ascii=False))
        return
    suffix = f" [{key}]" if key else ""
    print(f"error: {message}{suffix}", file=sys.stderr)


def run_command(args: argparse.Namespace, handler: Handler) -> int:
    try:
        payload = asyncio.run(handler(args))
    except headless.HeadlessError as exc:
        print_error(str(exc), exc.exit_code, args.json, exc.key)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 - la CLI nunca vuelca un traceback a un agente
        print_error(str(exc), headless.EXIT_FAILED, args.json)
        return headless.EXIT_FAILED
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print_human(command_name(args), payload)
    return headless.EXIT_OK if payload.get("ok", True) else headless.EXIT_FAILED


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_command(args, args.handler)


if __name__ == "__main__":
    raise SystemExit(main())
