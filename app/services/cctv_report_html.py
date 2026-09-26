"""`report.html` del modo CCTV (spec §4.11).

Plantilla en strings con `html.escape` en cada valor: el informe trae texto del
usuario (caso, operador, notas, nombre del archivo). Sin scripts, hojas de
estilo ni imagenes externas; se imprime a PDF desde el navegador.
"""

from __future__ import annotations

import html
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.services.cctv_report import LANE_NAMES
from app.services.cctv_report_model import (
    Acquisition,
    CctvReportV1,
    ClippingInfo,
    EnvironmentInfo,
    InputFile,
    OsdCheckInfo,
    OsdInfo,
    OutputFile,
    ProcessInfo,
    ReportStep,
    RoiFusionInfo,
    RoiSampleInfo,
    StillFrameInfo,
    StillPairInfo,
    TrimInfo,
)

EMPTY = "—"
KIND_LABELS = {"classic": "Classic (no AI)", "ai": "AI", "label": "AI label"}

STYLE = """
:root { color-scheme: light; }
body { font: 14px/1.45 system-ui, "Segoe UI", sans-serif; color: #111; background: #fff; margin: 24px auto; max-width: 1100px; padding: 0 16px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 8px; border-bottom: 1px solid #bbb; padding-bottom: 4px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #f2f2f2; }
code, pre, .hash { font-family: ui-monospace, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }
pre { white-space: pre-wrap; background: #f7f7f7; padding: 8px; margin: 4px 0; }
.ai { font-size: 26px; font-weight: 700; padding: 8px 12px; border: 3px solid #111; display: inline-block; margin: 12px 0; }
.ai.yes { border-color: #b00020; color: #b00020; }
.stills { display: flex; flex-wrap: wrap; gap: 16px; }
.stills figure { margin: 0; }
.stills img { max-width: 320px; height: auto; border: 1px solid #ccc; image-rendering: pixelated; }
footer { margin-top: 32px; font-size: 13px; }
"""


def esc(value: Any) -> str:
    return html.escape(EMPTY if value is None else str(value), quote=True)


def join(parts: Iterable[str]) -> str:
    return "\n".join(parts)


def row(cells: Sequence[str], tag: str = "td") -> str:
    return "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"


def table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    head = row([esc(header) for header in headers], "th")
    return f"<table><thead>{head}</thead><tbody>{join(row(cells) for cells in rows)}</tbody></table>"


def facts(pairs: Iterable[tuple[str, Any]]) -> str:
    return table(("Field", "Value"), ((esc(name), esc(value)) for name, value in pairs))


def section(title: str, body: str) -> str:
    return f"<section><h2>{esc(title)}</h2>\n{body}\n</section>"


def hash_cell(value: str | None) -> str:
    return f'<span class="hash">{esc(value)}</span>'


def link(url: str | None, text: str) -> str:
    return esc(text) if url is None else f'<a href="{esc(url)}">{esc(text)}</a>'


def command_text(argv: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(argv))


# --- Secciones ---


def header(report: CctvReportV1) -> str:
    source = report.inputs[0]
    verdict = "YES" if report.ai_used else "NO"
    css_class = "ai yes" if report.ai_used else "ai"
    return join(
        (
            "<header>",
            "<h1>Upflow processing report</h1>",
            f"<p>Original file: <strong>{esc(source.name)}</strong></p>",
            f"<p>SHA-256 (as Upflow received it): {hash_cell(source.sha256)}</p>",
            f"<p>Lane: <strong>{esc(LANE_NAMES[report.mode])}</strong></p>",
            f'<div class="{css_class}">AI used: {verdict}</div>',
            f"<p>Report generated {esc(report.generated_at.local)} ({esc(report.generated_at.utc)})"
            f" · Upflow {esc(report.upflow.version)}</p>",
            "</header>",
        )
    )


def case_section(report: CctvReportV1) -> str:
    case = report.case
    return section(
        "Case", facts((("Case", case.case_label), ("Operator", case.operator_name), ("Notes", case.notes)))
    )


def parameters_text(parameters: Mapping[str, Any]) -> str:
    return esc(", ".join(f"{name}={value}" for name, value in parameters.items()) or EMPTY)


def step_row(step: ReportStep) -> list[str]:
    what = f"{esc(step.description)}<br>{link(step.doc_url, 'Filter documentation')}"
    return [esc(step.index + 1), esc(step.label), what, parameters_text(step.parameters), esc(KIND_LABELS[step.category])]


def process_details(process: ProcessInfo) -> str:
    frames = f"frames in {esc(process.frames_in)}, out {esc(process.frames_out)}"
    summary = f"Command: {esc(process.label)} (exit code {esc(process.exit_code)}, {frames})"
    return f"<details><summary>{summary}</summary><pre>{esc(command_text(process.argv))}</pre></details>"


def steps_section(report: CctvReportV1) -> str:
    steps = table(("#", "Step", "What was done", "Settings", "Kind"), (step_row(step) for step in report.steps))
    commands = join(process_details(process) for process in report.processes)
    return section("Processing steps", f"{steps}\n{commands}")


def input_section(source: InputFile) -> str:
    working = source.working_copy
    pairs = (
        ("Name", source.name),
        ("Size (bytes)", source.size_bytes),
        ("Received (UTC)", source.received_at.utc),
        ("Received (local)", source.received_at.local),
        ("Container", source.container.label),
        ("Verified copy", source.verified_copy_path),
        ("Verified copy SHA-256", source.verified_copy_sha256),
        ("Working copy", working.note),
        ("Working copy method", working.method),
        ("Working copy SHA-256", working.sha256),
        ("Frame index SHA-256", None if source.frame_index is None else source.frame_index.csv_sha256),
    )
    return section("Original recording", facts(pairs))


def acquisition_section(acquisition: Acquisition) -> str:
    pairs = (
        ("Recorder make", acquisition.recorder_make),
        ("Recorder model", acquisition.recorder_model),
        ("Recorder serial number", acquisition.recorder_serial),
        ("Channel", acquisition.channel),
        ("Recorder clock offset (seconds, user-reported)", acquisition.clock_offset_seconds),
        ("How the offset was measured", acquisition.clock_offset_method),
        ("Export method", acquisition.export_method),
        ("Export date", acquisition.export_date),
    )
    return section("Acquisition", facts(pairs))


def box_text(box: Sequence[int]) -> str:
    x, y, w, h = box
    return f"x={x} y={y} w={w} h={h}"


def check_text(check: OsdCheckInfo) -> str:
    verdict = "looks like on-screen text" if check.looks_like_text else "doesn't look like on-screen text"
    return f"{box_text(check.box)}: {verdict}"


def osd_section(osd: OsdInfo) -> str:
    boxes = "; ".join(map(box_text, osd.boxes))
    checks = "; ".join(map(check_text, osd.checks))
    pairs = (
        ("On-screen text boxes", boxes or EMPTY),
        ("Boxes confirmed by the user", "yes" if osd.osd_boxes_confirmed else "no"),
        ("No on-screen text", "yes" if osd.no_osd else "no"),
        ("Box check", checks or EMPTY),
    )
    return section("On-screen text (OSD)", facts(pairs))


def trim_section(trim: TrimInfo | None) -> str:
    if trim is None:
        return ""
    pairs = (
        ("First frame", trim.start_frame),
        ("Last frame", trim.end_frame),
        ("Audio start (s)", trim.audio_start_seconds),
        ("Audio end (s)", trim.audio_end_seconds),
    )
    return section("Trim", facts(pairs))


def percent(value: float | None) -> str | None:
    return None if value is None else f"{value:.2f}%"


def clipping_section(clipping: ClippingInfo | None) -> str:
    if clipping is None:
        return ""
    pairs = (("Before", percent(clipping.before_pct)), ("After", percent(clipping.after_pct)))
    return section("Clipped pixels", facts(pairs))


def number_list(values: Sequence[float] | None) -> str | None:
    return None if values is None else ", ".join(f"{value:.4f}" for value in values)


def matrix_text(matrix: Sequence[Sequence[float]] | None) -> str:
    return EMPTY if matrix is None else "<br>".join(esc(number_list(line)) for line in matrix)


def roi_sample_row(sample: RoiSampleInfo) -> list[str]:
    ecc = None if sample.ecc is None else f"{sample.ecc:.4f}"
    cells = (sample.frame, sample.pict_type, sample.copy_group, sample.status, ecc, number_list(sample.shift))
    return [*(esc(cell) for cell in cells), f'<span class="hash">{matrix_text(sample.matrix)}</span>']


def roi_facts(roi: RoiFusionInfo) -> str:
    return facts(
        (
            ("Region type", roi.kind),
            ("Region (x, y, w, h) in stored pixels", box_text(roi.box)),
            ("Frames", f"{roi.first_frame}-{roi.last_frame} (reference {roi.reference_frame})"),
            ("Scale / combine", f"{roi.scale}x / {roi.method}"),
            ("Alignment", f"{roi.motion} (ECC minimum {roi.ecc_min})"),
            ("Frames used", f"{roi.frames_used} of {roi.frames_total} ({roi.effective_samples} carried new information)"),
            ("Rejected frames", ", ".join(map(str, roi.rejected_frames)) or None),
            ("Near-copies", "yes" if roi.near_copies else "no"),
            ("Libraries", ", ".join(f"{name} {version}" for name, version in roi.libraries.items())),
        )
    )


def roi_section(roi: RoiFusionInfo | None) -> str:
    if roi is None:
        return ""
    headers = ("Frame", "Type", "Copy group", "Status", "ECC", "Shift (px)", "Matrix (reference to frame)")
    return section("Multi-frame still", roi_facts(roi) + table(headers, map(roi_sample_row, roi.samples)))


def environment_section(environment: EnvironmentInfo) -> str:
    gpus = "; ".join(f"{gpu.name} (driver {gpu.driver_version or EMPTY})" for gpu in environment.gpus)
    runtime = environment.onnx_runtime
    pairs = (
        ("Operating system", environment.os),
        ("CPU", environment.cpu.model),
        ("CPU extensions", " ".join(environment.cpu.extensions) or EMPTY),
        ("GPU", gpus or EMPTY),
        ("ONNX Runtime", None if runtime is None else f"{runtime.version} ({', '.join(runtime.providers)})"),
        ("ffmpeg", environment.ffmpeg.version),
        ("ffmpeg.exe SHA-256", environment.ffmpeg_sha256),
        ("ffmpeg configuration", " ".join(environment.ffmpeg.configuration)),
        ("Python", environment.python),
    )
    return section("Environment", facts(pairs))


def output_row(output: OutputFile) -> list[str]:
    ai = "yes" if output.ai_applied else "no"
    return [esc(output.path), esc(output.role), hash_cell(output.sha256), esc(ai)]


def outputs_section(outputs: Sequence[OutputFile]) -> str:
    return section("Output files", table(("File", "Role", "SHA-256", "AI applied"), map(output_row, outputs)))


def still_figure(still: StillFrameInfo) -> str:
    caption = f"{esc(still.role)} · frame {esc(still.frame)} · {esc(still.timecode)}"
    image = f'<img src="{esc(still.file)}" alt="{esc(still.role)} frame {esc(still.frame)}">'
    return f"<figure>{image}<figcaption>{caption}<br>{hash_cell(still.framehash_sha256)}</figcaption></figure>"


def still_pair(pair: StillPairInfo) -> str:
    note = f"<p>{esc(pair.note)}</p>" if pair.note else ""
    return f'<div class="stills">{still_figure(pair.original)}{still_figure(pair.processed)}</div>{note}'


def stills_section(stills: Sequence[StillPairInfo]) -> str:
    return "" if not stills else section("Exported frames", join(map(still_pair, stills)))


def bullet_list(items: Iterable[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"


def limitations_section(report: CctvReportV1) -> str:
    return section("Limitations", bullet_list(item.text for item in report.limitations))


def warnings_section(warnings: Sequence[str]) -> str:
    return "" if not warnings else section("Warnings", bullet_list(warnings))


def footer(report: CctvReportV1) -> str:
    return f"<footer><p>{esc(report.guidelines)}</p><p><strong>{esc(report.disclaimer)}</strong></p></footer>"


def render_report_html(report: CctvReportV1) -> str:
    body = join(
        part
        for part in (
            header(report),
            case_section(report),
            steps_section(report),
            input_section(report.inputs[0]),
            acquisition_section(report.acquisition),
            osd_section(report.osd),
            trim_section(report.trim),
            clipping_section(report.clipping),
            roi_section(report.roi),
            environment_section(report.environment),
            outputs_section(report.outputs),
            stills_section(report.stills),
            limitations_section(report),
            warnings_section(report.warnings),
            footer(report),
        )
        if part
    )
    title = f"Upflow report — {esc(report.inputs[0].name)}"
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
