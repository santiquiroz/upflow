from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import pytest

from app.config import Settings
from app.services import cctv_clarify_runner as runner
from app.services import handover_package as package
from app.services.cctv_chain import steps_from_request
from app.services.cctv_ingest import (
    IngestTools,
    MediaTools,
    ReceivedAt,
    RemuxAttempt,
    SourceRecord,
    WorkingCopy,
    ingest_source,
    ingest_working_copy,
    make_verified_copy,
)
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.ffmpeg_filters import FrameGeometry
from app.services.media_signature import MATROSKA
from ffmpeg_support import needs_ffmpeg

JOB_ID = "0123abcd"
DAY = date(2026, 9, 25)
ORIGINAL = "cámara 1 & (50%).mp4"
CAPS = FfmpegCapabilities(
    binary_sha256="ab" * 32,
    version="ffmpeg version N-123588-g9c63742425-20260323",
    configuration=("--enable-gpl",),
    filters=frozenset(),
    encoders=frozenset(),
    cpu_extensions=("sse4_2", "avx2"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(name: str = ORIGINAL, container=MATROSKA) -> SourceRecord:
    return SourceRecord(name, 10, "2026-09-25T10:00:00.000Z", "cd" * 32, ReceivedAt("u", "l"), container)


def write(path: Path, data: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def fake_job(root: Path) -> Path:
    job = root / "job"
    write(job / "01_original" / ORIGINAL, b"original bytes")
    write(job / "02_processed" / "camara 1 _ _50____upflow-clarify__0123abcd.mkv", b"ffv1")
    write(job / "02_processed" / "camara 1 _ _50____upflow-clarify__0123abcd.mp4", b"x264")
    write(job / "03_comparisons" / "comparison.mp4", b"side by side")
    write(job / "04_stills" / "processed_f7.png", b"p7")
    write(job / "04_stills" / "original_f7.png", b"o7")
    for name in ("frame_index.csv", "report.json", "report.html", "reproduce.cmd", "README.txt"):
        write(job / name, name.encode())
    write(job / "work.mkv", b"session leftovers")
    write(job / "reproduced" / "analysis.mkv", b"a recipient's rebuild")
    write(job / "notes.tmp", b"junk")
    return job


# --- Nombres ---


def test_processed_names_are_derived_and_never_repeat_the_original_name() -> None:
    mkv = package.processed_name(ORIGINAL, JOB_ID, ".mkv")

    assert mkv == "camara 1 _ _50____upflow-clarify__0123abcd.mkv"
    assert mkv != ORIGINAL and mkv.isascii()
    assert package.processed_name("CAM01.dav", JOB_ID, ".mp4") == "CAM01__upflow-clarify__0123abcd.mp4"


def test_a_stem_with_nothing_ascii_left_still_gets_a_name() -> None:
    assert package.processed_name("监控.mp4", JOB_ID, ".mkv") == "clip__upflow-clarify__0123abcd.mkv"


@pytest.mark.parametrize("job_id", ["", "../x", "a b", "a\\b", "x" * 65])
def test_processed_names_refuse_unsafe_job_ids(job_id: str) -> None:
    with pytest.raises(ValueError, match="job id"):
        package.processed_name(ORIGINAL, job_id, ".mkv")


def test_the_package_root_uses_the_case_label_and_the_date() -> None:
    assert package.package_root_name("Caso 12/2026: robo", ORIGINAL, DAY) == "Caso 12_2026_ robo_2026-09-25_upflow"


@pytest.mark.parametrize("label", [None, "", "   ", "..."])
def test_the_package_root_falls_back_to_the_clip_name(label: str | None) -> None:
    assert package.package_root_name(label, ORIGINAL, DAY) == "cámara 1 & (50%)_2026-09-25_upflow"


def test_a_long_case_label_is_shortened() -> None:
    root = package.package_root_name("x" * 300, ORIGINAL, DAY)

    assert root == "x" * package.MAX_LABEL_CHARS + "_2026-09-25_upflow"


def test_place_processed_moves_both_copies_under_02_processed(tmp_path: Path) -> None:
    analysis = write(tmp_path / "out" / "analysis.mkv", b"ffv1")
    viewing = write(tmp_path / "out" / "viewing.mp4", b"x264")
    job = tmp_path / "job"

    placed = package.place_processed(job, ORIGINAL, JOB_ID, analysis, viewing)

    assert placed.analysis == job / "02_processed" / "camara 1 _ _50____upflow-clarify__0123abcd.mkv"
    assert placed.viewing == job / "02_processed" / "camara 1 _ _50____upflow-clarify__0123abcd.mp4"
    assert placed.analysis.read_bytes() == b"ffv1" and not analysis.exists() and not viewing.exists()


def test_place_processed_keeps_the_stabilization_motion_file_next_to_the_copies(tmp_path: Path) -> None:
    out = tmp_path / "out"
    analysis, viewing = write(out / "analysis.mkv", b"ffv1"), write(out / "viewing.mp4", b"x264")
    transforms = write(out / "transforms.trf", b"VID.STAB 1\n")
    job = tmp_path / "job"

    placed = package.place_processed(job, ORIGINAL, JOB_ID, analysis, viewing, transforms)

    assert placed.transforms == job / "02_processed" / "camara 1 _ _50____upflow-clarify__0123abcd.trf"
    assert placed.transforms.read_bytes() == b"VID.STAB 1\n" and not transforms.exists()
    assert package.place_processed(job, ORIGINAL, "x", write(out / "a.mkv", b""), write(out / "v.mp4", b"")).transforms is None


# --- Contenido del paquete ---


def test_package_members_follow_the_layout_and_skip_everything_else(tmp_path: Path) -> None:
    job = fake_job(tmp_path)
    write(job / "SHA256SUMS.txt", b"sums")

    names = [member.arcname for member in package.package_members(job)]

    assert names == [
        f"01_original/{ORIGINAL}",
        "02_processed/camara 1 _ _50____upflow-clarify__0123abcd.mkv",
        "02_processed/camara 1 _ _50____upflow-clarify__0123abcd.mp4",
        "03_comparisons/comparison.mp4",
        "04_stills/original_f7.png",
        "04_stills/processed_f7.png",
        "frame_index.csv",
        "report.json",
        "report.html",
        "SHA256SUMS.txt",
        "reproduce.cmd",
        "README.txt",
    ]


def test_checksums_cover_every_member_but_the_list_itself(tmp_path: Path) -> None:
    job = fake_job(tmp_path)

    paths = package.checksum_paths(job)

    assert "SHA256SUMS.txt" not in paths
    assert f"01_original/{ORIGINAL}" in paths and "reproduce.cmd" in paths and "report.html" in paths
    assert "work.mkv" not in paths and "notes.tmp" not in paths and not any(p.startswith("reproduced") for p in paths)


def test_finalize_writes_reproduce_readme_and_sums_in_that_order(tmp_path: Path) -> None:
    job = fake_job(tmp_path)
    facts = package.readme_facts(record(), "classic", CAPS)

    sums = package.finalize_package_files(job, "root", facts, "@echo off\r\n")

    assert (job / "reproduce.cmd").read_bytes() == b"@echo off\r\n"
    listed = dict(reversed(line.split(" *", 1)) for line in sums.read_text(encoding="utf-8").splitlines())
    assert listed["reproduce.cmd"] == sha256(job / "reproduce.cmd")
    assert listed["README.txt"] == sha256(job / "README.txt")
    assert listed[f"01_original/{ORIGINAL}"] == sha256(job / "01_original" / ORIGINAL)


def test_the_ai_lane_package_carries_no_reproduce_script(tmp_path: Path) -> None:
    job = fake_job(tmp_path)
    (job / "reproduce.cmd").unlink()

    package.finalize_package_files(job, "root", package.readme_facts(record(), "ai-visual", CAPS), None)

    assert not (job / "reproduce.cmd").exists()
    assert "reproduce.cmd" not in (job / "SHA256SUMS.txt").read_text(encoding="utf-8")


# --- README.txt ---


def readme(mode: str = "classic") -> str:
    return package.readme_text(package.readme_facts(record(), mode, CAPS), "Caso_2026-09-25_upflow")


def test_readme_is_bilingual_and_says_what_the_hashes_do_not_prove() -> None:
    text = readme()

    assert "they do not prove the recording is authentic or say what happened before it was loaded" in text
    assert "no prueban que la grabación sea auténtica ni dicen qué pasó antes de cargarla" in text
    assert "Anyone who changes the files can also rewrite SHA256SUMS.txt." in text
    assert "Quien altere los archivos también puede regenerar SHA256SUMS.txt." in text


def test_readme_says_the_original_is_01_original_and_how_to_check_it() -> None:
    text = readme()

    assert f"01_original\\{ORIGINAL}" in text and "cd" * 32 in text
    assert "The original recording is the file in 01_original" in text
    assert "La grabación original es el archivo de 01_original" in text
    assert f"Get-FileHash -Algorithm SHA256 -LiteralPath '01_original\\{ORIGINAL}'" in text
    assert "sha256sum -c SHA256SUMS.txt" in text


def test_readme_explains_reproduce_cmd_and_the_ffmpeg_build_it_needs() -> None:
    text = readme()

    assert "reproduce.cmd" in text and 'set "FFMPEG=' in text
    assert CAPS.version in text and CAPS.binary_sha256 in text and "sse4_2 avx2" in text


def test_readme_of_the_ai_lane_has_no_reproduce_steps_and_says_why() -> None:
    text = readme("ai-visual")

    assert 'set "FFMPEG=' not in text and "reproduce.cmd     " not in text and "lossless" not in text
    assert "cannot be reproduced bit for bit" in text and "no se pueden reproducir bit a bit" in text


def test_readme_carries_the_handover_checklist_guidelines_and_disclaimer() -> None:
    text = readme()

    assert "Processed version — not the original recording" in text
    assert "Versión procesada — no es la grabación original" in text
    assert "clock offset" in text and "desfase del reloj" in text
    assert "not a certified forensic tool" in text and "no es una herramienta forense certificada" in text
    assert "SWGDE" in text


def test_readme_never_calls_the_package_evidence() -> None:
    assert "evidence package" not in readme().lower()


# --- reproduce.cmd del paquete ---


def working_copy(path: Path, method: str = "auto", input_args: tuple[str, ...] = ()) -> WorkingCopy:
    attempts = (
        RemuxAttempt("auto", (), ok=method == "auto", detail="ok" if method == "auto" else "failed"),
        *((RemuxAttempt(method, input_args, ok=True, detail="ok"),) if method != "auto" else ()),
    )
    return WorkingCopy(path, "ef" * 32, method, attempts, copytb=False)


def reproduce_script(method: str = "auto", input_args: tuple[str, ...] = ()) -> str:
    verified = Path("C:/runtime/outputs/job/01_original") / ORIGINAL
    work = Path("C:/runtime/video-work/cctv-t/work.mkv")
    analysis = Path("C:/runtime/outputs/job/02_processed/analysis.mkv")
    argv = ("C:/vendor/ffmpeg.exe", "-y", "-i", str(work), "-c:v", "ffv1", str(analysis))
    return package.package_reproduce_script(
        package.ReproduceSources(verified, working_copy(work, method, input_args)),
        (runner.ReproduceStep("analysis copy (FFV1)", argv),),
        (package.ReproducedOutput(analysis, "02_processed/clip__upflow-clarify__0123abcd.mkv"),),
        CAPS,
    )


def test_reproduce_first_rebuilds_the_working_copy_from_01_original() -> None:
    lines = reproduce_script().splitlines()

    remux = next(line for line in lines if "+genpts" in line)
    assert f'"-i" "01_original\\{ORIGINAL.replace("%", "%%")}"' in remux
    assert '"-c" "copy" "reproduced\\work.mkv"' in remux
    analysis = next(line for line in lines if '"ffv1"' in line)
    assert '"-i" "reproduced\\work.mkv"' in analysis and analysis.endswith('"reproduced\\clip__upflow-clarify__0123abcd.mkv"')
    assert lines.index(remux) < lines.index(analysis)


def test_reproduce_uses_the_demuxer_that_worked_during_ingest() -> None:
    remux = next(line for line in reproduce_script("dhav", ("-f", "dhav")).splitlines() if "+genpts" in line)

    assert '"-f" "dhav" "-i"' in remux


def test_reproduce_checks_each_output_against_its_listed_name_and_holds_no_machine_paths() -> None:
    text = reproduce_script()

    assert 'set "PRODUCED=reproduced\\clip__upflow-clarify__0123abcd.mkv"' in text
    assert 'set "LISTED=02_processed/clip__upflow-clarify__0123abcd.mkv"' in text
    assert "C:/runtime" not in text and "C:\\runtime" not in text and "C:/vendor" not in text


def test_reproduce_refuses_non_ascii_listed_names() -> None:
    # findstr compara bytes: un nombre no ASCII nunca coincidiria con la linea de SHA256SUMS.txt.
    with pytest.raises(ValueError, match="ASCII"):
        package.ReproducedOutput(Path("x.mkv"), "02_processed/cámara.mkv")


# --- ZIP ---


def members_of(zip_path: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.infolist()


def test_the_package_is_an_uncompressed_zip_under_one_root_folder(tmp_path: Path) -> None:
    job = fake_job(tmp_path)
    package.finalize_package_files(job, "root", package.readme_facts(record(), "classic", CAPS), "@echo off\r\n")
    destination = tmp_path / "downloads" / "Caso_2026-09-25_upflow.zip"

    outcome = package.build_handover_package(job, destination, "Caso_2026-09-25_upflow")

    assert outcome.path == destination and outcome.warnings == ()
    infos = members_of(destination)
    assert {info.compress_type for info in infos} == {zipfile.ZIP_STORED}
    assert all(info.filename.startswith("Caso_2026-09-25_upflow/") for info in infos)
    with zipfile.ZipFile(destination) as archive:
        assert archive.read(f"Caso_2026-09-25_upflow/01_original/{ORIGINAL}") == b"original bytes"
    assert not destination.with_name(destination.name + ".part").exists()


def test_the_zip_allows_zip64(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = fake_job(tmp_path)
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 4)

    outcome = package.build_handover_package(job, tmp_path / "p.zip", "root")

    assert outcome.path is not None and members_of(outcome.path)


def test_without_disk_room_there_is_no_zip_and_a_warning(tmp_path: Path) -> None:
    job = fake_job(tmp_path)
    destination = tmp_path / "p.zip"

    outcome = package.build_handover_package(job, destination, "root", free_bytes=lambda path: 100)

    assert outcome.path is None and outcome.warnings == (package.NO_DISK_ROOM_WARNING,)
    assert not destination.exists()


def test_a_failed_write_leaves_no_partial_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = fake_job(tmp_path)
    destination = tmp_path / "p.zip"

    def broken(self, *args, **kwargs):
        raise OSError("disk went away")

    monkeypatch.setattr(zipfile.ZipFile, "write", broken)
    with pytest.raises(OSError, match="disk went away"):
        package.build_handover_package(job, destination, "root")

    assert not destination.exists() and not destination.with_name("p.zip.part").exists()


# --- Check files are unchanged ---


def finalized_job(tmp_path: Path) -> Path:
    job = fake_job(tmp_path)
    package.finalize_package_files(job, "root", package.readme_facts(record(), "classic", CAPS), "@echo off\r\n")
    return job


def test_unchanged_files_pass(tmp_path: Path) -> None:
    job = finalized_job(tmp_path)

    check = package.check_files_unchanged(job)

    assert check.ok and check.checked == 11 and check.mismatches == () and check.missing == ()
    assert check.to_json() == {"ok": True, "checked": 11, "mismatches": [], "missing": []}


def test_a_changed_file_and_a_missing_file_are_listed(tmp_path: Path) -> None:
    job = finalized_job(tmp_path)
    write(job / "04_stills" / "original_f7.png", b"edited")
    (job / "03_comparisons" / "comparison.mp4").unlink()

    check = package.check_files_unchanged(job)

    assert not check.ok and check.checked == 10
    assert check.mismatches == ("04_stills/original_f7.png",)
    assert check.missing == ("03_comparisons/comparison.mp4",)


def test_without_sha256sums_nothing_can_be_checked(tmp_path: Path) -> None:
    check = package.check_files_unchanged(fake_job(tmp_path))

    assert not check.ok and check.checked == 0 and check.missing == ("SHA256SUMS.txt",)


@pytest.mark.parametrize("line", ["not a checksum line", f"{'a' * 64} *../outside.txt", f"{'a' * 64} *C:/x.txt"])
def test_a_tampered_checksum_list_is_itself_a_mismatch(tmp_path: Path, line: str) -> None:
    job = finalized_job(tmp_path)
    with (job / "SHA256SUMS.txt").open("a", encoding="utf-8") as sums:
        sums.write(line + "\n")

    check = package.check_files_unchanged(job)

    assert not check.ok and check.mismatches == ("SHA256SUMS.txt",) and check.checked == 0


def test_an_emptied_checksum_list_is_itself_a_mismatch(tmp_path: Path) -> None:
    job = finalized_job(tmp_path)
    (job / "SHA256SUMS.txt").write_text("\n", encoding="utf-8")

    assert package.check_files_unchanged(job).mismatches == ("SHA256SUMS.txt",)


def test_reproduce_needs_a_successful_remux_on_record() -> None:
    working = WorkingCopy(Path("w.mkv"), "ef" * 32, "auto", (RemuxAttempt("auto", (), ok=False, detail="x"),), False)

    with pytest.raises(ValueError, match="remux"):
        package.remux_step(package.ReproduceSources(Path("o.mp4"), working))


# --- Con ffmpeg real ---


def settings_tools() -> tuple[MediaTools, runner.ClarifyTools]:
    settings = Settings()
    ffmpeg, ffprobe = settings.ffmpeg_binary_path, settings.ffprobe_binary_path
    return MediaTools(ffmpeg=ffmpeg, ffprobe=ffprobe), runner.ClarifyTools(ffmpeg, ffprobe)


def make_clip(path: Path) -> Path:
    command = [
        str(Settings().ffmpeg_binary_path), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25,noise=alls=12:allf=t",
        "-f", "lavfi", "-i", "sine=sample_rate=8000",
        "-t", "2", "-threads", "1", "-c:v", "libx264", "-x264-params", "threads=1", "-b:v", "200k", "-g", "25",
        "-c:a", "pcm_alaw", "-shortest", str(path),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return path


def clarify_plan(ingested) -> runner.ClarifyPlan:
    raw = [{"id": "trim", "params": {"start_frame": 3, "end_frame": 40}}, {"id": "denoise", "params": {"filter": "hqdn3d"}}]
    return runner.ClarifyPlan(
        work=ingested.working_copy.path,
        steps=steps_from_request(raw, "classic"),
        geometry=FrameGeometry(ingested.video.width, ingested.video.height),
        source_pix_fmt="yuv420p",
        frames=ingested.index.frames,
        has_audio=bool(ingested.audio),
        app_version="9.9.9",
    )


async def classic_job(tmp_path: Path) -> tuple[Path, str]:
    media, clarify = settings_tools()
    (tmp_path / "upload").mkdir()
    clip = make_clip(tmp_path / "upload" / "cámara 1 & (50%).mkv")
    job, session = tmp_path / "job", tmp_path / "session"
    session.mkdir()
    source = ingest_source(clip, clip.name, IngestTools())
    verified = make_verified_copy(clip, source, job)
    ingested = await ingest_working_copy(media, verified.path, session, source.container)
    threads = runner.ClarifyThreads(filter_threads=4, ffv1_slices=4, x264_threads=2)
    (job / "02_processed").mkdir()
    result = await runner.run_clarify(clarify, clarify_plan(ingested), job / "02_processed", threads)
    placed = package.place_processed(job, source.original_name, JOB_ID, result.analysis, result.viewing)
    shutil.copyfile(ingested.index.csv_path, job / "frame_index.csv")
    write(job / "report.json", b"{}\n")
    write(job / "report.html", b"<!doctype html>\n")
    caps = FfmpegCapabilities(sha256(clarify.ffmpeg), "ffmpeg version test", (), frozenset(), frozenset(), ())
    outputs = (
        package.ReproducedOutput(result.analysis, placed.analysis.relative_to(job).as_posix()),
        package.ReproducedOutput(result.viewing, placed.viewing.relative_to(job).as_posix()),
    )
    script = package.package_reproduce_script(
        package.ReproduceSources(verified.path, ingested.working_copy), result.reproduce_steps(), outputs, caps
    )
    root = package.package_root_name("Caso 7", source.original_name, DAY)
    package.finalize_package_files(job, root, package.readme_facts(source, "classic", caps), script)
    return job, root


@needs_ffmpeg
@pytest.mark.skipif(sys.platform != "win32", reason="reproduce.cmd is a Windows batch file")
async def test_real_package_unzips_checks_clean_and_reproduces_the_processed_copies(tmp_path: Path) -> None:
    job, root = await classic_job(tmp_path)
    outcome = package.build_handover_package(job, tmp_path / "downloads" / f"{root}.zip", root)
    recipient = tmp_path / "recipient"
    with zipfile.ZipFile(outcome.path) as archive:
        archive.extractall(recipient)
    unpacked = recipient / root

    assert package.check_files_unchanged(unpacked).ok
    env = {**os.environ, "FFMPEG": str(Settings().ffmpeg_binary_path)}
    run = subprocess.run(["cmd", "/c", str(unpacked / "reproduce.cmd")], capture_output=True, env=env)
    stdout = run.stdout.decode("utf-8", errors="replace")

    assert run.returncode == 0, stdout + run.stderr.decode("utf-8", errors="replace")
    assert stdout.count("MATCH") == 2 and "DIFFERENT" not in stdout and "WARNING" not in stdout
    assert package.check_files_unchanged(unpacked).ok


@needs_ffmpeg
async def test_real_check_spots_a_changed_processed_copy(tmp_path: Path) -> None:
    job, _ = await classic_job(tmp_path)
    viewing = next((job / "02_processed").glob("*.mp4"))
    with viewing.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        last = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([last[0] ^ 0xFF]))

    check = package.check_files_unchanged(job)

    assert not check.ok and check.mismatches == (viewing.relative_to(job).as_posix(),)
