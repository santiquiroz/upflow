from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.config import Settings
from app.services.cctv_chain import CCTV_CHAIN, FilterSpec, StepSpec
from app.services.ffmpeg_capabilities import (
    BUILDCONF_ARGS,
    CPU_FEATURES,
    ENCODERS_ARGS,
    FILTER_UNAVAILABLE,
    FILTERS_ARGS,
    REQUIRED_ENCODERS,
    VERSION_ARGS,
    FfmpegCapabilities,
    FfmpegCapabilityCache,
    FfmpegProbeError,
    ProbeTools,
    UnavailableFilter,
    cctv_mode_available,
    cpu_extensions,
    host_cpu_extensions,
    missing_encoders,
    mode_unavailable_reason,
    parse_buildconf,
    parse_encoders,
    parse_filters,
    parse_version,
    probe,
    run_ffmpeg_text,
    unavailable_filters,
    unavailable_steps,
)
from ffmpeg_support import needs_ffmpeg

FIXTURES = Path(__file__).parent / "fixtures" / "ffmpeg"
KINDS = {"version": VERSION_ARGS, "buildconf": BUILDCONF_ARGS, "filters": FILTERS_ARGS, "encoders": ENCODERS_ARGS}

GPL_ONLY_FILTERS = frozenset(
    {
        "blackframe", "boxblur", "colormatrix", "cover_rect", "cropdetect", "delogo", "eq", "find_rect",
        "fspp", "histeq", "hqdn3d", "interlace", "kerndeint", "mcdeint", "mpdecimate", "mptestsrc", "nnedi",
        "owdenoise", "perspective", "phase", "pp7", "pullup", "repeatfields", "sab", "signature",
        "smartblur", "spp", "stereo3d", "super2xsai", "tinterlace", "uspp", "vaguedenoiser", "w3fdif",
        "vidstabdetect", "vidstabtransform", "frei0r", "frei0r_src", "rubberband",
    }
)
GPL_ONLY_ENCODERS = frozenset({"libx264", "libx264rgb", "libx265", "libxvid", "libxavs2"})
GPL_CONFIGURE_FLAGS = (
    "--enable-gpl", "--enable-libx264", "--enable-libx265", "--enable-libxvid", "--enable-libxavs2",
    "--enable-libdavs2", "--enable-libvidstab", "--enable-frei0r", "--enable-librubberband",
)


def fixture_text(kind: str, build: str) -> str:
    return (FIXTURES / f"{kind}_{build}.txt").read_text(encoding="utf-8")


def listed_name(line: str) -> str | None:
    parts = line.split()
    return parts[1] if len(parts) > 1 else None


def drop_listed(text: str, names: frozenset[str]) -> str:
    return "".join(line for line in text.splitlines(keepends=True) if listed_name(line) not in names)


def drop_inline_flags(line: str) -> str:
    kept = line
    for flag in GPL_CONFIGURE_FLAGS:
        kept = re.sub(rf" {re.escape(flag)}(?=\s)", "", kept)
    return kept


def drop_flags(text: str) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(drop_inline_flags(line) for line in lines if line.strip() not in GPL_CONFIGURE_FLAGS)


LGPL_EDITS = {
    "version": drop_flags,
    "buildconf": drop_flags,
    "filters": lambda text: drop_listed(text, GPL_ONLY_FILTERS),
    "encoders": lambda text: drop_listed(text, GPL_ONLY_ENCODERS),
}


def record_fixtures(binary: Path) -> None:
    for kind, args in KINDS.items():
        gpl = run_ffmpeg_text(binary, args).replace("\r\n", "\n")
        (FIXTURES / f"{kind}_gpl.txt").write_text(gpl, encoding="utf-8", newline="\n")
        (FIXTURES / f"{kind}_lgpl.txt").write_text(LGPL_EDITS[kind](gpl), encoding="utf-8", newline="\n")


class RecordedRunner:
    def __init__(self, build: str) -> None:
        self.build = build
        self.calls: list[tuple[Path, tuple[str, ...]]] = []

    def __call__(self, binary: Path, args: Sequence[str]) -> str:
        self.calls.append((binary, tuple(args)))
        kind = next(name for name, known in KINDS.items() if tuple(args) == known)
        return fixture_text(kind, self.build)


class CountingHash:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, path: Path) -> str:
        self.calls += 1
        return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_binary(tmp_path: Path, name: str = "ffmpeg.exe", content: bytes = b"fake ffmpeg") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def tools_for(build: str, hash_file: CountingHash | None = None) -> ProbeTools:
    return ProbeTools(
        run=RecordedRunner(build),
        hash_file=hash_file or CountingHash(),
        cpu_extensions=lambda: ("sse2", "avx2"),
    )


def probed(tmp_path: Path, build: str) -> FfmpegCapabilities:
    return probe(fake_binary(tmp_path), tools_for(build))


def capabilities(filters: set[str], encoders: frozenset[str] = frozenset(REQUIRED_ENCODERS)) -> FfmpegCapabilities:
    return FfmpegCapabilities(
        binary_sha256="0" * 64,
        version="ffmpeg version test",
        configuration=(),
        filters=frozenset(filters),
        encoders=frozenset(encoders),
        cpu_extensions=(),
    )


def every_catalog_filter() -> set[str]:
    return {name for step in CCTV_CHAIN for spec in step.filters for name in spec.ffmpeg_filters}


def test_version_is_the_first_line_of_dash_version() -> None:
    assert parse_version(fixture_text("version", "gpl")) == (
        "ffmpeg version N-123588-g9c63742425-20260323 Copyright (c) 2000-2026 the FFmpeg developers"
    )


def test_version_of_empty_output_is_unknown() -> None:
    assert parse_version("\n\n") == "unknown"


def test_buildconf_lists_each_configure_flag() -> None:
    flags = parse_buildconf(fixture_text("buildconf", "gpl"))

    assert flags[0] == "--prefix=/ffbuild/prefix"
    assert "--enable-gpl" in flags
    assert "--enable-libx264" in flags
    assert all(flag.startswith("--") for flag in flags)


def test_the_lgpl_buildconf_has_no_gpl_flag() -> None:
    flags = parse_buildconf(fixture_text("buildconf", "lgpl"))

    assert "--enable-gpl" not in flags
    assert "--enable-libx264" not in flags
    assert "--enable-version3" in flags


def test_filters_parser_skips_the_legend() -> None:
    filters = parse_filters(fixture_text("filters", "gpl"))

    assert {"hqdn3d", "fspp", "spp", "pp7", "uspp", "eq", "deblock", "crop", "overlay"} <= filters
    assert not {"=", "T..", "Timeline", "------"} & filters


def test_encoders_parser_skips_the_legend() -> None:
    encoders = parse_encoders(fixture_text("encoders", "gpl"))

    assert {"ffv1", "libx264", "a64multi"} <= encoders
    assert not {"=", "V.....", "Video", "------"} & encoders


def test_the_gpl_build_offers_every_catalog_filter() -> None:
    assert every_catalog_filter() <= parse_filters(fixture_text("filters", "gpl"))


def test_the_lgpl_fixture_lacks_hqdn3d_and_libx264(tmp_path: Path) -> None:
    caps = probed(tmp_path, "lgpl")

    assert not caps.is_gpl
    assert not caps.has_filter("hqdn3d")
    assert "libx264" not in caps.encoders
    assert "ffv1" in caps.encoders
    assert caps.has_filter("deblock")


def test_probe_asks_the_binary_for_version_buildconf_filters_and_encoders(tmp_path: Path) -> None:
    binary = fake_binary(tmp_path)
    tools = tools_for("gpl")

    caps = probe(binary, tools)

    assert tools.run.calls == [(binary, args) for args in KINDS.values()]
    assert caps.is_gpl
    assert caps.cpu_extensions == ("sse2", "avx2")


def test_probe_records_the_sha256_of_the_binary(tmp_path: Path) -> None:
    binary = fake_binary(tmp_path, content=b"ffmpeg bytes")

    caps = probe(binary, ProbeTools(run=RecordedRunner("gpl"), cpu_extensions=tuple))

    assert caps.binary_sha256 == hashlib.sha256(b"ffmpeg bytes").hexdigest()


def test_probe_of_a_missing_binary_fails_with_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "ffmpeg.exe"

    with pytest.raises(FfmpegProbeError, match="nope"):
        probe(missing, tools_for("gpl"))


def test_the_gpl_build_disables_nothing(tmp_path: Path) -> None:
    caps = probed(tmp_path, "gpl")

    assert unavailable_filters(caps) == ()
    assert unavailable_steps(caps) == ()
    assert cctv_mode_available(caps)
    assert mode_unavailable_reason(caps) is None


def test_the_lgpl_build_disables_the_gpl_filters_with_a_reason(tmp_path: Path) -> None:
    disabled = unavailable_filters(probed(tmp_path, "lgpl"))

    assert {(item.step_id, item.filter) for item in disabled} == {
        ("denoise", "hqdn3d"),
        ("deblock", "fspp"),
        ("deblock", "spp"),
        ("deblock", "pp7"),
        ("deblock", "uspp"),
        ("levels", "eq"),
        ("stabilize", "vidstab"),
    }
    assert all(item.reason_key == FILTER_UNAVAILABLE for item in disabled)


def test_the_reason_names_the_missing_ffmpeg_filter() -> None:
    item = UnavailableFilter("denoise", "hqdn3d", ("hqdn3d",))

    assert item.reason_key == "cctv.filterUnavailable"
    assert "hqdn3d" in item.message
    assert item.to_json() == {
        "stepId": "denoise",
        "filter": "hqdn3d",
        "missing": ["hqdn3d"],
        "reasonKey": "cctv.filterUnavailable",
        "reason": item.message,
    }


def test_the_lgpl_build_only_loses_stabilize_whose_only_filter_is_gpl(tmp_path: Path) -> None:
    assert unavailable_steps(probed(tmp_path, "lgpl")) == ("stabilize",)


def test_a_step_is_disabled_when_none_of_its_filters_exist() -> None:
    denoisers = {"hqdn3d", "atadenoise", "tmedian", "fftdnoiz", "nlmeans", "bm3d"}
    caps = capabilities(every_catalog_filter() - denoisers)

    assert unavailable_steps(caps) == ("denoise",)


def test_a_filter_needing_two_ffmpeg_filters_names_only_the_missing_one() -> None:
    caps = capabilities(every_catalog_filter() - {"overlay"})

    osd = [item for item in unavailable_filters(caps) if item.step_id == "osd_protect"]

    assert osd == [UnavailableFilter("osd_protect", "osd_restore", ("overlay",))]


def test_steps_not_offered_in_any_lane_are_not_reported() -> None:
    caps = capabilities(every_catalog_filter() - {"lenscorrection", "v360"})

    assert unavailable_filters(caps) == ()


def test_a_build_without_libvidstab_disables_stabilize() -> None:
    caps = capabilities(every_catalog_filter() - {"vidstabtransform"})

    assert unavailable_filters(caps) == (UnavailableFilter("stabilize", "vidstab", ("vidstabtransform",)),)


def test_ai_filters_do_not_depend_on_ffmpeg() -> None:
    ai_only = StepSpec("ai_deblock", "AI", "ai", (FilterSpec("drunet_deblock", "AI"),), frozenset({"ai"}))

    assert unavailable_steps(capabilities(set()), (ai_only,)) == ()


def test_the_lgpl_build_cannot_run_cctv_mode_and_says_why(tmp_path: Path) -> None:
    caps = probed(tmp_path, "lgpl")

    assert missing_encoders(caps) == ("libx264",)
    assert not cctv_mode_available(caps)
    assert "libx264" in mode_unavailable_reason(caps)


def test_a_build_without_ffv1_cannot_run_cctv_mode() -> None:
    caps = capabilities(every_catalog_filter(), frozenset({"libx264"}))

    assert missing_encoders(caps) == ("ffv1",)
    assert "ffv1" in mode_unavailable_reason(caps)


def test_cpu_extensions_keep_the_catalog_order() -> None:
    present = {CPU_FEATURES["avx2"], CPU_FEATURES["sse2"]}

    assert cpu_extensions(lambda feature: feature in present) == ("sse2", "avx2")


def test_cpu_extensions_without_a_probe_are_empty() -> None:
    assert cpu_extensions(None) == ()


def test_cpu_features_use_the_windows_processor_feature_ids() -> None:
    assert CPU_FEATURES["avx2"] == 40
    assert CPU_FEATURES["avx512f"] == 41


@pytest.mark.skipif(sys.platform != "win32", reason="IsProcessorFeaturePresent only exists on Windows")
def test_the_host_reports_sse2_on_x64_windows() -> None:
    found = host_cpu_extensions()

    assert "sse2" in found
    assert set(found) <= set(CPU_FEATURES)


def test_report_has_what_the_forensic_report_needs(tmp_path: Path) -> None:
    report = probed(tmp_path, "lgpl").to_report()

    assert report["binarySha256"] == hashlib.sha256(b"fake ffmpeg").hexdigest()
    assert report["version"].startswith("ffmpeg version N-123588")
    assert report["gpl"] is False
    assert report["cpuExtensions"] == ["sse2", "avx2"]
    assert "--enable-version3" in report["configuration"]


def test_cache_probes_a_binary_once(tmp_path: Path) -> None:
    binary = fake_binary(tmp_path)
    hashes = CountingHash()
    tools = tools_for("gpl", hashes)
    cache = FfmpegCapabilityCache(tools)

    first = cache.get(binary)
    second = cache.get(binary)

    assert first is second
    assert len(tools.run.calls) == len(KINDS)
    assert hashes.calls == 1


def test_cache_is_keyed_by_the_sha256_not_the_path(tmp_path: Path) -> None:
    tools = tools_for("gpl")
    cache = FfmpegCapabilityCache(tools)

    cache.get(fake_binary(tmp_path, "a.exe"))
    cache.get(fake_binary(tmp_path, "b.exe"))

    assert len(tools.run.calls) == len(KINDS)


def test_cache_probes_again_when_the_binary_changes(tmp_path: Path) -> None:
    binary = fake_binary(tmp_path, content=b"build one")
    tools = tools_for("gpl")
    cache = FfmpegCapabilityCache(tools)
    first = cache.get(binary)

    binary.write_bytes(b"build number two")
    second = cache.get(binary)

    assert first.binary_sha256 != second.binary_sha256
    assert len(tools.run.calls) == 2 * len(KINDS)


def test_run_ffmpeg_text_returns_stdout() -> None:
    assert run_ffmpeg_text(Path(sys.executable), ("-c", "print('filters here')")).strip() == "filters here"


def test_run_ffmpeg_text_fails_on_a_non_zero_exit() -> None:
    with pytest.raises(FfmpegProbeError, match="exit code 3"):
        run_ffmpeg_text(Path(sys.executable), ("-c", "import sys; sys.exit(3)"))


def test_run_ffmpeg_text_fails_when_the_binary_cannot_start(tmp_path: Path) -> None:
    with pytest.raises(FfmpegProbeError):
        run_ffmpeg_text(tmp_path / "missing.exe", VERSION_ARGS)


@needs_ffmpeg
def test_the_real_binary_matches_the_recorded_gpl_fixture() -> None:
    caps = probe(Settings().ffmpeg_binary_path)

    assert caps.version.startswith("ffmpeg version")
    assert set(REQUIRED_ENCODERS) <= caps.encoders
    if caps.version != parse_version(fixture_text("version", "gpl")):
        pytest.skip(f"vendored ffmpeg changed ({caps.version}); re-record tests/fixtures/ffmpeg/*_gpl.txt")
    assert caps.filters == parse_filters(fixture_text("filters", "gpl"))
    assert caps.encoders == parse_encoders(fixture_text("encoders", "gpl"))
    assert caps.configuration == parse_buildconf(fixture_text("buildconf", "gpl"))
