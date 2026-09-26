from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app import headless
from app.config import Settings
from app.services.photo_restore_presets import PhotoFacts
from app.services.restore_outputs import restore_output_paths
from app.services.restore_session import AnalysisDetectors, RestoreSessionStore
from app.services.scale_fit import final_output_path


def test_output_and_engine_formats(tmp_path):
    assert headless.output_format_for(Path("a.webp"), None) == "webp"
    assert headless.output_format_for(Path("a.any"), "jxl") == "jxl"
    with pytest.raises(headless.UsageError):
        headless.output_format_for(Path("a.tiff"), None)
    assert headless.engine_format_for("jxl") == "png"
    assert headless.engine_format_for("webp") == "webp"


def test_deterministic_job_id(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    assert headless.deterministic_job_id(source, model="m", scale=2) == headless.deterministic_job_id(
        source, model="m", scale=2
    )
    assert headless.deterministic_job_id(source, model="m", scale=2) != headless.deterministic_job_id(
        source, model="m", scale=3
    )
    assert headless.deterministic_job_id(source, model="m", scale=2).startswith("cli-")


@pytest.mark.asyncio
async def test_headless_context_model_and_device_checks(tmp_path):
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    ctx = headless.build_context(settings)
    ctx.ncnn_engine.available = lambda: False
    with pytest.raises(headless.ModelNotInstalledError):
        headless.ensure_model_installed(ctx, "realesrgan-x4plus")
    ctx.ncnn_engine.available = lambda: True
    headless.ensure_model_installed(ctx, "realesrgan-x4plus")
    with pytest.raises(headless.ModelNotInstalledError):
        headless.ensure_model_installed(ctx, "nope")
    with pytest.raises(headless.DeviceError):
        await headless.resolve_device(ctx, "auto")
    with pytest.raises(headless.DeviceError):
        await headless.resolve_device(ctx, "no-such-device")


@pytest.mark.asyncio
async def test_upscale_image_with_fake_engine(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    ctx = headless.build_context(settings)
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 8), (10, 20, 30)).save(source)
    ctx.devices.validate = lambda device_id: {"id": device_id}
    ctx.ncnn_engine.available = lambda: True

    async def fake_run(job):
        produced = final_output_path(ctx.settings, job)
        produced.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 16), (10, 20, 30)).save(produced)
        job.output_path = produced
        job.metadata["effective"] = {
            "engine": "fake",
            "tileSize": 0,
            "tileOverlap": 10,
            "tileSizeMeaning": "auto",
            "resized": True,
        }
        return produced

    ctx.job_manager.run_inline = fake_run
    output = tmp_path / "out" / "result.png"
    result = await headless.upscale_image(ctx, source, output, scale=2, device="dml:0")
    assert result["ok"] is True
    assert result["width"] == 32
    assert result["height"] == 16
    assert result["scale"] == 2
    assert result["nativeScale"] == 4
    assert result["tile"]["size"] == 0
    assert output.exists()
    assert not any(settings.outputs_path.iterdir())
    assert result["output"] == str(output.resolve())

    async def fake_run_check(job):
        assert job.native_scale == 4
        assert job.scale == 2
        assert job.id.startswith("cli-")
        return await fake_run(job)

    ctx.job_manager.run_inline = fake_run_check
    await headless.upscale_image(ctx, source, tmp_path / "out" / "second.png", scale=2, device="dml:0")


@pytest.mark.asyncio
async def test_upscale_image_jxl_and_invalid_inputs(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    ctx = headless.build_context(settings)
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 8)).save(source)
    ctx.devices.validate = lambda device_id: {"id": device_id}
    ctx.ncnn_engine.available = lambda: True

    async def fake_run(job):
        produced = final_output_path(ctx.settings, job)
        produced.parent.mkdir(parents=True, exist_ok=True)
        source.copy_to(produced) if hasattr(source, "copy_to") else produced.write_bytes(source.read_bytes())
        job.output_path = produced
        job.metadata["effective"] = {"tileSize": 0, "tileOverlap": 10, "resized": False}
        return produced

    ctx.job_manager.run_inline = fake_run
    monkeypatch.setattr(headless, "convert_with_ffmpeg", lambda settings, source, target, fmt: target.write_bytes(source.read_bytes()))
    result = await headless.upscale_image(ctx, source, tmp_path / "result.jxl", scale=2, device="dml:0")
    assert result["format"] == "jxl"
    assert (tmp_path / "result.jxl").exists()
    with pytest.raises(headless.UsageError):
        await headless.upscale_image(ctx, tmp_path / "missing.png", tmp_path / "out.png")
    with pytest.raises(headless.UsageError):
        await headless.upscale_image(ctx, source, tmp_path / "bad.png", scale=9, device="dml:0")


@pytest.mark.asyncio
async def test_ffmpeg_failure_leaves_no_engine_output_behind(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    ctx = headless.build_context(settings)
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 8)).save(source)
    ctx.devices.validate = lambda device_id: {"id": device_id}
    ctx.ncnn_engine.available = lambda: True

    async def fake_run(job):
        produced = final_output_path(ctx.settings, job)
        produced.parent.mkdir(parents=True, exist_ok=True)
        produced.write_bytes(source.read_bytes())
        job.output_path = produced
        job.metadata["effective"] = {"tileSize": 0, "tileOverlap": 10}
        return produced

    ctx.job_manager.run_inline = fake_run

    def failing_convert(settings, source_png, target, fmt):
        raise headless.InferenceError("ffmpeg could not encode jxl: boom")

    monkeypatch.setattr(headless, "convert_with_ffmpeg", failing_convert)

    with pytest.raises(headless.InferenceError, match="boom"):
        await headless.upscale_image(ctx, source, tmp_path / "result.jxl", scale=4, device="dml:0")

    assert not any(settings.outputs_path.iterdir())
    assert not (tmp_path / "result.jxl").exists()


# ---------------------------------------------------------------- restore


def test_restore_preset_options_fill_only_the_chosen_steps():
    options = headless.preset_step_options("gentle", ("denoise", "tone"))
    assert set(options) == {"denoise", "tone"}
    assert options["denoise"]["strength"] == 0.3
    with pytest.raises(headless.UsageError):
        headless.preset_step_options("nope", ("denoise",))


def test_restore_options_merge_keeps_preset_values_the_user_did_not_touch():
    merged = headless.merge_restore_options(
        {"denoise": {"strength": 0.3, "keep_grain": 0.25}},
        {"denoise": {"strength": 0.6}, "upscale_mode": "classic"},
    )
    assert merged == {"denoise": {"strength": 0.6, "keep_grain": 0.25}, "upscale_mode": "classic"}


def test_restore_plan_from_steps_uses_the_preset_as_defaults():
    spec = headless.RestoreSpec(steps=("tone", "denoise"), preset="gentle", options={"tone": {"strength": 0.5}})
    plan = headless.plan_from_steps(spec)
    assert plan.steps == ("tone", "denoise")
    assert plan.options["tone"]["strength"] == 0.5
    assert plan.options["tone"]["fix_faded"] is True
    assert plan.options["preset"] == "gentle"
    assert plan.preset == "gentle"


def test_restore_plan_from_analysis_resolves_the_proposed_preset():
    analysis = SimpleNamespace(diagnosis=SimpleNamespace(facts=PhotoFacts(halftone=True), proposed_preset="newspaper"))
    plan = headless.plan_from_analysis(headless.RestoreSpec(options={"denoise": {"strength": 0.1}}), analysis)
    assert plan.preset == "newspaper"
    assert plan.steps == ("descreen", "denoise")
    assert plan.options["denoise"] == {"strength": 0.1, "keep_grain": 0.25}


def test_restore_plan_from_analysis_with_nothing_to_fix_is_a_usage_error():
    analysis = SimpleNamespace(diagnosis=SimpleNamespace(facts=PhotoFacts(), proposed_preset="gentle"))
    with pytest.raises(headless.UsageError, match="--steps"):
        headless.plan_from_analysis(headless.RestoreSpec(), analysis)


def test_restore_options_reject_unknown_fields_and_preview_crop():
    assert headless.validated_restore_options({"tone": {"strength": 0.4}}) == {"tone": {"strength": 0.4}}
    with pytest.raises(headless.UsageError):
        headless.validated_restore_options({"tone": {"strenght": 0.4}})
    with pytest.raises(headless.UsageError, match="preview"):
        headless.validated_restore_options({"preview_crop": [0, 0, 8, 8]})


def test_restore_format_only_accepts_what_the_restorer_writes():
    assert headless.restore_format_for(Path("a.jpeg"), None) == "jpeg"
    assert headless.restore_format_for(Path("a.any"), "webp") == "webp"
    with pytest.raises(headless.UsageError):
        headless.restore_format_for(Path("a.jxl"), None)


def test_relocated_sidecar_names_the_delivered_files_only():
    sidecar = {
        "outputs": [
            {"role": "restored", "file": "cli-1.png", "sha256": "a"},
            {"role": "uncolored", "file": "cli-1.uncolored.png", "sha256": "b"},
            {"role": "view", "file": "cli-1.view.jpg", "sha256": "c"},
        ],
        "steps": [],
    }
    relocated = headless.relocated_sidecar(sidecar, {"restored": "foto.png", "uncolored": "foto.uncolored.png"})
    assert relocated["outputs"] == [
        {"role": "restored", "file": "foto.png", "sha256": "a"},
        {"role": "uncolored", "file": "foto.uncolored.png", "sha256": "b"},
    ]
    assert relocated["steps"] == []
    assert sidecar["outputs"][0]["file"] == "cli-1.png"


def _restore_context(tmp_path):
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    ctx = headless.build_context(settings)
    ctx.devices.validate = lambda device_id: {"id": device_id}
    ctx.restore_runner.check_ready = lambda *args, **kwargs: None
    return ctx


def _fake_restore_run(ctx, seen, *, colorized=False):
    async def run(job):
        seen.append(job)
        paths = restore_output_paths(ctx.settings.outputs_path, job.id, job.output_format)
        paths.final.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), (90, 80, 70)).save(paths.final)
        outputs = [{"role": "restored", "file": paths.final.name, "sha256": "x"}]
        if colorized:
            Image.new("RGB", (12, 10), (80, 80, 80)).save(paths.uncolored)
            outputs.append({"role": "uncolored", "file": paths.uncolored.name, "sha256": "y"})
        for extra in (paths.view, paths.preview, paths.before_after):
            extra.write_bytes(b"jpg")
        paths.artifact_dir.mkdir()
        (paths.artifact_dir / "faces.json").write_text("{}", encoding="utf-8")
        paths.sidecar.write_text(json.dumps({"outputs": outputs}), encoding="utf-8")
        job.metadata["restore"] = {
            "warnings": ["w"],
            "faces": [],
            "cpuFallback": [],
            "compositeReasons": ["colorize"] if colorized else [],
            "badge": colorized,
            "upscale": {"mode": "none", "scale": 1.0},
            "digitalSourceType": "x",
        }
        job.output_path = paths.final
        return paths.final

    return run


@pytest.mark.asyncio
async def test_restore_image_delivers_result_uncolored_and_details(tmp_path):
    ctx = _restore_context(tmp_path)
    source = tmp_path / "foto.jpg"
    Image.new("RGB", (12, 10), (100, 90, 80)).save(source)
    seen = []
    ctx.job_manager.run_inline = _fake_restore_run(ctx, seen, colorized=True)
    output = tmp_path / "out" / "foto-restored.png"
    spec = headless.RestoreSpec(steps=("denoise", "tone"), preset="gentle", device="cpu")

    result = await headless.restore_image(ctx, output, spec, source=source)

    assert result["ok"] is True
    assert result["output"] == str(output.resolve())
    assert (result["width"], result["height"]) == (12, 10)
    assert result["steps"] == ["denoise", "tone"]
    assert result["preset"] == "gentle"
    assert result["warnings"] == ["w"]
    assert result["recomposeAvailable"] is False
    uncolored = output.with_name("foto-restored.uncolored.png")
    details = output.with_name("foto-restored.restore.json")
    assert result["uncolored"] == str(uncolored.resolve())
    assert result["details"] == str(details.resolve())
    assert uncolored.exists() and details.exists()
    names = [entry["file"] for entry in json.loads(details.read_text(encoding="utf-8"))["outputs"]]
    assert names == ["foto-restored.png", "foto-restored.uncolored.png"]
    assert not any(ctx.settings.outputs_path.iterdir())
    assert source.exists()
    job = seen[0]
    assert job.id.startswith("cli-")
    assert job.restore_steps == ["denoise", "tone"]
    assert job.restore_session is None
    assert job.restore_options["denoise"]["strength"] == 0.3


@pytest.mark.asyncio
async def test_restore_image_engine_failure_leaves_nothing_behind(tmp_path):
    ctx = _restore_context(tmp_path)
    source = tmp_path / "foto.png"
    Image.new("RGB", (12, 10)).save(source)

    async def boom(job):
        raise RuntimeError("887A0005")

    ctx.job_manager.run_inline = boom
    output = tmp_path / "o.png"
    with pytest.raises(headless.InferenceError, match="887A0005"):
        await headless.restore_image(ctx, output, headless.RestoreSpec(steps=("denoise",), device="cpu"), source=source)
    assert not output.exists()


@pytest.mark.asyncio
async def test_restore_image_rejects_bad_input_before_running(tmp_path):
    ctx = _restore_context(tmp_path)
    ctx.job_manager.run_inline = lambda job: pytest.fail("must not run")
    source = tmp_path / "foto.png"
    Image.new("RGB", (12, 10)).save(source)
    spec = headless.RestoreSpec(steps=("denoise",), device="cpu")
    output = tmp_path / "o.png"
    with pytest.raises(headless.UsageError):
        await headless.restore_image(ctx, output, spec, source=tmp_path / "missing.png")
    with pytest.raises(headless.UsageError):
        await headless.restore_image(ctx, output, spec)
    with pytest.raises(headless.UsageError):
        await headless.restore_image(ctx, output, headless.RestoreSpec(steps=("nope",), device="cpu"), source=source)
    with pytest.raises(headless.UsageError):
        await headless.restore_image(
            ctx, output, headless.RestoreSpec(steps=("denoise",), scale=7, device="cpu"), source=source
        )


@pytest.mark.asyncio
async def test_restore_image_missing_pack_is_model_not_installed(tmp_path):
    ctx = _restore_context(tmp_path)

    def missing(*args, **kwargs):
        raise ValueError("Falta el pack de restauración.")

    ctx.restore_runner.check_ready = missing
    source = tmp_path / "foto.png"
    Image.new("RGB", (12, 10)).save(source)
    with pytest.raises(headless.ModelNotInstalledError, match="Falta"):
        await headless.restore_image(
            ctx, tmp_path / "o.png", headless.RestoreSpec(steps=("repair",), device="cpu"), source=source
        )


def _plain_sessions(ctx):
    store = RestoreSessionStore(ctx.settings, lambda: AnalysisDetectors(damage=None, faces=None))
    ctx.restore_sessions = store
    ctx.restore_runner.sessions = store
    return store


def _session_dirs(ctx):
    root = ctx.settings.video_work_path
    return list(root.glob("restore-*")) if root.exists() else []


@pytest.mark.asyncio
async def test_restore_image_without_steps_analyzes_and_discards_its_session(tmp_path):
    ctx = _restore_context(tmp_path)
    _plain_sessions(ctx)
    source = tmp_path / "recorte.png"
    Image.new("RGB", (40, 30), (120, 110, 100)).save(source)
    seen = []
    ctx.job_manager.run_inline = _fake_restore_run(ctx, seen)
    spec = headless.RestoreSpec(preset="newspaper", device="cpu", options={"geometry": {"rotate90": 1}})

    result = await headless.restore_image(ctx, tmp_path / "out.png", spec, source=source)

    assert result["steps"] == ["descreen", "denoise"]
    assert result["preset"] == "newspaper"
    assert result["token"] is None
    job = seen[0]
    assert job.restore_session is not None
    assert job.restore_options["geometry"]["rotate90"] == 1
    assert job.original_filename == "recorte.png"
    assert _session_dirs(ctx) == []
    assert source.exists()


@pytest.mark.asyncio
async def test_restore_image_without_anything_to_fix_discards_its_session(tmp_path):
    ctx = _restore_context(tmp_path)
    _plain_sessions(ctx)
    ctx.job_manager.run_inline = lambda job: pytest.fail("must not run")
    source = tmp_path / "limpia.png"
    Image.new("RGB", (40, 30), (120, 120, 120)).save(source)
    with pytest.raises(headless.UsageError, match="--steps"):
        await headless.restore_image(ctx, tmp_path / "out.png", headless.RestoreSpec(device="cpu"), source=source)
    assert _session_dirs(ctx) == []


@pytest.mark.asyncio
async def test_restore_analyze_then_restore_with_the_token(tmp_path):
    ctx = _restore_context(tmp_path)
    _plain_sessions(ctx)
    source = tmp_path / "foto.png"
    Image.new("RGB", (40, 30), (120, 110, 100)).save(source)

    analysis = await headless.analyze_photo(ctx, source)

    assert analysis["ok"] is True
    assert analysis["originalName"] == "foto.png"
    assert analysis["proposedPreset"] == "gentle"
    assert Path(analysis["previewPath"]).is_file()
    assert source.exists()
    seen = []
    ctx.job_manager.run_inline = _fake_restore_run(ctx, seen)
    spec = headless.RestoreSpec(steps=("denoise",), device="cpu")
    result = await headless.restore_image(ctx, tmp_path / "out.png", spec, token=analysis["token"])
    assert result["token"] == analysis["token"]
    assert seen[0].restore_session == analysis["token"]
    assert len(_session_dirs(ctx)) == 1
    no_steps = headless.RestoreSpec(device="cpu")
    with pytest.raises(headless.UsageError, match="steps"):
        await headless.restore_image(ctx, tmp_path / "out.png", no_steps, token=analysis["token"])
    with pytest.raises(headless.UsageError, match="session"):
        await headless.restore_image(ctx, tmp_path / "out.png", spec, token="0" * 32)
