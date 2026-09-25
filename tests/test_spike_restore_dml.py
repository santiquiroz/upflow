from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from test_restore_models import core_bundle, install, make_spec

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "spike_restore_dml.py"
MIB = 1024 * 1024
BUDGET_MS = 1200.0
DML = "dml:0"


def _load_script():
    spec = importlib.util.spec_from_file_location("spike_restore_dml", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


spike = _load_script()


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


class FakeNode:
    def __init__(self, name: str, shape: list, type_: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_


def restore_1x(batch: np.ndarray) -> np.ndarray:
    return batch[:, :3]


def brown(batch: np.ndarray) -> np.ndarray:
    return np.full_like(batch[:, :3], 0.4)


def with_nan(batch: np.ndarray) -> np.ndarray:
    out = batch[:, :3].copy()
    out[0, 0, 0, 0] = np.nan
    return out


class FakeSession:
    def __init__(
        self,
        clock: FakeClock,
        *,
        ms_per_mpx: float = 1000.0,
        transform=restore_1x,
        channels: int = 3,
        fixed_side: int | None = None,
        providers: tuple[str, ...] = ("DmlExecutionProvider", "CPUExecutionProvider"),
        inputs: list[FakeNode] | None = None,
    ) -> None:
        self.clock = clock
        self.ms_per_mpx = ms_per_mpx
        self.transform = transform
        self.providers = providers
        height = fixed_side or "height"
        width = fixed_side or "width"
        self.inputs = inputs or [FakeNode("input", [1, channels, height, width])]

    def get_inputs(self) -> list[FakeNode]:
        return self.inputs

    def get_outputs(self) -> list[FakeNode]:
        return [FakeNode("output", [1, 3, "height", "width"])]

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def run(self, names, feeds):
        batch = feeds["input"]
        self.clock.advance_ms(self.ms_per_mpx * batch.shape[2] * batch.shape[3] / 1_000_000)
        return [self.transform(batch)]


class FakeBackend:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.sessions: dict[tuple[str, str], dict] = {}
        self.opened: list[tuple[str, str, bool]] = []
        self.needs_disable_all: set[str] = set()

    def add(self, filename: str, device: str, **session_kwargs) -> None:
        self.sessions[(filename, device)] = session_kwargs

    def open(self, path: Path, device: str, disable_all: bool) -> FakeSession:
        self.opened.append((path.name, device, disable_all))
        if path.name in self.needs_disable_all and not disable_all:
            raise RuntimeError("Non-zero status code returned while running Gather node (80070057)")
        return FakeSession(self.clock, **self.sessions[(path.name, device)])


def model_file(tmp_path: Path, name: str, precision: str, size: int = 100 * MIB):
    return spike.ModelFile(precision=precision, path=tmp_path / name, sha256="a" * 64, size=size)


def drunet_target(tmp_path: Path):
    return spike.ModelTarget(
        "drunet-color",
        (model_file(tmp_path, "drunet-color.onnx", "fp32"), model_file(tmp_path, "drunet-color-fp16.onnx", "fp16")),
    )


def add_drunet(backend: FakeBackend, *, fp32: dict | None = None, fp16: dict | None = None) -> None:
    backend.add("drunet-color.onnx", "cpu", channels=4, providers=("CPUExecutionProvider",))
    backend.add("drunet-color.onnx", DML, **{"channels": 4, "ms_per_mpx": 4000.0, **(fp32 or {})})
    backend.add("drunet-color-fp16.onnx", DML, **{"channels": 4, "ms_per_mpx": 1000.0, **(fp16 or {})})


def make_harness(backend: FakeBackend, *, cpu_nodes=None, vram=None, device: str = DML):
    readings = iter(vram or [])
    return spike.Harness(
        device=device,
        budget_ms=BUDGET_MS,
        open_session=backend.open,
        cpu_nodes=cpu_nodes or (lambda path, disable_all: ()),
        vram_usage_mb=lambda: next(readings, None),
        clock=backend.clock,
        catalog={},
    )


def by_precision(results) -> dict:
    return {result.precision: result for result in results}


def test_report_table_has_the_agreed_columns(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend)
    results, skipped = spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])

    report = spike.render_report(spike.ReportContext("2026-09-25", DML, "dist", "1.24.4", BUDGET_MS), results, skipped)

    header = "| " + " | ".join(spike.RESULT_COLUMNS) + " |"
    assert spike.RESULT_COLUMNS == (
        "Modelo", "Precisión", "Archivo", "SHA-256", "Providers", "Nodos en CPU", "Tile", "ms/Mpx",
        "Mediana ms", "vs DML fp32", "vs CPU fp32", "NaN/Inf", "vram_factor", "ORT_DISABLE_ALL", "Veredicto",
    )
    assert header in report
    rows = spike.report_rows(report)
    assert set(rows) == {("drunet-color", "fp32"), ("drunet-color", "fp16")}
    assert rows[("drunet-color", "fp16")]["SHA-256"] == "a" * 64
    assert rows[("drunet-color", "fp16")]["Veredicto"] == "ok"
    assert "| drunet-color | {'fp32': 384, 'fp16': 512} | 'drunet-color-fp16.onnx' | False | n/d |" in report


def test_calibration_runs_per_precision_with_its_own_tile(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend)

    results, _ = spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])

    rows = by_precision(results)
    assert rows["fp32"].calibration.tile == 384
    assert rows["fp16"].calibration.tile == 512
    assert rows["fp32"].median_ms == pytest.approx(4000.0 * 384 * 384 / 1_000_000)
    assert rows["fp32"].median_ms < BUDGET_MS / 2


def test_fp16_that_comes_out_brown_is_discarded_by_the_canary(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend, fp16={"transform": brown})

    results, _ = spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])

    fp16 = by_precision(results)["fp16"]
    assert not fp16.non_finite
    assert fp16.vs_fp32 < 50.0
    assert spike.verdict(fp16).startswith("fp16 descartado: canario vs DML fp32")
    suggestion = spike.suggest_specs(results)[0]
    assert suggestion.fp16_filename is None
    assert dict(suggestion.tile_by_precision) == {"fp32": 384}


def test_non_finite_output_fails_the_row(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend, fp32={"transform": with_nan})

    results, _ = spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])

    fp32 = by_precision(results)["fp32"]
    assert fp32.non_finite
    assert spike.verdict(fp32) == "falla: NaN/Inf en la salida"


def test_nodes_that_run_on_cpu_fail_the_row(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend)
    harness = make_harness(backend, cpu_nodes=lambda path, disable_all: ("Resize", "Where", "Resize"))

    results, _ = spike.validate_targets(harness, [drunet_target(tmp_path)])

    assert spike.verdict(by_precision(results)["fp32"]) == "falla: 3 nodos en CPU (Resize, Where)"


def test_a_session_that_ort_moved_to_cpu_fails_the_row(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend, fp32={"providers": ("CPUExecutionProvider",)})

    results, _ = spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])

    assert spike.verdict(by_precision(results)["fp32"]) == "falla: la sesión corre en CPUExecutionProvider"


def test_fixed_shape_model_over_half_budget_is_reported_as_cpu(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    backend.add("gfpgan.onnx", "cpu", fixed_side=512, providers=("CPUExecutionProvider",))
    backend.add("gfpgan.onnx", DML, fixed_side=512, ms_per_mpx=4000.0)
    target = spike.ModelTarget("gfpgan-v1.4", (model_file(tmp_path, "gfpgan.onnx", "fp32"),))

    results, _ = spike.validate_targets(make_harness(backend), [target])

    assert results[0].calibration.tile is None
    assert results[0].median_ms is None
    assert spike.verdict(results[0]) == "falla: corre en CPU (tdrBudget)"


def test_median_over_half_budget_is_an_issue() -> None:
    issue = spike.median_issue(700.0, BUDGET_MS)

    assert issue == "mediana 700 ms > 600 ms"
    assert spike.median_issue(599.0, BUDGET_MS) is None


def test_vram_factor_is_session_usage_over_file_bytes(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend)
    harness = make_harness(backend, vram=[1000, 1300, 1300, 1450])

    results, _ = spike.validate_targets(harness, [drunet_target(tmp_path)])

    rows = by_precision(results)
    assert rows["fp32"].vram_factor == pytest.approx(3.0)
    assert rows["fp16"].vram_factor == pytest.approx(1.5)
    assert spike.suggest_specs(results)[0].vram_factor == pytest.approx(3.0)


def test_vram_factor_is_unknown_when_usage_cannot_be_read() -> None:
    assert spike.vram_factor(None, 1300, 100 * MIB) is None
    assert spike.vram_factor(1000, 900, 100 * MIB) == 0.0


def test_graph_that_fails_with_optimizations_is_retried_with_ort_disable_all(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend)
    backend.needs_disable_all.add("drunet-color-fp16.onnx")

    results, _ = spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])

    fp16 = by_precision(results)["fp16"]
    assert fp16.ort_disable_all and spike.verdict(fp16) == "ok"
    assert ("drunet-color-fp16.onnx", DML, True) in backend.opened
    assert spike.suggest_specs(results)[0].ort_disable_all


def test_a_graph_that_never_opens_reports_the_first_error(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend)
    del backend.sessions[("drunet-color-fp16.onnx", DML)]

    results, _ = spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])

    assert spike.verdict(by_precision(results)["fp16"]).startswith("fp16 descartado: error: KeyError")


def test_device_removal_aborts_the_run_instead_of_retrying(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend, fp32={"transform": _raise_device_removed})

    with pytest.raises(RuntimeError, match="887A0005"):
        spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])
    assert ("drunet-color.onnx", DML, True) not in backend.opened


def _raise_device_removed(batch: np.ndarray) -> np.ndarray:
    raise RuntimeError("DXGI_ERROR_DEVICE_REMOVED 887A0005")


def test_detectors_and_uint8_frame_graphs_are_listed_as_not_validated(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    u8_inputs = [FakeNode("input", [1, "height", "width", 3], "tensor(uint8)"), FakeNode("strength", [])]
    backend.add("drunet-deblock-color-u8.onnx", "cpu", inputs=u8_inputs)
    targets = [
        spike.ModelTarget("bopbtl-scratch-detector", (model_file(tmp_path, "bopbtl.onnx", "fp32"),)),
        spike.ModelTarget("drunet-deblock-color-u8", (model_file(tmp_path, "drunet-deblock-color-u8.onnx", "fp32"),)),
    ]

    results, skipped = spike.validate_targets(make_harness(backend), targets)

    assert results == []
    reasons = {item.model_id: item.reason for item in skipped}
    assert "CPU por diseño" in reasons["bopbtl-scratch-detector"]
    assert reasons["drunet-deblock-color-u8"] == "contrato no soportado: entrada tensor(uint8)"
    assert ("bopbtl.onnx", "cpu", False) not in backend.opened


def test_models_dir_reads_models_lock(tmp_path: Path) -> None:
    for name in ("drunet-color.onnx", "drunet-color-fp16.onnx", "retinaface-r34.onnx"):
        (tmp_path / name).write_bytes(b"onnx")
    lock = {
        "assets": [
            {"file": "drunet-color-fp16.onnx", "model": "drunet-color", "precision": "fp16"},
            {"file": "retinaface-r34.onnx", "model": "retinaface-r34", "precision": "fp32"},
            {"file": "drunet-color.onnx", "model": "drunet-color", "precision": "fp32"},
        ]
    }
    (tmp_path / "models.lock.json").write_text(json.dumps(lock), encoding="utf-8")

    targets = spike.discover_models_dir(tmp_path)

    assert [target.model_id for target in targets] == ["drunet-color", "retinaface-r34"]
    drunet = targets[0]
    assert [f.precision for f in drunet.files] == ["fp32", "fp16"]
    assert drunet.files[0].sha256 == spike.sha256_file(tmp_path / "drunet-color.onnx")
    assert drunet.files[0].size == 4


def test_models_dir_without_lock_pairs_fp16_files_by_name(tmp_path: Path) -> None:
    for name in ("drunet_deblock_real_op17.onnx", "drunet_deblock_real_op17_fp16.onnx", "migan-fp16.onnx"):
        (tmp_path / name).write_bytes(b"x")

    targets = spike.discover_models_dir(tmp_path)

    files = {t.model_id: [(f.precision, f.path.name) for f in t.files] for t in targets}
    assert files == {
        "drunet_deblock_real_op17": [
            ("fp32", "drunet_deblock_real_op17.onnx"),
            ("fp16", "drunet_deblock_real_op17_fp16.onnx"),
        ],
        "migan": [("fp16", "migan-fp16.onnx")],
    }


def test_installed_targets_come_from_installed_bundles(tmp_path: Path) -> None:
    bundle = core_bundle()
    install(tmp_path, bundle)
    models = {"drunet-color": make_spec()}

    targets = spike.installed_targets(tmp_path, models, {"core": bundle, "faces": spike.RestoreBundle("faces", ())})

    assert [(t.model_id, [f.path.name for f in t.files]) for t in targets] == [
        ("drunet-color", ["drunet-color.onnx", "drunet-color-fp16.onnx"])
    ]


def test_the_catalog_sets_tile_min_and_candidates_for_dynamic_graphs(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend)
    harness = spike.Harness(
        device=DML,
        budget_ms=BUDGET_MS,
        open_session=backend.open,
        cpu_nodes=lambda path, disable_all: (),
        vram_usage_mb=lambda: None,
        clock=backend.clock,
        catalog={"drunet-color": make_spec(tile_min=64, tile_candidates=(128, 192))},
    )

    results, _ = spike.validate_targets(harness, [drunet_target(tmp_path)])

    assert {r.calibration.tile for r in results} == {192}


def test_cpu_device_only_measures_fp32(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend)

    results, _ = spike.validate_targets(make_harness(backend, device="cpu"), [drunet_target(tmp_path)])

    assert [r.precision for r in results] == ["fp32"]
    assert results[0].vs_fp32 is None and math.isinf(results[0].vs_cpu)


def test_a_graph_published_only_with_fp16_weights_uses_itself_on_cpu_as_reference(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    backend.add("ddcolor-tiny-fp16.onnx", "cpu", fixed_side=512, providers=("CPUExecutionProvider",))
    backend.add("ddcolor-tiny-fp16.onnx", DML, fixed_side=512, ms_per_mpx=500.0)
    target = spike.ModelTarget("ddcolor-tiny", (model_file(tmp_path, "ddcolor-tiny-fp16.onnx", "fp16"),))

    results, skipped = spike.validate_targets(make_harness(backend), [target])

    assert skipped == []
    assert results[0].vs_cpu == pytest.approx(0.0) and spike.verdict(results[0]) == "ok"
    assert ("ddcolor-tiny-fp16.onnx", "cpu", False) in backend.opened


def test_a_non_finite_fp32_does_not_poison_the_fp16_comparison(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend, fp32={"transform": with_nan})

    results, _ = spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])

    fp16 = by_precision(results)["fp16"]
    assert fp16.vs_fp32 is None
    assert spike.verdict(fp16) == "ok"


def test_a_graph_whose_cpu_reference_fails_is_listed_as_not_validated(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend)
    backend.needs_disable_all.add("drunet-color.onnx")

    results, skipped = spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])

    assert results == []
    assert skipped[0].reason.startswith("la referencia en CPU falla: RuntimeError: Non-zero status")


def test_expected_report_flags_changed_hashes_and_lost_verdicts(tmp_path: Path) -> None:
    backend = FakeBackend(FakeClock())
    add_drunet(backend)
    results, skipped = spike.validate_targets(make_harness(backend), [drunet_target(tmp_path)])
    context = spike.ReportContext("2026-09-25", DML, "dist", "1.24.4", BUDGET_MS)
    expected = spike.render_report(context, results, skipped)

    fp32, fp16 = results
    changed = [fp32, dataclasses.replace(fp16, file=dataclasses.replace(fp16.file, sha256="c" * 64))]
    broken = [dataclasses.replace(fp32, issues=("mediana 700 ms > 600 ms",)), fp16]

    assert spike.report_differences(expected, expected) == []
    assert spike.report_differences(expected, spike.render_report(context, changed, skipped)) == [
        f"drunet-color fp16: SHA-256 {'a' * 64} -> {'c' * 64}"
    ]
    assert spike.report_differences(expected, spike.render_report(context, broken, skipped)) == [
        "drunet-color fp32: veredicto ok -> falla: mediana 700 ms > 600 ms"
    ]
    assert spike.report_differences(expected, spike.render_report(context, results[:1], skipped)) == [
        "drunet-color fp16: falta en la corrida nueva"
    ]


def test_gpu_needs_the_explicit_opt_in() -> None:
    assert not spike.gpu_allowed(DML, {})
    assert spike.gpu_allowed(DML, {"UPFLOW_GPU_TESTS": "1"})
    assert spike.gpu_allowed("cpu", {})


def test_main_refuses_the_gpu_without_opt_in(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("UPFLOW_GPU_TESTS", raising=False)

    code = spike.main(["--models-dir", str(tmp_path), "--device", DML, "--report", str(tmp_path / "r.md")])

    assert code == 2
    assert "UPFLOW_GPU_TESTS=1" in capsys.readouterr().err
    assert not (tmp_path / "r.md").exists()


def test_restore_sessions_skip_native_eps_and_use_restore_options(monkeypatch) -> None:
    import onnxruntime as ort

    calls = []
    monkeypatch.setattr(
        spike.ep_registry,
        "create_session",
        lambda path, device, settings, **kwargs: calls.append((path, device, kwargs)) or "session",
    )

    assert spike.open_restore_session(Path("m.onnx"), DML, object(), disable_all=True, profile=True) == "session"

    path, device, kwargs = calls[0]
    assert (path, device, kwargs["prefer_native"]) == ("m.onnx", DML, False)
    options = kwargs["sess_options_factory"]()
    assert options.enable_mem_pattern is False
    assert options.enable_profiling is True
    assert options.intra_op_num_threads == 1
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL


def test_profiled_cpu_nodes_come_from_the_ort_profile(tmp_path: Path) -> None:
    events = [
        {"cat": "Node", "name": "conv", "args": {"op_name": "Conv", "provider": "DmlExecutionProvider"}},
        {"cat": "Node", "name": "resize", "args": {"op_name": "Resize", "provider": "CPUExecutionProvider"}},
        {"cat": "Session", "name": "load"},
    ]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(events), encoding="utf-8")

    class ProfiledSession(FakeSession):
        def end_profiling(self) -> str:
            return str(profile)

    session = ProfiledSession(FakeClock(), channels=4)

    assert spike.profile_cpu_nodes(lambda: session, DML) == ("Resize",)
    assert not profile.exists()
    assert spike.profile_cpu_nodes(lambda: pytest.fail("cpu never profiles"), "cpu") == ()


def _tiny_restore_graph(path: Path, gain: float = 1.0) -> None:
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "one"], ["output"])],
        "tiny_restore",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "height", "width"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "height", "width"])],
        [helper.make_tensor("one", TensorProto.FLOAT, [], [gain])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))


def test_main_writes_the_report_for_a_tiny_cpu_graph(tmp_path: Path) -> None:
    models = tmp_path / "dist"
    models.mkdir()
    _tiny_restore_graph(models / "tiny-restore.onnx")
    report = tmp_path / "report.md"

    code = spike.main(["--models-dir", str(models), "--device", "cpu", "--report", str(report), "--budget-ms", "1200"])

    assert code == 0
    rows = spike.report_rows(report.read_text(encoding="utf-8"))
    row = rows[("tiny-restore", "fp32")]
    assert row["Veredicto"] == "ok"
    assert row["Providers"] == "CPUExecutionProvider"
    assert row["Nodos en CPU"] == "0"
    assert row["vs CPU fp32"] == "∞ dB"
    assert row["SHA-256"] == spike.sha256_file(models / "tiny-restore.onnx")


def test_expect_report_writes_a_rerun_next_to_it_and_never_overwrites_it(tmp_path: Path) -> None:
    models = tmp_path / "dist"
    models.mkdir()
    _tiny_restore_graph(models / "tiny-restore.onnx")
    expected = tmp_path / "2026-09-25-restore-dml-validation.md"
    assert spike.main(["--models-dir", str(models), "--device", "cpu", "--report", str(expected)]) == 0
    original = expected.read_text(encoding="utf-8")

    code = spike.main(["--models-dir", str(models), "--device", "cpu", "--expect-report", str(expected)])

    assert code == 0
    assert expected.read_text(encoding="utf-8") == original
    rerun = tmp_path / "2026-09-25-restore-dml-validation-rerun.md"
    assert spike.report_rows(rerun.read_text(encoding="utf-8")).keys() == {("tiny-restore", "fp32")}
    _tiny_restore_graph(models / "tiny-restore.onnx", gain=0.5)
    assert spike.main(["--models-dir", str(models), "--device", "cpu", "--expect-report", str(expected)]) == 1
    rerun_rows = spike.report_rows(rerun.read_text(encoding="utf-8"))
    assert rerun_rows[("tiny-restore", "fp32")]["Veredicto"] == "ok"
