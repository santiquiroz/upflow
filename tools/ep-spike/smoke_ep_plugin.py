"""Smoke autocontenido: plugins EP nativos sobre onnxruntime-directml 1.24.4.

Corre en cualquier GPU (NVIDIA/AMD/Intel). No descarga modelos: genera un
mini upscaler x2 (conv stack + pixel shuffle) en memoria. Compara DirectML
vs el EP nativo disponible y reporta JSON al final para copiar/pegar.

Uso (ver README.md de esta carpeta):
    python smoke_ep_plugin.py
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import time

import numpy as np

RESULT: dict = {
    "machine": platform.node(),
    "os": platform.platform(),
    "python": sys.version.split()[0],
    "steps": {},
}


def step(name: str, status: str, detail: str = "") -> None:
    RESULT["steps"][name] = {"status": status, "detail": detail}
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def build_tiny_upscaler() -> bytes:
    import onnx
    from onnx import TensorProto, helper

    rng = np.random.RandomState(7)
    convs = []
    inits = []
    prev, prev_ch = "xf", 3
    for i, ch in enumerate((32, 32, 32, 12)):
        w = rng.randn(ch, prev_ch, 3, 3).astype(np.float32) * 0.1
        b = rng.randn(ch).astype(np.float32) * 0.01
        inits += [
            helper.make_tensor(f"w{i}", TensorProto.FLOAT, w.shape, w.flatten()),
            helper.make_tensor(f"b{i}", TensorProto.FLOAT, b.shape, b),
        ]
        convs += [
            helper.make_node("Conv", [prev, f"w{i}", f"b{i}"], [f"c{i}"], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", [f"c{i}"], [f"r{i}"]),
        ]
        prev, prev_ch = f"r{i}", ch
    nodes = [
        helper.make_node("Cast", ["x"], ["xf"], to=TensorProto.FLOAT),
        *convs,
        helper.make_node("DepthToSpace", [prev], ["up"], blocksize=2),
        helper.make_node("Clip", ["up", "lo", "hi"], ["clip"]),
        helper.make_node("Cast", ["clip"], ["y"], to=TensorProto.UINT8),
    ]
    inits += [
        helper.make_tensor("lo", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("hi", TensorProto.FLOAT, [], [255.0]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_upscaler_x2",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 3, 512, 512])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 3, 1024, 1024])],
        inits,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 20)])
    onnx.checker.check_model(model)
    return model.SerializeToString()


def bench(session, feed: dict, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        session.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(runs):
        session.run(None, feed)
    return (time.perf_counter() - t0) / runs * 1000.0


def preload_plugin_deps(lib_path: str) -> None:
    lib_dir = os.path.dirname(lib_path)
    os.add_dll_directory(lib_dir)
    for name in sorted(os.listdir(lib_dir)):
        if not name.lower().endswith(".dll") or name == os.path.basename(lib_path):
            continue
        try:
            ctypes.WinDLL(os.path.join(lib_dir, name))
        except OSError:
            pass  # deps opcionales pueden faltar; el registro reporta el fallo real


def try_native_plugin(ort, registration_name: str, lib_path: str) -> list:
    preload_plugin_deps(lib_path)
    ort.register_execution_provider_library(registration_name, lib_path)
    return [d for d in ort.get_ep_devices() if d.ep_name == registration_name]


def detect_nvidia_plugin():
    try:
        import onnxruntime_ep_nv_tensorrt_rtx as nv

        return "NvTensorRTRTXExecutionProvider", nv.get_library_path()
    except ImportError:
        return None


def detect_webgpu_plugin():
    try:
        import onnxruntime_ep_webgpu as wg

        return "WebGpuExecutionProvider", wg.get_library_path()
    except ImportError:
        return None


def run_provider(ort, model: bytes, feed: dict, ep_devices=None, providers=None, warmup=3, runs=10):
    so = ort.SessionOptions()
    if ep_devices:
        so.add_provider_for_devices(ep_devices, {})
        t0 = time.perf_counter()
        sess = ort.InferenceSession(model, sess_options=so)
    else:
        t0 = time.perf_counter()
        sess = ort.InferenceSession(model, sess_options=so, providers=providers)
    create_ms = (time.perf_counter() - t0) * 1000.0
    out = sess.run(None, feed)[0]
    avg_ms = bench(sess, feed, warmup, runs)
    return out, create_ms, avg_ms


def main() -> None:
    import onnxruntime as ort

    RESULT["onnxruntime"] = ort.__version__
    step("import", "OK", f"onnxruntime {ort.__version__} | {ort.get_available_providers()}")
    if "DmlExecutionProvider" not in ort.get_available_providers():
        step("dml_presente", "FAIL", "este smoke espera el wheel onnxruntime-directml")
        return

    model = build_tiny_upscaler()
    x = np.random.randint(0, 256, (1, 3, 512, 512), dtype=np.uint8)
    feed = {"x": x}
    step("modelo", "OK", "tiny_upscaler_x2 generado en memoria (uint8 IO)")

    RESULT["ep_devices_base"] = [
        f"{d.ep_name}|{d.device.type}|{d.device.vendor}" for d in ort.get_ep_devices()
    ]

    y_cpu, _, cpu_ms = run_provider(ort, model, feed, providers=["CPUExecutionProvider"], runs=3)
    RESULT["cpu_ms"] = round(cpu_ms, 2)
    step("cpu_baseline", "OK", f"{cpu_ms:.1f} ms/frame")

    try:
        y_dml, dml_create, dml_ms = run_provider(ort, model, feed, providers=["DmlExecutionProvider"])
        RESULT["dml_ms"] = round(dml_ms, 2)
        RESULT["dml_create_ms"] = round(dml_create, 1)
        mismatch = int(np.abs(y_dml.astype(np.int16) - y_cpu.astype(np.int16)).max())
        step("dml", "OK", f"{dml_ms:.1f} ms/frame | diff-uint8 max {mismatch}")
        if mismatch > 2:
            step("dml_correctitud", "WARN", f"diff {mismatch} > 2")
    except Exception as e:
        step("dml", "FAIL", str(e)[:300])
        y_dml = None

    for label, detected in (("trt_rtx", detect_nvidia_plugin()), ("webgpu", detect_webgpu_plugin())):
        if not detected:
            step(f"{label}_plugin", "SKIP", "paquete pip no instalado")
            continue
        reg_name, lib_path = detected
        try:
            devices = try_native_plugin(ort, reg_name, lib_path)
        except Exception as e:
            hint = " (esperado si la GPU no es NVIDIA)" if label == "trt_rtx" else ""
            step(f"{label}_registro", "FAIL", str(e)[:300] + hint)
            continue
        if not devices:
            step(f"{label}_devices", "EMPTY", "registrado pero 0 devices (hardware no soportado)")
            continue
        step(f"{label}_registro", "OK", f"{len(devices)} device(s)")
        try:
            y_nat, nat_create, nat_ms = run_provider(ort, model, feed, ep_devices=devices[:1])
            RESULT[f"{label}_ms"] = round(nat_ms, 2)
            RESULT[f"{label}_create_ms"] = round(nat_create, 1)
            mismatch = int(np.abs(y_nat.astype(np.int16) - y_cpu.astype(np.int16)).max())
            speedup = RESULT.get("dml_ms", 0) / nat_ms if nat_ms else 0
            step(
                f"{label}_inferencia",
                "OK",
                f"{nat_ms:.1f} ms/frame (compilacion 1er uso {nat_create:.0f} ms) | "
                f"{speedup:.2f}x vs DML | diff-uint8 max {mismatch}",
            )
            if mismatch > 2:
                step(f"{label}_correctitud", "WARN", f"diff {mismatch} > 2")
        except Exception as e:
            step(f"{label}_inferencia", "FAIL", str(e)[:300])

    if y_dml is not None:
        try:
            sess = ort.InferenceSession(model, providers=["DmlExecutionProvider"])
            sess.run(None, feed)
            step("dml_post_plugins", "OK", "DML sigue funcionando tras registrar plugins")
        except Exception as e:
            step("dml_post_plugins", "FAIL", str(e)[:300])

    print("\n=== RESULTADO (copiar y pegar todo) ===")
    print(json.dumps(RESULT, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
