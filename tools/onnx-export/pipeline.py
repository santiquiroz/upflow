import os, time, queue, threading, statistics, json
import numpy as np
import onnxruntime as ort
from PIL import Image
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL = 'onnx/anime-x4-uint8.onnx'
IN_DIR, OUT_DIR = 'in', 'out'
N = 30
IDX = list(range(N))

def make_session():
    return ort.InferenceSession(MODEL, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

sess = make_session()
INN = sess.get_inputs()[0].name
OUTN = sess.get_outputs()[0].name

def load_frame(i):
    with Image.open(f'{IN_DIR}/{i:03d}.png') as im:
        return np.asarray(im.convert('RGB'), dtype=np.uint8)[None]  # NHWC uint8 [1,H,W,3]

def infer_plain(arr):
    return sess.run([OUTN], {INN: arr})[0]

def infer_iobind(arr):
    io = sess.io_binding()
    x = ort.OrtValue.ortvalue_from_numpy(arr, 'dml', 0)
    io.bind_ortvalue_input(INN, x)
    io.bind_output(OUTN, 'dml')
    sess.run_with_iobinding(io)
    return io.copy_outputs_to_cpu()[0]

def save_png(arr, i, level=6):
    Image.fromarray(arr[0], 'RGB').save(f'{OUT_DIR}/{i:03d}.png', compress_level=level)

def save_ppm(arr, i):
    h, w, _ = arr[0].shape
    with open(f'{OUT_DIR}/{i:03d}.ppm', 'wb') as f:
        f.write(f'P6\n{w} {h}\n255\n'.encode()); f.write(arr[0].tobytes())

def med(fn, reps, warm=2):
    ts = []
    for k in range(reps + warm):
        t0 = time.perf_counter(); r = fn(k); t1 = time.perf_counter()
        if k >= warm: ts.append((t1 - t0) * 1000)
    return round(statistics.median(ts), 1), r

# warm up inference (compile DML shaders for 720p shape)
warm = load_frame(0)
for _ in range(3):
    _ = infer_iobind(warm); _ = infer_plain(warm)

# ---- per-stage isolated costs (median over frames) ----
loaded0 = load_frame(0)
out0 = infer_iobind(loaded0)
load_ms, _ = med(lambda k: load_frame(k % N), 12)
infer_plain_ms, _ = med(lambda k: infer_plain(loaded0), 12)
infer_iobind_ms, _ = med(lambda k: infer_iobind(loaded0), 12)
save_png_ms, _ = med(lambda k: save_png(out0, 900), 8)
save_png1_ms, _ = med(lambda k: save_png(out0, 901, level=1), 8)
save_ppm_ms, _ = med(lambda k: save_ppm(out0, 902), 8)

stage = {
    'load_png_ms': load_ms,
    'infer_plain_ms': infer_plain_ms, 'infer_iobind_ms': infer_iobind_ms,
    'save_png_l6_ms': save_png_ms, 'save_png_l1_ms': save_png1_ms, 'save_ppm_ms': save_ppm_ms,
    'out_shape': list(out0.shape),
}

# ---- sequential end-to-end (load -> infer(iobind) -> save png), 30 frames ----
t0 = time.perf_counter()
for i in IDX:
    a = load_frame(i); o = infer_iobind(a); save_png(o, i)
seq_elapsed = time.perf_counter() - t0

# ---- threaded pipeline: loaders + single GPU infer + savers ----
def run_pipeline(save_fn, n_load=3, n_save=4):
    load_q = queue.Queue(maxsize=6)
    save_q = queue.Queue(maxsize=6)
    todo = queue.Queue()
    for i in IDX:
        todo.put(i)

    def loader():
        while True:
            try:
                i = todo.get_nowait()
            except queue.Empty:
                return
            load_q.put((i, load_frame(i)))

    def saver():
        while True:
            item = save_q.get()
            if item is None:
                save_q.task_done(); return
            i, o = item
            save_fn(o, i)
            save_q.task_done()

    loaders = [threading.Thread(target=loader) for _ in range(n_load)]
    savers = [threading.Thread(target=saver) for _ in range(n_save)]
    t0 = time.perf_counter()
    for t in loaders + savers:
        t.start()
    for _ in IDX:
        i, a = load_q.get()
        o = infer_iobind(a)
        save_q.put((i, o))
    for _ in savers:
        save_q.put(None)
    for t in savers:
        t.join()
    return time.perf_counter() - t0

pipe_png = run_pipeline(lambda o, i: save_png(o, i))
pipe_ppm = run_pipeline(lambda o, i: save_ppm(o, i))

result = {
    'active_provider': sess.get_providers()[0],
    'stage_medians': stage,
    'baseline_ncnn_fps': 5.4,
    'sequential_png': {'elapsed_s': round(seq_elapsed, 2), 'fps': round(N / seq_elapsed, 2)},
    'pipeline_png': {'elapsed_s': round(pipe_png, 2), 'fps': round(N / pipe_png, 2)},
    'pipeline_ppm': {'elapsed_s': round(pipe_ppm, 2), 'fps': round(N / pipe_ppm, 2)},
}
print(json.dumps(result, indent=2))
