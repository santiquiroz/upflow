import os, numpy as np, onnxruntime as ort
from PIL import Image
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- validate uint8 model on DML vs fp32 model ---
rng = np.random.default_rng(3)
small = rng.integers(0, 256, (1, 96, 128, 3), dtype=np.uint8)   # NHWC uint8
s8 = ort.InferenceSession('onnx/anime-x4-uint8.onnx', providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
o8 = s8.run(None, {s8.get_inputs()[0].name: small})[0]
print('uint8 DML  in', small.shape, small.dtype, '-> out', o8.shape, o8.dtype,
      'range', int(o8.min()), int(o8.max()), 'active', s8.get_providers()[0])

# reference: fp32 model (NCHW) fed the same image, manual pre/post
ref_path = '../realesr-animevideov3-x4.onnx'
if os.path.exists(ref_path):
    sf = ort.InferenceSession(ref_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    nchw = np.transpose(small[0].astype(np.float32) / 255.0, (2, 0, 1))[None]
    of = sf.run(None, {sf.get_inputs()[0].name: nchw})[0]
    of_u8 = np.rint(np.clip(np.transpose(of[0], (1, 2, 0)) * 255.0, 0, 255)).astype(np.uint8)
    diff = np.abs(o8[0].astype(int) - of_u8.astype(int))
    print('vs fp32 pipeline: max|diff|=%d  mean|diff|=%.4f  (uint8 levels)' % (diff.max(), diff.mean()))
else:
    print('fp32 ref not found, skipping parity')

# --- generate 30 realistic-ish 720p input frames on disk ---
os.makedirs('in', exist_ok=True)
H, W = 720, 1280
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
grad = np.stack([xx / W * 255, yy / H * 255, (xx + yy) / (W + H) * 255], -1)
for i in range(30):
    r = np.random.default_rng(100 + i)
    # low-freq structure (upsampled blocks) + mid detail, keeps PNG non-trivial but realistic
    blocks = r.integers(0, 256, (45, 80, 3), dtype=np.uint8)
    blocks = np.array(Image.fromarray(blocks).resize((W, H), Image.BILINEAR), dtype=np.float32)
    detail = r.integers(0, 40, (H, W, 3)).astype(np.float32)
    frame = np.clip(0.5 * grad + 0.4 * blocks + detail, 0, 255).astype(np.uint8)
    Image.fromarray(frame).save(f'in/{i:03d}.png')
sizes = [os.path.getsize(f'in/{i:03d}.png') for i in range(30)]
print('generated 30 input frames, avg PNG size %.2f MB' % (np.mean(sizes) / 1e6))
