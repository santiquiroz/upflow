# Vendored from santiquiroz/port-uvr-deecho-onnx driver/vr_params.py @ commit
# 23b2564 (see app/services/engines/vr_deecho/__init__.py for sync notes).
# Verbatim except this header -- no internal `driver.*` imports to rewrite.
"""VR 4band_v3 multiband parameters, transcribed from python-audio-separator's
uvr_lib_v5/vr_network/modelparams/4band_v3.json (MIT). Kept as plain constants so
driver/ has zero file/package dependencies beyond numpy+scipy."""

SR = 44100
BINS = 672
N_FFT = BINS * 2
PRE_FILTER_START = 668
PRE_FILTER_STOP = 672
OFFSET = 64
WINDOW_SIZE = 512

# band index 1 = lowest. crop rows into the combined 673-bin spectrogram are
# contiguous in this order: 85 + 83 + 199 + 305 = 672 (row 672 stays zero).
BANDS = (
    {"sr": 7350, "hl": 80, "n_fft": 640, "crop_start": 0, "crop_stop": 85,
     "lpf_start": 25, "lpf_stop": 53},
    {"sr": 7350, "hl": 80, "n_fft": 320, "crop_start": 4, "crop_stop": 87,
     "hpf_start": 25, "hpf_stop": 12, "lpf_start": 31, "lpf_stop": 62},
    {"sr": 14700, "hl": 160, "n_fft": 512, "crop_start": 17, "crop_stop": 216,
     "hpf_start": 48, "hpf_stop": 24, "lpf_start": 139, "lpf_stop": 210},
    {"sr": 44100, "hl": 480, "n_fft": 960, "crop_start": 78, "crop_stop": 383,
     "hpf_start": 130, "hpf_stop": 86},
)

AGGR_SPLIT_BIN = BANDS[0]["crop_stop"]
