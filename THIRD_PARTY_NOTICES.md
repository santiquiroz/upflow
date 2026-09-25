# Third-party notices

Upflow itself is released under the MIT License (see `LICENSE`). The portable zip
and the Windows installer also redistribute the third-party components listed
below, each under its own license. Paths are relative to the Upflow install folder.

Every entry names the component, its license (SPDX identifier), copyright holders
and source. `tests/test_third_party_notices.py` cross-checks this file against
what `scripts/package-release.ps1` ships, the restoration model catalog, the
bundled fonts and every source file that carries an `Adapted from` provenance line.

Components that Upflow downloads on first launch or from a pack button (ncnn
engines, FFmpeg, speech and audio models, and so on) are fetched from their
upstream publishers and are not redistributed by the release; their licenses
travel with each download.

## Components bundled in the release

### Real-ESRGAN ONNX exports

- Component: `vendor/realesrgan-onnx/`
- License: BSD-3-Clause
- Copyright: Copyright (c) 2021, Xintao Wang
- Source: https://github.com/xinntao/Real-ESRGAN (weights from releases v0.1.0 `RealESRGAN_x4plus.pth`, v0.2.2.4 `RealESRGAN_x4plus_anime_6B.pth` and v0.2.5.0 `realesr-animevideov3.pth`)
- Training data (declared): according to the Real-ESRGAN paper, the general models were trained on DIV2K, Flickr2K and OutdoorSceneTraining, image collections distributed for research use; the anime models were trained on anime images the authors do not further specify.
- Modifications: exported to ONNX (opset 17) by `scripts/export-realesrgan-onnx.py`, with uint8 NHWC pre- and post-processing baked into the graph; the x2 and x3 files add an area resample of the x4 output; the `-fp16` files run the network body in half precision.

```text
BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Apollo audio restoration model

- Component: `vendor/apollo/`
- License: CC-BY-SA-4.0
- License URL: https://creativecommons.org/licenses/by-sa/4.0/
- Copyright: Kai Li and Yi Luo, "Apollo: Band-sequence Modeling for High-Quality Audio Restoration" (ICASSP 2025, arXiv:2409.08514)
- Source: https://github.com/JusperLee/Apollo (official weights: https://huggingface.co/JusperLee/Apollo)
- Training data (declared): MUSDB18 according to the upstream model card; the dataset is distributed for non-commercial research use.
- Modifications: the PyTorch model was exported to a single ONNX graph (`apollo.onnx`, 44.1 kHz mono, input `audio`, output `restored`). This adapted model is shared under the same CC BY-SA 4.0 license.
- Disclaimer: the licensed material is provided as-is and as-available, without warranties of any kind (Section 5 of the license).

### fetchflow

- Component: `vendor/wheels/`
- License: MIT
- Copyright: Copyright (c) 2026 Santiago Quiroz
- Source: https://github.com/santiquiroz/fetchflow
- Notes: shipped as a wheel built without its dependencies; those (yt-dlp among them) are installed from PyPI on first launch and are not redistributed by the release.

### Python embeddable distribution

- Component: `python/`
- License: PSF-2.0
- Copyright: Copyright (c) 2001 Python Software Foundation. All rights reserved.
- Source: https://www.python.org/downloads/windows/ (version pinned in `scripts/package-release.ps1`)
- License text: `python/LICENSE.txt`, shipped unmodified with the interpreter; it also lists the licenses of the libraries CPython bundles.
- Includes: pip, setuptools and wheel (MIT), installed into the embedded interpreter; each keeps its license file in its `*.dist-info` folder.
- Modifications: `python3*._pth` enables `import site` and `Lib\site-packages`.

### React, React DOM and scheduler

- Component: `frontend/dist/`
- Package: react
- Package: react-dom
- License: MIT
- Copyright: Copyright (c) Facebook, Inc. and its affiliates.
- Source: https://github.com/facebook/react
- Includes: scheduler (MIT, same copyright)

### React Router

- Component: `frontend/dist/`
- Package: react-router-dom
- License: MIT
- Copyright: Copyright (c) React Training LLC 2015-2019
- Copyright: Copyright (c) Remix Software Inc. 2020-2021
- Source: https://github.com/remix-run/react-router
- Includes: react-router and @remix-run/router (MIT, same copyright)

### TanStack Query

- Component: `frontend/dist/`
- Package: @tanstack/react-query
- License: MIT
- Copyright: Copyright (c) 2021-present Tanner Linsley
- Source: https://github.com/TanStack/query
- Includes: @tanstack/query-core (MIT, same copyright)

### Lucide icons

- Component: `frontend/dist/`
- Package: lucide-react
- License: ISC
- Copyright: Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
- Source: https://github.com/lucide-icons/lucide

The MIT-licensed web UI packages above are distributed under these terms:

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Lucide is distributed under these terms:

```text
ISC License

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

## Fonts

Fonts bundled under `app/assets/fonts/` get one entry each here, with their
`OFL.txt` shipped next to the font file. No font is bundled yet.

## Photo restoration models

The photo restoration models are downloaded as `restore-*` packs; none of them is
bundled in the release. Each pack installs the model's `LICENSE` and `NOTICE.txt`
under `vendor/restore/licenses/<model>/`. Every model in the catalog gets an entry
here with a `Restore model:` line, its license, copyright, source and declared
training data. No model is published yet.

## Ported source code

Source files that adapt third-party code start with a provenance line
(`# Adapted from <repo>@<commit> (<license>, © <author>)`) and get an entry here
listing them with `Ported into:` lines.

### AudioSR DSP and DDIM sampler

- Ported into: app/services/engines/audiosr/dsp.py
- Ported into: app/services/engines/audiosr/ddim.py
- License: MIT
- Copyright: Copyright (c) 2012-2023 Scott Chacon and others (as written in the upstream LICENSE file)
- Source: https://github.com/haoheliu/versatile_audio_super_resolution (audiosr 0.0.7)
- Modifications: the DSP glue (`utils.py`, `lowpass.py`, post-processing) and the DDIM sampler were rewritten in NumPy and parity-tested against the original PyTorch pipeline.

### VR De-Echo driver

- Ported into: app/services/engines/vr_deecho/dsp.py
- Ported into: app/services/engines/vr_deecho/multiband.py
- Ported into: app/services/engines/vr_deecho/pipeline.py
- Ported into: app/services/engines/vr_deecho/vr_params.py
- License: MIT
- Copyright: Copyright (c) 2026 Santiago Quiroz (port-uvr-deecho-onnx)
- Copyright: Copyright (c) 2023 karaokenerds (python-audio-separator)
- Source: https://github.com/santiquiroz/port-uvr-deecho-onnx (commit 02cd199), a NumPy port of the reference flow in https://github.com/nomadkaraoke/python-audio-separator
- Modifications: vendored from `driver/` with only its internal imports rewritten.

### RetinaFace priors and decoding

- Ported into: app/services/engines/face_detect.py
- License: MIT
- Copyright: Copyright (c) 2024 Yakhyokhuja Valikhujaev
- Copyright: Copyright (c) 2019 biubug6 (the upstream LICENSE names no holder)
- Source: https://github.com/yakhyo/retinaface-pytorch (commit 7601e1c)
- Source: https://github.com/biubug6/Pytorch_Retinaface (commit b984b4b), the prior box and decoding scheme yakhyo builds on
- Modifications: the `cfg_re34` prior boxes, the box and landmark decoding, the NMS and the BGR mean of `detect.py` were rewritten in NumPy; candidates below the score threshold are dropped before NMS.

### facexlib face alignment and paste

- Ported into: app/services/face_geometry.py
- License: MIT
- Copyright: Copyright (c) 2020 Xintao Wang
- Source: https://github.com/xinntao/facexlib (commit 260620a), `facexlib/utils/face_restoration_helper.py`
- Modifications: the FFHQ-512 template, the LMEDS similarity alignment with the gray border and the square template mask were rewritten; the paste warps only the face's bounding box, the mask is built in template space, and the inverse transform adds 0.5 * (scale - 1) instead of 0.5 * scale to land on the upscaled pixel centers.

### DDColor pre- and post-processing

- Ported into: app/services/engines/colorize.py
- License: Apache-2.0
- Copyright: Copyright (c) 2023 Alibaba (DAMO Academy, Alibaba Group)
- Source: https://github.com/piddnad/DDColor (commit 2adb63f, Apache-2.0 since d695d09), `ddcolor/pipeline.py`
- Modifications: the gray Lab(L, 0, 0) model input and the join of the predicted ab with the photo's lightness were rewritten in NumPy; the ab is clipped to +-110 and resized with bicubic instead of nearest, and the join works in linear light with the photo's own luminance, pulling out-of-gamut colors toward the gray of the same luminance so the lightness is kept.
