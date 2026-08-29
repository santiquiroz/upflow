# Artefactos de modelos — URLs verificadas 2026-08-28

Verificación: HF tree API (size + sha256 LFS), HEAD en cada resolve URL, licencia en origen. Para registrar en `download-*.ps1` (patrón `File/Files + BaseUrl + Sha256 + Size + Label`).

| Modelo | Archivo | URL directa | Bytes | sha256 | Licencia en origen |
|---|---|---|---|---|---|
| BS-RoFormer-SW 6-stem (F1b) | model_BandSplit-Roformer_SW_by-jarredou.ckpt | https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/model_BandSplit-Roformer_SW_by-jarredou.ckpt | 699412152 | 24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e | MIT (card del re-host; autor original no declaró) |
| — config | config_BandSplit-Roformer_SW_by-jarredou.yaml | https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/BandSplit/config_BandSplit-Roformer_SW_by-jarredou.yaml | 4613 | - | idem |
| KARA_2 lead/backing (F2a) | UVR_MDXNET_KARA_2.onnx | https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/UVR_MDXNET_KARA_2.onnx | 52786726 | bf32e15105a09c0f7dddd2b67346146334d6f3ecb399ed7638eba2ab07cbf5f4 | MIT (re-host); alt oficial TRvlvr/model_repo sin licencia |
| Mel-RoFormer Karaoke aufr33/viperx (F2 upgrade, requiere export) | mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt | https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/MelBand/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt | 913096801 | 1de20d459332fe8869aeb01327a31df0032262706e1365114e852dc271779813 | MIT (re-host) |
| — config | config_melband_roformer_karaoke_aufr33_viperx_sdr_10.1956.yaml | https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Roformer/MelBand/config_melband_roformer_karaoke_aufr33_viperx_sdr_10.1956.yaml | 1726 | - | idem |
| WeSpeaker CAM++-LM (F2 embeddings, preferido: Apache-2.0, dim 512) | voxceleb_CAM++_LM.onnx | https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM/resolve/main/voxceleb_CAM%2B%2B_LM.onnx | 29292449 | 1068e4ac3a76bb9c769e6816ef30bf89363f6e966f1d938210cb8ed4038f8e93 | Apache-2.0 |
| WeSpeaker ResNet34-LM (alt, dim 256, fbank 80, 16kHz) | voxceleb_resnet34_LM.onnx | https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx | 26530309 | 7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068 | CC-BY-4.0 |
| Basic Pitch AMT (F3a) | nmp.onnx | https://raw.githubusercontent.com/spotify/basic-pitch/main/basic_pitch/saved_models/icassp_2022/nmp.onnx | 230444 | 2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec | Apache-2.0 |
| MDX23C DrumSep kick/snare/toms/hh/ride/crash (F1b) | MDX23C-DrumSep-aufr33-jarredou.ckpt | https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDX23C/MDX23C-DrumSep-aufr33-jarredou.ckpt | 437652699 | d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0 | MIT (re-host; release original de jarredou BORRADO) |
| — config | config_drumsep_mdx23c.yaml | https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDX23C/config_drumsep_mdx23c.yaml | 2417 | - | idem |

Notas críticas:
- `jarredou/BS-ROFO-SW-Fixed` (HF) y `github.com/jarredou/models` YA NO EXISTEN. Mirror byte-idéntico confirmado por sha256 en Politrees y enerjazzer — integridad cruzada verificada.
- El yaml de SW NO está en ZFTurbo/MSST (listados los 45 configs) — solo en los mirrors.
- KARA_2 no tiene yaml: los params MDX (n_fft etc.) salen del JSON de UVR keyed por hash — hardcodear en el spec como con los otros MDX.
- Embeddings: preferir CAM++ (Apache-2.0, dim 512); ojo `%2B%2B` en la URL.
- Nada gated hoy; re-verificar HEAD al implementar cada pack.
- Licencia honesta para UI: "MIT (re-host Politrees; autor original sin licencia declarada — uso comercial a criterio del usuario)" para SW/KARA_2/Karaoke/DrumSep.
