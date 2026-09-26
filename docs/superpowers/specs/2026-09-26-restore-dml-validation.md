# Validación DML de los modelos de restauración — 2026-09-26

- Device: `dml:0` · onnxruntime 1.24.4 · presupuesto por llamada 1200 ms (la mediana tiene que quedar < 600 ms)
- Modelos: `C:\personal\port-restore-onnx\dist`
- Sesiones con `create_session(..., prefer_native=False)`; si ORT no abre el grafo en el device o lo abre en CPU sin avisar, se reintenta con `ORT_DISABLE_ALL` y después sin la fusión de grafo de DML (columna Sesión).
- Nodos en CPU por perfilado de ORT; calibración TDR por precisión y canario contra DML fp32 y CPU fp32 (spec §3.5 y §6.1).
- Generado por `scripts/spike_restore_dml.py`.

## Resultados

| Modelo | Precisión | Archivo | SHA-256 | Providers | Nodos en CPU | Tile | ms/Mpx | Mediana ms | vs DML fp32 | vs CPU fp32 | NaN/Inf | vram_factor | Sesión | Veredicto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ddcolor-tiny | fp16 | ddcolor-tiny-fp16.onnx | c5d142b82894fa45ce9f2f0762bfe3ef9ef280f3e3eb8b5b53f4749fc3b8d9d7 | DmlExecutionProvider, CPUExecutionProvider | 0 | 512 | 354 | 91 | n/d | ΔE p99 86127.5 | no | 7.46 | sin fusión DML | fp16 descartado: canario vs CPU fp32 ΔE p99 86127.5 (umbral ΔE p99 2.0) |
| drunet-color | fp32 | drunet-color.onnx | 3f9e22b92617eb1dc476a4ff74ede0acf8f29a33aca4ca532a5b13f770c365e6 | DmlExecutionProvider, CPUExecutionProvider | 0 | 512 | 381 | 59 | n/d | 133.9 dB | no | 5.20 | por defecto | ok |
| drunet-color | fp16 | drunet-color-fp16.onnx | a2a3d3fce4c219b07c161666617f471465ea4567fe3edde4acca5f2d35b53ba5 | DmlExecutionProvider, CPUExecutionProvider | 0 | 512 | 255 | 39 | 67.9 dB | 67.9 dB | no | 5.31 | por defecto | ok |
| drunet-deblock-color | fp32 | drunet-deblock-color.onnx | 00c4bacf6b8635798c4da382e10b4e49b5672ceca37a0ee6bb9fe903c5b2681b | DmlExecutionProvider, CPUExecutionProvider | 0 | 512 | 381 | 59 | n/d | 134.6 dB | no | 5.14 | por defecto | ok |
| drunet-deblock-color | fp16 | drunet-deblock-color-fp16.onnx | 78dca7e6a1eb21407d964ca45c5edb2d2449557219ab9b9486b8b5533fd24848 | DmlExecutionProvider, CPUExecutionProvider | 0 | 512 | 258 | 39 | 67.9 dB | 67.9 dB | no | 5.31 | por defecto | ok |
| drunet-deblock-color-u8 | fp32 | drunet-deblock-color-u8.onnx | 3073e420943dda5fa196a4c84bc780b8b06251c6b36927fec64fbd8984dee32b | DmlExecutionProvider, CPUExecutionProvider | 0 | 512 | 390 | 49 | n/d | 92.0 dB | no | 17.52 | por defecto | ok |
| drunet-deblock-color-u8 | fp16 | drunet-deblock-color-u8-fp16.onnx | 94cd4d8a1b4f189d37a9d3f55bdd6f040d338ace42c570d05b1aa235d894fd76 | DmlExecutionProvider, CPUExecutionProvider | 0 | 512 | 265 | 37 | 59.2 dB | 59.2 dB | no | 34.16 | por defecto | ok |
| gfpgan-v1.4 | fp32 | gfpgan-v1.4.onnx | 0c7eb4fcf070f8eaff96608443e74f7fe9ce976ae56f1026182f1ecc4eeec265 | DmlExecutionProvider, CPUExecutionProvider | 0 | 512 | 87 | 23 | n/d | 132.9 dB | no | 4.18 | por defecto | ok |
| gfpgan-v1.4 | fp16 | gfpgan-v1.4-fp16.onnx | fd6bfc438d29bc76bf7c7b9540e4ab7b44f6af212c2b3741565dc1c69dbead80 | DmlExecutionProvider, CPUExecutionProvider | 0 | 512 | 67 | 15 | 53.5 dB | 53.5 dB | no | 7.35 | por defecto | ok |

## Valores para RestoreModelSpec

| Modelo | tile_by_precision | fp16_filename | ort_disable_all | dml_graph_fusion | vram_factor |
|---|---|---|---|---|---|
| ddcolor-tiny | {} | None | False | True | 7.46 |
| drunet-color | {'fp32': 512, 'fp16': 512} | 'drunet-color-fp16.onnx' | False | True | 5.31 |
| drunet-deblock-color | {'fp32': 512, 'fp16': 512} | 'drunet-deblock-color-fp16.onnx' | False | True | 5.31 |
| drunet-deblock-color-u8 | {'fp32': 512, 'fp16': 512} | 'drunet-deblock-color-u8-fp16.onnx' | False | True | 34.16 |
| gfpgan-v1.4 | {'fp32': 512, 'fp16': 512} | 'gfpgan-v1.4-fp16.onnx' | False | True | 7.35 |

## Cuadros enteros del grafo de video (carril IA de CCTV)

Misma regla que `tdr_start_tile`: cuadro entero si la extrapolación desde el tile mínimo da ≤ presupuesto/2; si no, el tile calibrado (el cuadro entero no se llama).

| Modelo | Precisión | Cuadro | Con padding | Previsto ms | Mediana ms | Arranque | IOBinding |
|---|---|---|---|---|---|---|---|
| drunet-deblock-color-u8 | fp32 | 1920x1080 | 1920x1080 | 809 | n/d | tiles de 512 | sí |
| drunet-deblock-color-u8 | fp32 | 960x1080 | 960x1080 | 405 | 171 | cuadro entero | sí |
| drunet-deblock-color-u8 | fp16 | 1920x1080 | 1920x1080 | 550 | 172 | cuadro entero | sí |
| drunet-deblock-color-u8 | fp16 | 960x1080 | 960x1080 | 275 | 95 | cuadro entero | sí |

## Sin validar

| Modelo | Motivo |
|---|---|
| bopbtl-scratch-detector | detector de daño: CPU por diseño (§3.4.2) |
| retinaface-r34 | detector de caras: CPU por diseño (§3.4.7) |

## Detectores en el device (informativo: en la app corren en CPU)

| Modelo | Archivo | SHA-256 | Entrada | Providers | Nodos en CPU | Mediana ms | máx. \|Δ\| vs CPU | NaN/Inf |
|---|---|---|---|---|---|---|---|---|
| bopbtl-scratch-detector | bopbtl-scratch-detector.onnx | 8921f3f74491d484fe8b0803adc84458265da7a5ae2d6bfd9eb47b8188cf8a2d | 1x1x256x336 | DmlExecutionProvider, CPUExecutionProvider | 56: Add, Div, Gather, Sub, Unsqueeze | 11 | output 4.20e-05 | no |
| retinaface-r34 | retinaface-r34.onnx | 8cb85f3be1563e577daa39fe664f494123662dafc389ca550fc88b96e2612175 | 1x3x960x1280 | DmlExecutionProvider, CPUExecutionProvider | 14: Concat, Gather, Slice, Unsqueeze | 13 | loc 9.60e-06, conf 1.79e-07, landmarks 1.38e-05 | no |
