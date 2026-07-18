# SP13 — AudioSR core (versatile_audio_super_resolution) ONNX/DirectML

**Objetivo:** portar AudioSR (haoheliu/versatile_audio_super_resolution, MIT) a ONNX multi-EP (DirectML/CPU/CUDA) y
correrlo dentro de Upflow como un **segundo motor de restauración de audio** junto a Apollo. AudioSR es
super-resolución de audio general (cualquier banda → 48 kHz) por difusión latente — mucho más grande y capaz que
Apollo, y (clave) **compute-bound → acelera bien en GPU** (3.5–8× medido en los 2 grafos ya exportados), a diferencia
de Apollo que era dispatch-bound (~1.4×).

**Estado de partida (spike ya hecho — NO rehacer):** el mapa técnico completo está en
`~/Desktop/upflow-audio-spike/apollo-port-code/AUDIOSR_PORT_PLAN.md`. Descompone el modelo en 5 grafos ONNX + un
driver Python. **2 de 5 grafos ya exportados y validados en la RX 7800 XT** (VAE decoder rel-err 0.000 @247ms 8×;
vocoder HiFi-GAN rel-err 0.000 @1261ms 3.5×). Baseline CPU medido: build 12s + 48s de inferencia para 10.24s @ 25
pasos DDIM, UNet 258M params. Este plan es la ejecución de lo que queda, no el research.

---

## Decisión de repos (la pregunta): DOS repos, split por conflicto de dependencias

El port tiene dos mitades con entornos INCOMPATIBLES:

| Mitad | Deps | Dónde |
|---|---|---|
| **Toolkit de port** (exportar pesos→ONNX + validar parity) | `audiosr==0.0.7` pinnea **numpy==1.23.5**, matplotlib 3.7.5, setuptools<81, torch, librosa 0.9.2 | **REPO NUEVO** `audiosr-onnx` |
| **Runtime** (cargar los .onnx + driver DDIM/CFG/mel en la app) | onnxruntime + **numpy≥1.26** + scipy + soundfile (deps que Upflow YA tiene) | **DENTRO de Upflow** |

**Por qué un repo nuevo para el toolkit (no es opcional):** Upflow pinnea `numpy>=1.26,<3`; audiosr 0.0.7 exige
`numpy==1.23.5`. No coexisten en el mismo venv. El toolkit además arrastra torch + el paquete audiosr con sus gotchas
de Windows (torchaudio→soundfile monkeypatch, etc.). Es un **generador de artefactos de una sola vez** + un aporte
open-source valioso (primer port ONNX de AudioSR, MIT) → merece su propio repo. Ya vive de facto en
`~/Desktop/upflow-audio-spike/`; formalizarlo es mover eso a `github.com/santiquiroz/audiosr-onnx`.

**Por qué el runtime va en Upflow (no repo nuevo):** el runtime solo necesita cargar 5 .onnx y hacer el loop DDIM +
CFG + mel/STFT en numpy — todo compatible con las deps actuales de Upflow (sin torch, sin audiosr, sin conflicto de
numpy). Reusa TODA la infra existente: `onnxruntime-directml`, cache de sesión, chunking TDR-safe, el `AudioJobManager`,
el `AudioPipeline`, la sección Audio del frontend, y el bundling de modelos en el instalador (igual que Apollo).

**Contrato entre repos:** `audiosr-onnx` produce N archivos `.onnx` + un `manifest.json` (shapes, sample rate, config
del scheduler, mel basis). Upflow los vendorea en `vendor/audiosr/` (gitignored, bajados/bundleados como Apollo) y el
runtime los consume. El único acoplamiento es ese set de artefactos + sus shapes/constantes documentadas.

---

## Repo 1 — `audiosr-onnx` (toolkit de port)

**Meta:** de los pesos oficiales de audiosr 0.0.7 → 5 grafos ONNX validados contra el baseline PyTorch (rel-err < 1e-3).

**Descomposición (del AUDIOSR_PORT_PLAN):**
1. **UNet / `ddpm`** — el core + riesgo principal (258M). `DiffusionWrapper.forward` → `UNetModel.forward(x[B,32,T,32],
   t[B]) → [B,16,T,32]`. Config: model_channels 128, mult [1,2,3,5], attn_res [8,4,2], res_blocks 2. **Exportar primero
   los 2 fáciles, este segundo.**
2. **VAE decoder** — ✅ HECHO (509MB, rel-err 0.000, scale_factor horneado).
3. **Vocoder HiFi-GAN** — ✅ HECHO (726MB, rel-err 0.000).
4. **`vae_feature_extract`** — el conditioner: `vae.encode(lowpass_mel).sample()` → latente cond 16ch. Uncond para CFG
   = constante `-11.4981`.
5. **VAE encoder** (opcional, solo si hace falta init en espacio latente).

**Tasks del toolkit:**
1. Formalizar el repo: mover el spike (`export_components.py`, los 2 .onnx validados, los `_in/_ref.npy`), pinnear el
   env exacto (numpy 1.23.5, matplotlib 3.7.5, setuptools<81, audiosr 0.0.7, torchaudio→soundfile patch), README con la
   receta de setup Windows.
2. **Exportar `vae_feature_extract`** (bajo riesgo, conv nets) + parity vs PyTorch.
3. **Exportar el UNet** (opset 17; el riesgo real — spatial-transformer + attention). Validar shapes `[B,32,T,32]→[B,16,T,32]`
   y parity numérica en CPU-EP contra `apply_model`. Si algún op no exporta limpio, mapear el corte de grafo de OpenVINO
   (mismo cut, ya de-riskeado según recon).
4. **Validar el UNet en DirectML** con chunking a 5.12s (límite de tensor largo de DML, ya visto en Apollo). Medir fp16
   acá (el UNet es compute-heavy 258M → fp16 PODRÍA ayudar, a diferencia de Apollo; medir, no asumir).
5. Emitir `manifest.json` (sample rate 48k, n_fft/win 2048, hop 480, n_mel 256, fmin 20 fmax 24000, scheduler cosine
   linear_start 0.0015 linear_end 0.0195 1000 steps, v-parameterization, cfg_scale 3.5, latent_t_per_second 12.8,
   window 5.12s) + la mel basis de librosa como `.npy`.
6. Script `download-audiosr-onnx.ps1` que baja pesos + exporta los 5 (paralelo a `download-realesrgan-onnx.ps1`).

**Salida:** `vendor-artifacts/{ddpm,vae_decoder,vocoder,vae_feature_extract}.onnx` + `manifest.json` + `mel_basis.npy`.

---

## Repo 2 — Upflow (runtime, motor nuevo `AudioSrRestorer`)

Espeja `ApolloRestorer` + el wiring de `restore=apollo`. TODO con deps actuales (numpy≥1.26, scipy, soundfile,
onnxruntime) — **sin conflicto**.

**Dominio (`app/services/engines/audiosr_restore.py`):**
1. `AudioSrRestorer(settings)` con cache de sesión por (grafo, device) — mismo patrón que Apollo (`_session_cache` +
   `_session_lock`, todo off-loop vía `asyncio.to_thread`).
2. **Front-end de mel/STFT en numpy** (del AUDIOSR_PORT_PLAN, NO en ONNX): STFT n_fft 2048 hop 480 center=False
   reflect-pad 784, log-mel C=1 con la `mel_basis.npy` del manifest. Reusar el STFT-como-matmul ya escrito para Apollo
   (`real_stft.py` del spike) adaptado a estos parámetros.
3. **Lowpass sim** (scipy butter/cheby1/ellip/bessel orden 8) + `mel_replace_ops` + la reposición de banda baja por
   STFT-librosa en postproc (ddpm.py:1567/1577) — reproducir fiel (es lo que hace que suene bien).
4. **Driver DDIM en numpy** (el corazón, NO en ONNX): loop de DDIM con scheduler cosine + **v-parameterization**
   (`predict_eps_from_z_and_v`), **CFG** (2 llamadas al UNet por paso: `uncond + 3.5*(cond-uncond)`),
   `vae_feature_extract` para el latente cond, `decode_first_stage` (z/scale) → VAE decoder → vocoder → wav.
5. **Segmentación**: ventana deslizante de 5.12s + overlap-add (nativo del modelo Y necesario por el límite de tensor
   de DirectML — reusar la lógica de chunking + crossfade de Apollo).
6. `run(input_wav, output_wav, device)` idéntico en firma a `ApolloRestorer.run` para que el pipeline no distinga.

**Aplicación (wiring, espeja Apollo):**
7. `AUDIO_RESTORE_MODES` en config: agregar `"audiosr"` junto a `"apollo"`. `ENABLE_AUDIO_SR` + `AUDIOSR_MODEL_DIR`
   (carpeta con los 5 .onnx + manifest). `audiosr_*_chunk_seconds` / throttle análogos a los de Apollo.
8. `AudioPipeline` + `VideoUpscaler._prepare_audio`: cuando `restore=="audiosr"`, usar el nuevo restorer. Ambos motores
   conviven; el usuario elige (Apollo = rápido/liviano band-restore; AudioSR = pesado/general SR de más calidad).
9. `audio_job_manager` / `routes` / `schemas`: validar `restore in {apollo, audiosr}` (ya hay allowlist para apollo).
10. **Frontend**: en la sección Audio, el selector de restore pasa de toggle a 2 opciones (Apollo / AudioSR) con aviso
    de costo (AudioSR es ~10-50× más lento que Apollo por la difusión — mostrar estimación como el SlowPresetCostHint).

**Configuración / release:**
11. `download-audiosr-onnx.ps1` en `scripts/` que baja los .onnx del release de `audiosr-onnx` (o los exporta). Los .onnx
    (~1.7GB en total: ddpm+vae_decoder+vocoder+vae_fe) son grandes → decidir bundle-en-installer (como Apollo 74MB) vs
    descarga primera-vez. **Recomendado: descarga on-demand** (1.7GB es mucho para el .exe; Apollo cabía a 74MB, esto no).
12. `package-release.ps1`: NO bundlear por tamaño; el launcher baja bajo demanda cuando el usuario activa AudioSR.

**Pruebas:**
13. Parity end-to-end: el driver runtime vs el baseline PyTorch del toolkit sobre los mismos `_ref.npy` (rel-err < 1e-3
    por componente + audible en el output final).
14. Unit tests del driver DDIM/CFG/mel (puros numpy, sin GPU) + del chunking/overlap-add. Smoke real en la 7800 XT
    (DirectML) + CPU-EP end-to-end sobre un clip degradado, comparar contra `reference.wav` del spike.

---

## Orden de ejecución (de menor a mayor riesgo)

1. **Formalizar `audiosr-onnx`** (mover spike + env pinneado + los 2 grafos ya hechos). — S
2. **Exportar `vae_feature_extract`** (bajo riesgo). — S/M
3. **Exportar + validar el UNet** en CPU-EP, después DirectML + chunk 5.12s. **El riesgo y el compute principal.** — L
4. **Driver DDIM/CFG/mel en numpy** dentro de Upflow, validado componente-a-componente contra el baseline. — L
5. **Wiring en Upflow** (motor + pipeline + config + UI + descarga on-demand). — M
6. **Smoke real end-to-end** (DirectML + CPU) vs `reference.wav`, medir fps/RTF, review adversarial. — M

## Riesgos

- **UNet no exporta limpio** (attention/spatial-transformer): fallback = el corte de grafo de OpenVINO (mismo cut, ya
  de-riskeado por recon). Si un op falla en DirectML, chunk más chico o CPU-EP para ese grafo.
- **Calidad del driver DDIM**: v-parameterization + CFG + la reposición de banda baja son sutiles; un error se oye. Por
  eso el parity componente-a-componente contra el baseline PyTorch es obligatorio antes de wire.
- **Tamaño (~1.7GB de .onnx)**: descarga on-demand, no bundle. Documentar el costo de red.
- **Latencia**: difusión = 25 pasos × 2 UNet (CFG) por ventana de 5.12s. Aun con GPU es ~decenas de segundos por minuto
  de audio. Es restauración de alta calidad, no tiempo real — comunicar el trade-off en la UI (aviso de costo).
- **fp16 del UNet**: medir (podría ayudar por ser compute-bound); no asumir (en Apollo fue callejón sin salida).

## Esfuerzo

~1-2 semanas (recon). El baseline + el mapa de fronteras + 2/5 grafos ya están hechos. Lo que queda: 2 exports (1 fácil
+ el UNet difícil) + el driver DDIM/CFG/mel from-scratch + el glue no-neural + wiring en Upflow + validación de parity.

## Self-Review

Repo split correcto (conflicto numpy 1.23.5 vs ≥1.26 obliga toolkit separado; runtime reusa deps de Upflow) ✓,
construye sobre el spike (2/5 grafos + baseline hechos) sin rehacer research ✓, motor nuevo espeja Apollo (conviven) ✓,
5 grafos ONNX + driver numpy con parity obligatorio ✓, descarga on-demand por tamaño ✓, aviso de costo en UI ✓,
riesgos con fallbacks concretos (OpenVINO cut, CPU-EP, chunk) ✓.
