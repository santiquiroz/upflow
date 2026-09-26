# Linaje de datos de los modelos de P4 — 2026-09-26

Completa la fila "a verificar" de la tabla de linaje de §3.7 (tarea P4-LIC). Cada dataset se verificó
en su **fuente primaria** (la página o el repositorio de los autores) el 2026-09-26. El catálogo
en código es `TRAINING_DATASETS` y `P4_MODEL_DATASETS` de `app/services/restore_models.py`;
`lineage_of(...)` da el valor de `RestoreModelSpec.data_lineage` y `excluded_uses_of(...)` los usos
que la licencia de los datos prohíbe. Los tests de `tests/test_restore_models.py` comprueban que
cada dataset y cada checkpoint del catálogo aparezcan en este documento.

## Criterio

- **Sin deuda** (`none (permissive training data)`): la fuente primaria aplica una licencia permisiva
  **a los datos** con palabras explícitas. CC BY 4.0 exige atribución: la tarjeta del pack cita el dataset.
- **D1a**: cláusula research-only o no comercial sobre las imágenes, **o** datos sin licencia, **o**
  una licencia declarada que no dice si cubre las imágenes (`unclear-scope`). Este último caso se trata
  como D1a de forma conservadora: si el autor confirma por escrito que la licencia cubre los datos,
  el dataset pasa a "sin deuda" cambiando solo su `clause`.
- **D1b**: cláusulas ND o SA de la compilación. **D1c**: datos no declarados.

## Datasets

| Clave | Dataset | Fuente primaria | Qué dice (cita) | Tipo de cláusula | Decisión |
|---|---|---|---|---|---|
| `gopro` | GoPro (Nah et al., CVPR 2017) | https://seungjunnah.github.io/Datasets/gopro.html | "GOPRO dataset is released under CC BY 4.0 license." | permisiva (CC BY 4.0) | sin deuda; atribución |
| `sidd` | SIDD (Abdelhamed et al., CVPR 2018) | https://abdokamel.github.io/sidd/ | "The dataset and the associated code repositories are under the MIT License." | permisiva (MIT) | sin deuda; conservar el aviso |
| `dpdd` | DPDD (Abuolaim y Brown, ECCV 2020) | https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel | El README no declara licencia para el dataset (solo pide citar el paper si se usan "our dataset or code"). El `LICENSE` del repo es MIT con redacción de software. Las fotos las capturaron los autores (Canon EOS 5D Mark IV). | alcance dudoso | D1a (pasa a sin deuda si el autor confirma que el MIT cubre las imágenes) |
| `davis-2017` | DAVIS 2017 (Pont-Tuset et al.) | https://davischallenge.org/davis2017/code.html | La página de descarga no declara términos para los cuadros. El README del paquete de evaluación (https://github.com/davisvideochallenge/davis-2017) dice "DAVIS is released under the BSD License", junto a un `LICENSE` BSD-3 de software ("Copyright (c) 2016, Federico Perazzi"). Ninguna de las dos fuentes dice de dónde salen los videos. | alcance dudoso | D1a |
| `reds` | REDS (Nah et al., CVPRW 2019) | https://seungjunnah.github.io/Datasets/reds.html | "REDS dataset is released under CC BY 4.0 license." | permisiva (CC BY 4.0) | sin deuda; atribución |
| `lol` | LOL (Wei et al., BMVC 2018) | https://daooshee.github.io/BMVC2018website/ | La página del proyecto ofrece la descarga (Google Drive y Baidu) sin licencia ni términos. El repo del código (`weichen582/RetinexNet`) es MIT, pero no habla del dataset. | sin licencia | D1a |
| `sa-1b` | SA-1B (Kirillov et al., 2023) | https://ai.meta.com/datasets/segment-anything/ | La página de Meta: "Intended Use Cases: Research purposes only", "License: Limited; see full license language for use", "The images are licensed from a large photo company". El texto de la *SA-1B Dataset Research License* se leyó en una copia literal (https://huggingface.co/datasets/xiuqhou/SA-Det-100k/blob/main/LICENSE), porque la página de descargas de Meta se genera con JavaScript: "'Research Purposes' means ... on a non-commercial basis" y "You may not use, modify, copy, reproduce, create derivative works of, or distribute the Licensed Content (or any derivative works thereof) ... for (i) any commercial or production purpose, (ii) purposes of surveillance, including any research or development relating to surveillance, (iii) biometric processing, ... (v) attempting to identify any individual". | research-only + usos prohibidos | D1a; **nunca en CCTV** |
| `mit-adobe-fivek` | MIT-Adobe FiveK (Bychkovsky et al., CVPR 2011) | https://data.csail.mit.edu/graphics/fivek/ | "You can use these photos for research under the terms of the following licenses". `LicenseAdobe.txt`: "solely for your own research purposes, and you shall not exercise any of these rights in any manner that is intended for or directed toward commercial advantage or monetary compensation". | research-only | D1a |
| `imagenet` | ImageNet | https://image-net.org/download.php | "Researcher shall use the Database only for non-commercial research and educational purposes." | research/NC | D1a |
| `sid` | See-in-the-Dark (Chen et al., CVPR 2018) | https://github.com/cchen156/Learning-to-See-in-the-Dark | El README dice "If you use our code and dataset for research, please cite our paper" y después "License: MIT License.", sin decir si el MIT cubre las imágenes. | alcance dudoso | D1a |

## Checkpoints candidatos de P4

Los datos de cada checkpoint salen de su repo oficial (README y archivos de opciones de entrenamiento). También cuentan las redes preentrenadas que el entrenamiento usa en la pérdida, con el mismo criterio que el ArcFace de GFPGAN en §3.7.

| Checkpoint | Repo oficial (licencia del código) | Datos de entrenamiento | `data_lineage` | Usos excluidos |
|---|---|---|---|---|
| `nafnet-gopro` | megvii-research/NAFNet (MIT, "Copyright (c) 2022 megvii-model"; GitHub lo muestra como NOASSERTION) | GoPro | none (permissive training data) | — |
| `nafnet-sidd` | megvii-research/NAFNet | SIDD | none (permissive training data) | — |
| `restormer-motion-deblur` | swz30/Restormer (MIT) | GoPro (`Motion_Deblurring/README.md`) | none (permissive training data) | — |
| `restormer-defocus-deblur` | swz30/Restormer | DPDD (`Defocus_Deblurring/README.md`) | D1a | — |
| `restormer-real-denoise` | swz30/Restormer | SIDD (`Denoising/README.md`, parte de ruido real) | none (permissive training data) | — |
| `fastdvdnet` | m-tassano/fastdvdnet (MIT) | "The 2017 DAVIS dataset was used for training." | D1a | — |
| `realbasicvsr` | ckkelvinchan/RealBasicVSR (Apache-2.0) | REDS + pérdida perceptual `vgg_type='vgg19'` (VGG19 de torchvision, ImageNet) según `realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds.py` | D1a | — |
| `retinexformer-lol-v1` | caiyuanhao1998/Retinexformer (MIT) | LOL-v1 (`LOL_v1.pth`) | D1a | — |
| `retinexformer-fivek` | caiyuanhao1998/Retinexformer | MIT-Adobe FiveK (`FiveK.pth`) | D1a | — |
| `retinexformer-sid` | caiyuanhao1998/Retinexformer | SID (`SID.pth`) | D1a | — |
| `mobilesam` | ChaoningZhang/MobileSAM (Apache-2.0) | "trained ... with 100k datasets (1% of the original images)" de SA-1B, destilado de SAM (entrenado en SA-1B) | D1a | vigilancia, procesamiento biométrico, identificar personas |

## Consecuencias

1. **Sin deuda de datos** quedan NAFNet (GoPro y SIDD) y Restormer (desenfoque por movimiento y ruido real). Son los únicos modelos de restauración verificados hasta hoy que siguen disponibles aunque D1a = no. Obligación: atribución CC BY 4.0 de GoPro y REDS, y aviso MIT de SIDD, en la tarjeta del pack y en `THIRD_PARTY_NOTICES.md` si algún día se bundlean.
2. **MobileSAM (P4-SAMCLICK) no puede usarse en CCTV**: la licencia de SA-1B prohíbe explícitamente la vigilancia y la investigación o desarrollo relacionados con ella, además del procesamiento biométrico. Es la misma exclusión que ya tienen BlazeFace y Face Mesh; `excluded_uses_of(P4_MODEL_DATASETS["mobilesam"])` la expone para que el carril CCTV la aplique.
3. **Tres datasets bajan a "sin deuda" con un correo**: DPDD, DAVIS y SID declaran MIT/BSD con redacción de software, pero no dicen si cubren las imágenes. Una confirmación escrita de los autores mueve `restormer-defocus-deblur` y `retinexformer-sid` a "sin deuda"; `fastdvdnet` además necesita saber de dónde salen los videos de DAVIS.
4. **LOL no tiene licencia**: sin permiso expreso, no se concede nada. Los checkpoints de LOL (Retinexformer y afines) quedan en D1a.
5. **FiveK, ImageNet y SA-1B** son research-only explícitos: D1a sin vuelta atrás.

## Pendiente (fuera de P4-LIC)

- **SCI**: el README dice que usa 500 pares de MIT-Adobe FiveK **o** de LSRW, pero no dice qué checkpoint (`easy`, `medium`, `difficult`) sale de cada uno. La licencia de LSRW no está verificada. SCI no entra en `P4_MODEL_DATASETS` hasta aclararlo.
- **Retinexformer** con LOL-v2, SMID, SDSD o NTIRE 2024: no verificados. `flyywh/SGM-Low-Light` (LOL-v2) no tiene `LICENSE` en GitHub.
- **RealBasicVSR**: el entrenamiento arranca de un SPyNet preentrenado (`spynet_pretrained`, bajado de OpenMMLab). Hay que verificar con qué datos se entrenó ese checkpoint al elegirlo. SPyNet se suele entrenar con Flying Chairs, que es research-only ("Any commercial use is prohibited", https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html), así que no cambiaría la decisión (sigue D1a).
- **Restormer** de lluvia (Rain13K) y de ruido gaussiano (DIV2K, Flickr2K, WED y BSD): no son candidatos de P4. El linaje de DIV2K, BSD y WED ya está en §3.7 (D1a).
- **FastDVDnet**: el README termina con "The sequences are Copyright GoPro 2018". Se refiere a secuencias de prueba (Set8), no al entrenamiento, pero no hay que redistribuirlas.
