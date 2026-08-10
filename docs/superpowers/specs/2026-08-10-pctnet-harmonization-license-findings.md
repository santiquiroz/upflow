# F3 · Armonización aprendida (PCT-Net): por qué NO entra todavía (verificado 2026-08-10)

Fase 3 del "object transfer" del Editor: reemplazar la igualación de color
clásica (Reinhard/Monge-Kantorovich en `app/services/object_transfer.py`) por un
modelo de armonización aprendida. Se investigó antes de escribir una línea de
integración, con el mismo criterio que `2026-08-05-3d-engines-license-findings.md`:
leer el `LICENSE` crudo de cada repo, no el badge del sidebar.

Tres barridos independientes coincidieron en todo lo que sigue.

## Veredicto

**No se integra en esta tanda.** El código de PCT-Net está limpio; **los pesos
publicados no lo están**, y ese es el bloqueo real.

| Capa | Estado | Detalle |
|---|---|---|
| Código PCT-Net | **Limpio** | `rakutentech/PCT-Net-Image-Harmonization`, MPL-2.0 verbatim (16.725 B), API `spdx_id: MPL-2.0`. **Sin cláusula no comercial en ningún lado**: ni en el LICENSE, ni en el README, ni hay NOTICE/THIRD_PARTY |
| Pesos PCT-Net | **Limpios del lado de Rakuten** | `PCTNet_ViT.pth` (19,3 MB) y `PCTNet_CNN.pth` (8,8 MB) commiteados EN el repo, sin licencia aparte ni carve-out — cubiertos por el mismo MPL-2.0 |
| Datos de entrenamiento | **BLOQUEANTE** | iHarmony4 incluye **HAdobe5k**, derivado de MIT-Adobe FiveK, cuya licencia se titula literalmente **"RESEARCH LICENSE"** |
| ONNX exportabilidad | **Fácil** | Ver abajo: no hay ops exóticas |

### La cláusula que bloquea

`https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobe.txt` (© 2011 Adobe
Systems Incorporated), cláusula 1, textual:

> "You may only exercise these rights granted to you solely for your own research
> purposes, and you shall not exercise any of these rights in any manner that is
> intended for or directed toward commercial advantage or monetary compensation."

`LicenseAdobeMIT.txt` repite la restricción. El repo de iHarmony4 es MIT, pero eso
cubre **el código de BCMI, no las fotos** — BCMI no puede relicenciar material de
Adobe.

Si los pesos entrenados son obra derivada de las imágenes de entrenamiento es una
pregunta legal genuinamente **no resuelta**, y ni Adobe ni los autores de
iHarmony4 dicen nada sobre modelos. Pero la cláusula restringe *ejercer los
derechos* con fines dirigidos a ventaja comercial, así que "los pesos no son
derivados" es una defensa más débil de lo que parece. Esta app se distribuye como
producto, no como experimento: la duda no se resuelve a favor.

**Cambiar de modelo NO resuelve esto.** Harmonizer, INR-Harmonization, CDTNet,
DCCF, MKL-Harmonizer y los pesos que redistribuye libcom se entrenan todos sobre
iHarmony4. Cambiar de repo cambia la licencia del código, no la exposición real.

## Alternativas evaluadas

| Proyecto | Licencia código | Pesos | ¿Comercial? |
|---|---|---|---|
| **PCT-Net** (rakutentech) | MPL-2.0 | En repo, sin términos aparte | Código sí; datos HAdobe5k |
| **MKL-Harmonizer** (maria-larchenko) | MIT | En repo (`.pt` 16,4 MB + `.tflite` + `.pte`) | Licencia limpia; mismos datos |
| **PIH** (adobe/PIH) | Apache-2.0 | Drive, **93M params (~370 MB)** | **Único que NO usa iHarmony4** (Artist Retouched Dataset), pero procedencia no declarada y pesa demasiado |
| INR-Harmonization | Apache-2.0 | Drive | Licencia ok; **0,81 it/s @1024×2048 — inusable** |
| DCCF | MPL-2.0 | Sin confirmar | Misma postura que PCT-Net |
| libcom (bcmi) | Apache-2.0 | HF, redistribuye `PCTNet.pth` con tag apache-2.0 | **El tag es una relicencia unilateral de material MPL — no confiable** |
| **Harmonizer** (ZHKKKe) | **CC BY-NC-SA 4.0** | Drive | **NO — no comercial explícito.** README: *"released under the Creative Commons Attribution NonCommercial ShareAlike 4.0 license"* |
| **CDTNet** (bcmi) | **Ninguna** (API 404) | En repo | **NO — sin concesión de derechos** |
| iSSAM (SamsungLabs) | — | — | **Repo dado de baja**: 404 en api/raw/HTML, en `SamsungLabs/` y `saic-vul/` |

También existe **un solo** ONNX publicado de PCT-Net: `pccaza/harmonizer-onnx`
(`pct_net.onnx`, 24,8 MB, 0 estrellas). **Declara MIT sobre un artefacto
MPL-2.0 ajeno: esa relicencia es inválida.** Sirve como prueba de que la
exportación funciona, no como fuente.

## Cuánto se pierde por no integrarlo

Métricas de arXiv 2511.12785 (iHarmony4 256×256, RTX 4060Ti), fMSE — menor es mejor:

| Método | fMSE | it/s @1024×2048 |
|---|---|---|
| PCT-Net | **201** | 63,7 |
| Harmonizer | 258 | 47,6 |
| MKL-Harmonizer | 438 | 137,2 |
| **Color matching clásico (Reinhard)** | **1836** | — |

El hueco es real: ~9x en fMSE. No hay que fingir que el color matching clásico
compite. Lo que sí cambia el cálculo es la **fase 2**: la costura —que es lo que
de verdad delata un pegado— ahora la arregla el inpaint por difusión diferencial,
no el color matching. F3 quedaría para la **iluminación global** del objeto, que
es una mejora menor y no la que el usuario nota primero.

## Si algún día entra: cómo

El puerto es fácil, y eso importa para no repetir la investigación:

- El backbone **no es HRNet ni un ViT de timm**: `vit_base.py` son ~23 líneas —
  `einops.Rearrange` (patch 4×4) → `Linear` a dim 256 → `nn.TransformerEncoder`
  de stock, 9 capas, 2 cabezas. La variante CNN es la UNet de iSSAM.
- La rama de baja resolución es **fija en 256×256** (`low_res_size`).
- La transformación de color **no es una MLP**: es un mapa de **12 canales**
  (matriz 3×3 + bias 3) aplicado por píxel con `movedim/view/matmul/add/clamp`.
  Sin autograd custom, sin `grid_sample`, sin ops exóticas.
- `kornia` está en `requirements` pero **solo se usa en el camino HSV**, y las dos
  configs publicadas son RGB. No es dependencia real de inferencia.
- Única fricción: un `for n in range(N)` sobre el batch y un `zip()` en `forward`.
  A batch=1 (el caso de esta app) se desenrollan y trazan limpio.

**Estrategia de export recomendada**: exportar **solo el predictor 256×256**
(entrada fija `1×4×256×256` RGB+máscara → salida `1×12×256×256`), y hacer el
upsample bicúbico del mapa de parámetros + el afín por píxel en NumPy a
resolución completa (~10 líneas). Eso elimina shapes dinámicos, el nodo `Resize`
y el loop de batch — grafo estático, seguro en ORT/DirectML.

## Caminos si se retoma

1. **Reentrenar** la arquitectura MPL-2.0 sobre datos sin HAdobe5k (HCOCO +
   HFlickr + Hday2night, o composites sintéticos propios). Son ~2-5M parámetros:
   una sola corrida de entrenamiento. Es el único camino que deja los pesos
   limpios sin perder el tier de calidad.
2. **PIH (Apache-2.0)**, si 370 MB es aceptable y se acepta que la procedencia de
   su dataset no está declarada.
3. **MKL-Harmonizer (MIT código + MIT pesos en repo)** como fallback conservador
   si lo único que preocupa es la licencia del repo y no la de los datos.

En los tres casos conviene meter la armonización detrás de una interfaz con el
color matching clásico como implementación por defecto, para que el backend
aprendido sea un archivo reemplazable y licenciable por separado.

## Notas menores verificadas

- `microsoft/DirectML` se autodescribe hoy como **en modo mantenimiento**. No
  afecta a PCT-Net en particular, pero pesa en cualquier decisión de backend.
- Los términos individuales de HCOCO / HFlickr / Hday2night quedaron **sin
  confirmar**; por defecto no son permisivos. HCOCO son fotos de Flickr vía COCO
  (anotaciones CC BY 4.0, imágenes sin concesión) y HFlickr no declara términos.
