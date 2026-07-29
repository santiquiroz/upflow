# Gestor unificado de capacidades y modelos — Design

**Fecha:** 2026-07-29
**Estado:** Draft para revisión (implementación en otra sesión)

## Motivación

La sección Models tiene hoy dos buscadores distintos, dos caminos de instalación y ningún lugar donde el usuario descubra qué puede hacer la app. Y las mejoras de descubrimiento que se construyeron en v0.15.0-v0.16.0 —badges de compatibilidad detectada, pre-flight con disco y VRAM por dispositivo, avisos en vez de bloqueos, picker de precisión y de checkpoint— viven **solo** en el camino de generación de imágenes, aunque nada en ellas es específico de text-to-image.

La propuesta del usuario es organizar todo como un árbol por tarea: primero qué querés hacer, después con qué. Es la dirección correcta. Pero el inventario del código dice que hay un problema estructural debajo que el árbol solo, por sí mismo, no resuelve.

## Inventario real (medido el 2026-07-29)

### Coexisten dos regímenes de gestión de modelos

| Régimen | Dominios | Mecanismo | ¿Visible en la app? |
|---|---|---|---|
| **Registro + instalador** | Upscalers (`builtin-ncnn`, `onnx`), generación (`diffusion-onnx`) | `ModelRegistry`, `ModelInstaller` / `GenerationModelInstaller`, búsqueda en Hugging Face, install jobs con progreso | **Sí**: buscar, instalar, borrar |
| **Vendorizado + scripts** | RIFE y GMFSS (interpolación), DeepFilterNet y RNNoise (ruido), Apollo y AudioSR (restauración), ffmpeg | `scripts/download-*.ps1` → `vendor/<nombre>/`, referenciado por settings (`RIFE_BINARY`, `GMFSS_MODEL_DIR`, `APOLLO_RESTORE_MODEL`…) | **No**: manual, fuera de la app |

`ModelKind` tiene exactamente tres valores (`builtin-ncnn`, `onnx`, `diffusion-onnx`). El registro **no tiene concepto** de modelo de audio ni de interpolación.

**Consecuencia para el árbol:** una rama como "mejora de video → generación de fotogramas" abriría un nodo sin botón de instalar, porque RIFE no vive en el registro. El árbol construido sobre los dos regímenes tal como están se sentiría partido a la mitad.

### Qué existe de las quince hojas propuestas

| Rama propuesta | Estado | Dónde vive |
|---|---|---|
| video → reescalado | **implementado** | registro |
| video → generación de fotogramas | **implementado** (RIFE, GMFSS) | vendorizado |
| video → generación de subtítulos | **no implementado** | roadmap del README, sin marcar (whisper.cpp) |
| imágenes → reescalado | **implementado** | registro, mismo motor que video |
| audio → reducción de ruido | **implementado** (DeepFilterNet, RNNoise) | vendorizado |
| audio → restauración | **implementado** (Apollo, AudioSR) | vendorizado |
| audio → mejora de voz | **se solapa** con reducción de ruido | — |
| audio → separación de stems | **no implementado** | nada en el árbol de código |
| generación → texto a imagen | **implementado** | registro |
| generación → imagen a imagen | **no implementado** (las clases ORT img2img sí existen) | — |
| generación → imagen a 3D | **no implementado** | — |
| generación → texto a 3D | **no implementado** | — |
| generación → texto a video | **no implementado** | — |
| generación → video a video | **no implementado** | — |
| generación → texto a sonido | **no implementado** | — |
| generación → sonido a sonido | **no implementado** | — |

Seis implementadas, nueve no, y una ambigua. El árbol tiene que tratar "no implementado" como un estado de primera clase, no esconderlo ni prometerlo.

### La app está organizada por superficie, no por tarea

Rutas actuales: `/` (Enhance: imagen y video), `/audio`, `/generate`, `/models`, `/realtime`, `/settings`, `/users`. Cada superficie tiene su panel y su forma de elegir modelo. El usuario tiene que saber de antemano en qué pestaña vive lo que quiere.

## Decisiones de diseño propuestas

### 1. La hoja del árbol es una *capacidad*, no un modelo

Una capacidad es la unidad que el usuario reconoce ("reescalar video"). Declara:

```python
@dataclass(frozen=True)
class Capability:
    id: str                      # "video.upscale"
    domain: str                  # "video" | "image" | "audio" | "generate"
    label: str
    status: CapabilityStatus     # available | needs_setup | not_implemented
    provisioning: Provisioning   # registry | vendored_pack | none
    job_kind: str | None         # a qué tipo de job alimenta
    unavailable_reason: str | None
```

El árbol se **deriva** de un catálogo de capacidades, no se escribe a mano en el frontend. Un endpoint nuevo (`GET /api/v1/capabilities/tree`) lo expone ya resuelto contra el estado real de la máquina: si `vendor/rife/` no existe, esa capacidad viene `needs_setup` y no `available`.

Esto es lo que evita el árbol cosmético: el frontend no puede mentir sobre lo que hay, porque no decide.

### 2. Unificar los dos regímenes antes de unificar la UI

Sin esto, el árbol es fachada. La propuesta mínima:

- `ModelKind` gana los valores que faltan (interpolación, audio), o —mejor— se separa el concepto: un `ModelEntry` describe **un archivo instalado**, y una `Capability` describe **qué se puede hacer**. Un modelo puede servir a varias capacidades (RealESRGAN sirve a imagen y a video).
- Los paquetes vendorizados entran al registro como entradas de tipo `vendored_pack`, con su script de descarga como método de provisión. El botón "Descargar" de la UI ejecuta el mismo script que hoy se corre a mano, con progreso y con el mismo manejo de errores que un install de Hugging Face.
- El estado se deriva de disco (¿existe `vendor/rife/rife-v4.6/`?), no de un flag persistido, para que borrar la carpeta a mano no deje la UI mintiendo.

### 3. "No implementado" se muestra, con el motivo, sin prometer fecha

Las nueve hojas sin implementar son valiosas como mapa de lo que la app quiere ser. Se muestran **inertes y explicadas**: sin botón, con una línea de por qué. Nunca "próximamente" a secas, que es una promesa sin fecha.

El motivo va como **texto persistente**, no como tooltip. Es la única crítica concreta que la investigación de progressive disclosure arrojó: usar hover como único mecanismo de revelación para información crítica es un fallo de accesibilidad conocido.

Ejemplo de redacción honesta: *"Separación de stems: no implementado. Necesita un modelo tipo Demucs y un motor de inferencia que hoy la app no incluye."*

### 4. Las mejoras de búsqueda se generalizan por estrategia, no copiando código

Lo construido en v0.15.0-v0.16.0 es casi todo genérico. Lo único específico de difusión es la clasificación:

| Pieza | Genérica hoy | Qué hace falta |
|---|---|---|
| Pre-flight (disco, VRAM por dispositivo, RAM) | **sí** | nada |
| Avisos en vez de bloqueos, `null` nunca avisa | **sí** | nada |
| Badge de compatibilidad | shell genérico | una **estrategia por dominio** |
| `classify` (diffusers) | no, es de difusión | hermana para upscalers |
| Picker (precisión / checkpoint) | específico | generalizar a "opciones de instalación" |

La propuesta: una interfaz `CompatStrategy` con una implementación por dominio. La de generación ya existe (`generation_compat` + `generation_single_file`); la de upscalers hoy es la validación implícita dentro de `model_installer` y se extrae. El pre-flight, los avisos y la tarjeta se comparten.

**Regla que no se negocia al generalizar:** el botón de instalar no gana una rama `disabled`. Es el requisito central que v0.15.0 fijó y que sus tests protegen.

### 5. Video: la pila de pasos y los presets no son alternativas

La propuesta del usuario es que al soltar un video se puedan ir agregando filtros y mejoras. La observación de las herramientas del rubro (Corel, CapCut, Kapwing) es que todas son **biblioteca de presets primero, ajuste fino después** — no pilas crudas.

Upflow ya tiene el catálogo de perfiles de video. La síntesis que propongo:

```
[ suelta un video ]

Perfil:  Anime Balanced 2x        ← punto de entrada, como hoy
         ↓ (rellena la pila)

Pasos:   1. Reescalar    RealESRGAN AnimeVideo v3 · x2      [quitar]
         2. Interpolar   RIFE v4.6 · 2x fps                 [quitar]
         3. Audio        DeepFilterNet                      [quitar]
         + agregar paso

Salida:  libx264 · CRF 17 · mp4
```

El perfil **rellena** la pila en vez de reemplazarla. Se puede quitar un paso, agregar otro, reordenar donde el orden importe. El catálogo de perfiles existente se conserva íntegro y gana un rol nuevo: plantilla de pila.

Esto además hace visible algo que hoy está implícito: que un job de video ya es una cadena de etapas.

### 6. La navegación pasa a ser por tarea, conservando los módulos

Hoy: `/` Enhance (imagen + video), `/audio`, `/generate`. Propuesta: una entrada única que pregunta la tarea y **rutea a la superficie que ya existe**, con el estado preseleccionado. Los módulos no se reescriben; se les agrega una puerta de entrada común.

Concretamente `/` pasa a ser el selector de tarea, y las superficies actuales quedan en `/enhance/video`, `/enhance/image`, `/audio`, `/generate`. `/models` deja de ser "los modelos" y pasa a ser "el catálogo de capacidades y sus modelos", con el mismo árbol.

## Riesgos y lo que puede salir mal

- **El árbol crece más rápido que la implementación.** Nueve hojas inertes contra seis vivas es una proporción incómoda: la app puede parecer más vacía de lo que está. Mitigación: agrupar las no implementadas bajo un encabezado explícito de mapa de ruta en vez de intercalarlas con las vivas.
- **Traer lo vendorizado al registro toca código que hoy funciona.** RIFE, GMFSS, DeepFilterNet, Apollo y AudioSR están cableados por settings desde hace muchas versiones. El riesgo de regresión es real y la mitigación es que la provisión por script quede como el mismo script, no una reimplementación.
- **`ModelKind` es persistido.** Está en `providers.json`/registro en disco; agregarle valores necesita migración o tolerancia a valores desconocidos al leer.
- **La ambigüedad de "mejora de voz"** hay que resolverla con el usuario antes de implementar: hoy no se distingue de reducción de ruido.

## Fuera de alcance de este spec

Implementar cualquiera de las nueve capacidades faltantes. Este spec cubre el **gestor y la navegación**; cada capacidad nueva es su propio trabajo con su propia investigación de viabilidad (y varias, como texto a 3D o texto a video, tienen el mismo problema de techo de runtime que ya se documentó para FLUX.2 y Z-Image: no alcanza que exista el modelo, tiene que existir el camino ONNX).
