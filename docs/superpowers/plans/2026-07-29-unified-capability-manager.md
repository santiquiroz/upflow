# Gestor unificado de capacidades — Plan de implementación

> **Para la próxima sesión.** Este plan se escribió el 2026-07-29 junto con
> `docs/superpowers/specs/2026-07-29-unified-capability-manager-design.md`.
> Leé el spec primero: contiene el inventario medido del que sale todo lo de acá.

**Goal:** Que la app se navegue por tarea, que el catálogo de capacidades sea la única fuente de verdad sobre qué se puede hacer, y que las mejoras de descubrimiento de v0.15.0-v0.16.0 sirvan a todos los dominios y no solo a generación de imágenes.

**Arquitectura:** Un catálogo de capacidades en el backend, derivado del estado real de la máquina, expuesto como árbol. Los dos regímenes de gestión de modelos (registro e instalador contra vendorizado y scripts) se unifican detrás de una noción de *provisión*. Las piezas genéricas del pre-flight se comparten; lo específico de cada dominio entra por una estrategia.

**Antes de empezar:** hay **una pregunta abierta para el usuario** (Fase 0). No arrancar sin resolverla.

---

## Fase 0 — Resolver la ambigüedad y confirmar el orden

- [ ] **Preguntar al usuario qué es "mejora de voz"** y en qué se diferencia de "reducción de ruido". Hoy no hay nada que las distinga: DeepFilterNet y RNNoise hacen supresión de ruido, y Apollo/AudioSR restauran. Si "mejora de voz" es un nombre alternativo para lo mismo, la rama se colapsa; si es otra cosa (ecualización, de-esser, normalización de loudness), es una capacidad nueva sin implementar.
- [ ] **Confirmar el orden de fases.** El plan propone unificar el backend antes de tocar la UI, porque un árbol sobre los regímenes actuales sería fachada. Si el usuario prefiere ver la UI primero para validar la idea, la Fase 3 puede ir antes con datos simulados, aceptando que se reescribe después.

---

## Fase 1 — Catálogo de capacidades (backend, sin cambios de UI)

Es la base de todo lo demás y se puede shipear sola: sin ella, el resto es cosmético.

### Task 1.1 — El modelo de datos

**Archivos:** crear `app/services/capabilities.py`; test `tests/test_capabilities.py`.

- [ ] Definir `CapabilityStatus = Literal["available", "needs_setup", "not_implemented"]` y `Provisioning = Literal["registry", "vendored_pack", "none"]`.
- [ ] Definir el dataclass `Capability` con los campos del spec (`id`, `domain`, `label`, `status`, `provisioning`, `job_kind`, `unavailable_reason`).
- [ ] Escribir el catálogo estático como una tupla de `Capability` **sin** el campo `status`: el status no es estático, se resuelve contra la máquina (Task 1.2). El catálogo declara la forma del árbol y qué provisión necesita cada hoja.
- [ ] Las quince hojas del spec, con las nueve no implementadas marcadas y con su `unavailable_reason` redactado. El spec tiene la tabla completa; copiar de ahí, no reinventar los motivos.
- [ ] Tests: el catálogo cubre los cuatro dominios; toda capacidad `not_implemented` tiene `unavailable_reason` no vacío; ninguna capacidad `not_implemented` declara `job_kind`; los ids son únicos.

### Task 1.2 — Resolución contra el estado real

**Archivos:** `app/services/capabilities.py`; test `tests/test_capabilities.py`.

- [ ] `resolve_capabilities(settings, registry) -> list[Capability]`, que parte del catálogo y calcula `status`.
- [ ] Para `provisioning="registry"`: `available` si el registro tiene al menos un modelo del kind que la capacidad necesita, si no `needs_setup`.
- [ ] Para `provisioning="vendored_pack"`: `available` si el path del vendor existe en disco, si no `needs_setup`. **Derivar de disco, nunca de un flag persistido** — borrar la carpeta a mano no debe dejar la UI mintiendo.
- [ ] Para `not_implemented`: el status no se calcula, se conserva.
- [ ] Tests con `tmp_path`: crear y borrar los directorios de vendor y verificar que el status sigue al disco; registro vacío da `needs_setup`; una capacidad no implementada nunca pasa a `available` por más que existan archivos.

### Task 1.3 — El endpoint del árbol

**Archivos:** `app/schemas.py`, `app/api/routes.py`; test `tests/test_capabilities_api.py`.

- [ ] `GET /api/v1/capabilities/tree` devuelve los dominios con sus capacidades resueltas, en camelCase con `serialization_alias` como el resto del archivo.
- [ ] La respuesta agrupa por dominio y **separa** las no implementadas en su propia lista dentro del dominio, para que el frontend pueda darles el encabezado de mapa de ruta que pide el spec sin filtrar por su cuenta.
- [ ] Tests llamando la corrutina directo con dobles, siguiendo el patrón de `tests/test_generation_api.py`.

### Task 1.4 — Commit

- [ ] Suite completa verde. Commit: el catálogo es la fuente de verdad, sin cambios de UI todavía.

---

## Fase 2 — Unificar la provisión de los paquetes vendorizados

Es la fase con más riesgo de regresión: toca código que funciona desde muchas versiones. La regla que la mantiene acotada es **reusar el script, no reimplementar la descarga**.

### Task 2.1 — Provisión por script, con progreso

**Archivos:** crear `app/services/pack_provisioner.py`; test `tests/test_pack_provisioner.py`.

- [ ] Un job de provisión que ejecuta el `scripts/download-*.ps1` correspondiente y reporta progreso, con el mismo ciclo de vida que un install job (queued → running → done/error).
- [ ] Mapear cada capacidad `vendored_pack` a su script. La tabla sale del inventario del spec.
- [ ] Plataforma: los scripts son PowerShell. Seguir el guard que ya usa `providers_service._start_litellm` en el proyecto hermano y que este repo aplica en otros lados: PowerShell en Windows, y en otras plataformas la capacidad queda `needs_setup` con motivo explícito en vez de fallar raro.
- [ ] Tests con el subprocess mockeado: éxito, fallo con código distinto de cero, y script inexistente.

### Task 2.2 — Endpoint de provisión

**Archivos:** `app/api/routes.py`, `app/schemas.py`; tests.

- [ ] `POST /api/v1/capabilities/{id}/provision` arranca el job; `GET .../provision/{job_id}` da estado. Mismo shape que los install jobs para que el frontend reuse los hooks.
- [ ] Rechazar con 400 una capacidad que no sea `vendored_pack`.

### Task 2.3 — No romper lo que ya andaba

- [ ] Verificar a mano que un job de interpolación con RIFE y uno de audio con DeepFilterNet siguen corriendo igual. Esta fase no cambia cómo se **usan** los modelos, solo cómo se **obtienen**.
- [ ] Suite completa verde. Commit.

---

## Fase 3 — El árbol en la UI

### Task 3.1 — Servicio y hook

**Archivos:** `frontend/src/lib/apiTypes.ts`, `frontend/src/services/capabilities.ts`, `frontend/src/hooks/useCapabilities.ts`; tests.

- [ ] Tipos del árbol, fetch del endpoint, hook con TanStack Query.

### Task 3.2 — El selector de tarea

**Archivos:** crear `frontend/src/modules/capabilities/CapabilityTree.tsx` y su test.

- [ ] Primer nivel: los cuatro dominios. Segundo nivel: las capacidades del dominio elegido.
- [ ] Una capacidad `available` navega a su superficie con el estado preseleccionado. Una `needs_setup` ofrece el botón de descargar el paquete. Una `not_implemented` se muestra **inerte con su motivo como texto persistente**, nunca como tooltip — la única crítica concreta que arrojó la investigación de progressive disclosure es que usar hover como único mecanismo para información crítica es un fallo de accesibilidad.
- [ ] Las no implementadas van bajo un encabezado explícito de mapa de ruta, no intercaladas con las vivas: nueve inertes contra seis vivas intercaladas haría parecer la app más vacía de lo que está.
- [ ] Tests: los tres estados renderizan distinto; el motivo de una no implementada está en el DOM sin interacción; una `needs_setup` dispara la provisión.

### Task 3.3 — Rutas

**Archivos:** `frontend/src/App.tsx`, `frontend/src/pages/`.

- [ ] `/` pasa a ser el selector. Las superficies actuales se mueven a `/enhance/video`, `/enhance/image`, `/audio`, `/generate`.
- [ ] **Los módulos no se reescriben.** Solo se les agrega la puerta de entrada y la capacidad de recibir estado preseleccionado.
- [ ] Los tests existentes de las páginas siguen pasando; ajustar solo los que afirmen la ruta vieja.

---

## Fase 4 — Generalizar las mejoras de descubrimiento

### Task 4.1 — Extraer la estrategia de compatibilidad

**Archivos:** crear `app/services/compat_strategy.py`; refactor de `generation_compat.py`; tests.

- [ ] Un `Protocol CompatStrategy` con `classify(filenames, gated) -> (verdict, reason)` y las opciones de instalación que el dominio ofrezca.
- [ ] La de generación se implementa envolviendo lo que ya existe, **sin cambiar su comportamiento**. Sus tests actuales son el contrato: si alguno cambia, el refactor está mal.
- [ ] Una estrategia nueva para upscalers, extrayendo la validación que hoy vive implícita en `model_installer`.

### Task 4.2 — Compartir el pre-flight

**Archivos:** `app/services/generation_preflight.py` → renombrar a `app/services/model_preflight.py`; tests.

- [ ] El pre-flight recibe la estrategia por parámetro. Disco, VRAM por dispositivo, RAM y el reporte degradado no se tocan: ya son genéricos.
- [ ] **Regla que no se negocia:** el botón de instalar no gana una rama `disabled` en ningún dominio. Es el requisito central de v0.15.0 y sus tests lo protegen.

### Task 4.3 — Tarjeta compartida

**Archivos:** `frontend/src/modules/models/`.

- [ ] Generalizar `GenerationModelCard` a una tarjeta por dominio, con badges y opciones de instalación que vienen de los datos y no del dominio hardcodeado.
- [ ] Los tests de la tarjeta de generación siguen pasando sin cambios de comportamiento.

---

## Fase 5 — Video como pila de pasos

Es la fase más independiente: se puede hacer antes o después de las 3 y 4.

### Task 5.1 — La pila en el panel de video

**Archivos:** `frontend/src/modules/enhance/VideoPanel.tsx` y su test (hoy 28 tests, son el contrato).

- [ ] El perfil elegido **rellena** la pila en vez de reemplazarla. El catálogo de perfiles se conserva íntegro y gana el rol de plantilla.
- [ ] Se puede quitar un paso, agregar otro y reordenar donde el orden importe. Reescalar antes o después de interpolar no es lo mismo: el orden es semántico, no cosmético.
- [ ] Los 28 tests actuales de `VideoPanel` tienen que seguir pasando o cambiar con justificación explícita en el commit. Fijan comportamiento real (que el perfil autoselecciona el modelo, que un override manual del ModelPicker sobrevive, que el CTA se deshabilita sin archivo o sin perfil).

### Task 5.2 — El request refleja la pila

**Archivos:** `app/schemas.py`, `app/services/video_job_manager.py`.

- [ ] Decidir si el request de video pasa a llevar una lista de pasos o se mantienen los campos actuales con la pila como azúcar del frontend. **Recomendación: empezar por lo segundo.** La pila es una mejora de UI sobre capacidades que ya existen; cambiar el contrato del job es un trabajo aparte con su propia migración.

---

## Verificación final

- [ ] `.\.venv\Scripts\python.exe -m pytest -q` verde
- [ ] `cd frontend && npm test` y `npm run build` verdes
- [ ] A mano: un job de video con RIFE, uno de audio con DeepFilterNet y una instalación de generación siguen funcionando
- [ ] `GET /api/v1/capabilities/tree` con `vendor/rife/` borrado devuelve esa capacidad como `needs_setup`, y volviéndola a crear la devuelve `available`

## Lo que este plan NO hace

Implementar ninguna de las nueve capacidades faltantes. Cada una es su propio trabajo con su propia investigación de viabilidad, y varias arrastran el mismo techo que ya se documentó para FLUX.2 y Z-Image: no alcanza que el modelo exista, tiene que existir el camino a ONNX. Texto a video y texto a 3D son los candidatos más probables a no tenerlo.
