# Personas 3/4: Editor en tres columnas, Personas, Revisar y retoque (frontend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** El Editor pasa a tres columnas y suma la herramienta Personas (elegir → pulir → revisar), el borrado automático por defecto, y las herramientas Restaurar, Clonar, Corrector puntual y Ajustar borde.

**Architecture:** `EditorPanel.tsx` (925 líneas) se parte en riel de herramientas, escenario y panel lateral contextual (Personas / Acción / Revisar). La lógica nueva vive en módulos puros testeables (`peopleState.ts`, `retouch.ts`, `zones.ts`, cambios en `editorMaskCanvas.ts`) y los componentes solo los conectan. Las ediciones de píxeles producen una nueva imagen base subida como imagen inicial (como "Seguir editando este resultado").

**Tech Stack:** React 18 + TypeScript + Vite, TanStack Query, Tailwind con los tokens de `index.css`, lucide-react 0.462, vitest + jsdom + Testing Library.

**Spec:** `docs/superpowers/specs/2026-10-30-editor-personas-design.md` (Comportamiento: pasos 1-3, Herramientas de retoque; Lo que decidió el dueño: tres columnas y borrado automático sin SDXL; criterios 5 y 6). Maquetas: `C:\personal\upflow-mockups\2026-10-30-personas\personas-flujo-v3.html` y `editor-tres-columnas.html`.

**Depende de:** Plan 1 (`/editor/people`, `/editor/segment` con `box`) y Plan 2 (`eraser-auto`, `eraseEngine`, `eraseZones`, `maskGrowPx`/`maskFeatherPx`, `/editor/heal`).

## Global Constraints

- Tema existente: `--bg`, `--surface`, `--surface-2`, `--border`, `--text`, `--text-dim`, `--text-faint`, `--accent #f5a524`, `--ok`, `--danger`; overlays del Editor rojo `#ef4444` / verde `#22c55e` como hoy. Un solo botón ámbar lleno por pantalla (el de acción del panel); la herramienta activa: fondo `--surface-2` + borde ámbar a la izquierda (`box-shadow: inset 3px 0 0 var(--accent)`).
- Íconos lucide-react con `aria-hidden="true"` y texto visible; nunca emoji; nunca color solo (Conservar = check + texto, Quitar = x + texto). Foco visible (`:focus-visible` ya global). Botones ≥ 36 px de alto. `prefers-reduced-motion` respetado (ya global en `index.css:149`).
- Textos nuevos en `frontend/src/i18n/es.ts` y `en.ts` con las mismas claves (lo exige `i18n.test.ts`), voseo en español ("Tocá", "Pulí", "Revisá").
- Código: funciones atómicas, complejidad baja, sin comentarios de documentación.
- Tests: `cd C:/wt/upflow-borrado/frontend && npx vitest run src/modules/editor` (y el archivo nuevo que toque cada tarea); los tests corren con locale `en`.
- No push ni merge; commits en español sin Co-Authored-By.

## Review Focus

- Detección que falla o tarda (red, paquete sin instalar): la herramienta Personas queda deshabilitada con motivo; el resto del Editor funciona igual y nada se bloquea mientras corre.
- Foto con 0 personas: estado vacío "No encontré personas en esta foto"; con 30+ personas la lista scrollea dentro del panel (el documento nunca scrollea, `index.css`).
- Ctrl+Z justo después de "Agregar siluetas": quita todas las siluetas de una vez; un segundo Ctrl+Z deshace lo anterior (máscara o edición de base, en el orden en que pasaron).
- Rehacer una zona cuando la imagen base cambió después del borrado (se usó Restaurar o Clonar): Rehacer usa la base actual y la máscara original recortada a la caja de la zona.
- Atajos de teclado mientras se escribe en un campo (prompt de Reemplazar, "Describir el fondo…"): no se disparan.

---

### Task 1: Tres columnas, herramienta activa y atajos

**Files:**
- Create: `frontend/src/modules/editor/EditorToolRail.tsx`, `frontend/src/modules/editor/EditorSidePanel.tsx`, `frontend/src/modules/editor/useEditorShortcuts.ts`
- Modify: `frontend/src/modules/editor/EditorPanel.tsx` (barra :568-677, controles :727-920), `frontend/src/i18n/es.ts`, `en.ts`
- Test: `frontend/src/modules/editor/EditorPanel.test.tsx`, `frontend/src/modules/editor/useEditorShortcuts.test.ts`

**Interfaces:**
- Produces:
  - `type Tool = "people" | "tap" | "protect" | "brush" | "eraser" | "lasso" | "rect" | "restore" | "clone" | "heal" | "pan"` (las nuevas se habilitan en sus tareas; hasta entonces no se renderizan).
  - `EditorToolRail({ tool, onToolChange, disabledTools: Partial<Record<Tool, string>>, zoom actions..., onUndo, onClear })` — grupos "Seleccionar" (people, tap, protect, brush, eraser, lasso, rect) y "Retocar" (restore, clone, heal), luego pan y la fila de zoom/ajustar/deshacer/limpiar; `role="radiogroup"` con `role="radio"` por herramienta (los tests actuales buscan por rol y nombre).
  - `EditorSidePanel({ view: "people" | "action" | "review", children })` — contenedor de 340 px con scroll propio.
  - `useEditorShortcuts({ onUndo, onBrushSmaller, onBrushBigger, enabled })` — Ctrl+Z / Cmd+Z, `[` y `]`; ignorado si el foco está en `input`, `textarea` o `[contenteditable]`.
  - Layout: `grid` de tres columnas (riel 148 px, escenario flexible, panel 340 px). El panel muestra "action" con los controles que hoy están debajo de la foto (Quitar/Reemplazar, modelo, dispositivo, avanzadas, Insertar objeto, botón de acción, "Usar otra imagen"); la tarjeta del trabajo y "Seguir editando este resultado" pasan al panel en vista "review".

- [ ] **Step 1: Write the failing tests** — `useEditorShortcuts`: Ctrl+Z llama `onUndo`, `[`/`]` cambian tamaño, nada dispara con foco en un `input`. `EditorPanel`: la herramienta activa tiene `aria-checked="true"` y no la clase de relleno ámbar; el botón "Remove selection" está dentro de la región del panel (`getByRole("complementary")`).
- [ ] **Step 2: Run** `npx vitest run src/modules/editor` → FAIL.
- [ ] **Step 3: Implement** — mover JSX sin cambiar comportamiento; ajustar los tests existentes solo donde cambie la ubicación, no la conducta.
- [ ] **Step 4: Run** → PASS (los 105 tests previos más los nuevos).
- [ ] **Step 5: Commit** `feat: Editor en tres columnas con riel de herramientas y atajos`

### Task 2: Borrado automático por defecto

**Files:**
- Modify: `frontend/src/modules/editor/eraseJobParams.ts` (:4-25), `EditorPanel.tsx` (`InpaintModelSelect` :69-114, `modelId` inicial), `frontend/src/lib/apiTypes.ts` (:725 `GenerationEraseParams`)
- Test: `frontend/src/modules/editor/eraseJobParams.test.ts`, `EditorPanel.test.tsx`

**Interfaces:**
- Consumes: modelo `eraser-auto` en las capacidades (Plan 2, Task 6).
- Produces:
  - `GenerationEraseParams` suma `eraseEngine?: "lama" | "osor" | "lama+osor" | "sdxl"`, `maskGrowPx?: number`, `maskFeatherPx?: number`; `buildRequestBody` los manda solo si hay `initImageToken`.
  - `defaultEraseModel(models): string | null` — `eraser-auto` si está instalado; si no, el comportamiento previo.
  - `shouldUseAutoErase` sigue para el camino viejo (SDXL elegido + LaMa); con `eraser-auto` no hace falta `eraseStrategy`.
  - `canReplace(models): boolean` — hay un SDXL de relleno instalado; si no, el modo Reemplazar queda deshabilitado con el motivo `editor.replace.needsSdxl`.

- [ ] **Step 1: Write the failing tests** — `defaultEraseModel` elige `eraser-auto`; el modo Reemplazar sin SDXL está deshabilitado con el motivo visible; el cuerpo del trabajo lleva `protectMaskToken` con `eraser-auto`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: el Editor usa el borrado automático por defecto`

### Task 3: Servicios y estado de Personas

**Files:**
- Modify: `frontend/src/services/editor.ts` (:6-40)
- Create: `frontend/src/modules/editor/peopleState.ts`, `frontend/src/modules/editor/usePeople.ts`
- Test: `frontend/src/modules/editor/peopleState.test.ts`, `frontend/src/services/editor.test.ts` (o el archivo de tests de servicios existente)

**Interfaces:**
- Produces:
  - `EditorCapabilities` suma `people: boolean`.
  - `type EditorPerson = { id: number; box: [number, number, number, number]; score: number; role: "subject" | "background"; face: [number, number, number, number] | null }`.
  - `detectPeople(imageToken: string): Promise<EditorPerson[]>`; `segmentEditorBox(imageToken: string, box: [number, number, number, number]): Promise<Blob>` (mismo manejo de errores que `segmentEditorObject`).
  - `usePeople(baseInfo: InitImageResponse | null, enabled: boolean)` — `useQuery(["editorPeople", token])`, en segundo plano al cargar la imagen.
  - `peopleState.ts` (puro): `type Choice = "keep" | "remove"`; `initialChoices(people): Record<number, Choice>` (subject → keep, background → remove); `toggleChoice(choices, id)`; `setGroup(choices, people, role, choice)`; `countRemove(choices): number`; `groupPeople(people): { front: EditorPerson[]; back: EditorPerson[] }`; `anchorPoint(person): {x, y}` (centro de la cara o del cuarto superior de la caja, como el backend).

- [ ] **Step 1: Write the failing tests** para cada función pura y para `detectPeople` (fetch mockeado: POST con `imageToken`, error → `ApiError`).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: servicios y estado de la herramienta Personas`

### Task 4: Paso 1, Elegir

**Files:**
- Create: `frontend/src/modules/editor/PeoplePanel.tsx`, `frontend/src/modules/editor/PeopleOverlay.tsx`
- Modify: `EditorPanel.tsx`, `EditorToolRail.tsx`, i18n
- Test: `frontend/src/modules/editor/PeoplePanel.test.tsx`

**Interfaces:**
- Consumes: Task 3 (`usePeople`, `peopleState`), `EditorSidePanel`.
- Produces:
  - `PeoplePanel({ people, choices, hoveredId, onHover, onToggle, onGroup, protectKept, onProtectKeptChange, onAccept, status: "loading" | "ready" | "empty" | "error" | "unavailable", imageUrl })` — esqueleto mientras carga; grupos "Al frente · N" / "Al fondo · N" con acción de grupo; fila con miniatura (recorte por CSS `background-position` de `imageUrl`, cara o cabeza), "Persona N", selector Conservar/Quitar (`role="radiogroup"`); filas enfocables, flechas arriba/abajo mueven el foco, espacio alterna; casilla "Proteger a las que conservo" (marcada por defecto); botón "Agregar N siluetas al borrado" deshabilitado con N = 0.
  - `PeopleOverlay({ people, choices, hoveredId, onHover, onToggle, scale })` — contornos punteados (verde keep, rojo remove) sobre el escenario; el resaltado (`hoveredId`) con borde ámbar y etiqueta; clic alterna.

- [ ] **Step 1: Write the failing tests** — render con 3 frente / 7 fondo muestra los dos grupos y "Add 7 silhouettes"; pasar el mouse por una fila marca la misma persona en el overlay (`data-hovered`); clic en el overlay alterna la fila; "Keep all" en fondo deja el botón deshabilitado; espacio sobre una fila enfocada alterna; estado vacío y no disponible muestran su texto.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: herramienta Personas, paso Elegir`

### Task 5: Paso 2, Pulir (siluetas como un solo paso de deshacer)

**Files:**
- Modify: `frontend/src/modules/editor/editorMaskCanvas.ts` (`HistoryEntry` :4-9, `replayMasks` :49, `maskHasContent` :113, `hasProtection` :117), `EditorPanel.tsx`, `PeoplePanel.tsx`
- Test: `frontend/src/modules/editor/editorMaskCanvas.test.ts`, `PeoplePanel.test.tsx`

**Interfaces:**
- Produces:
  - `HistoryEntry.kind` suma `"batch"` con `entries: HistoryEntry[]`; `replayMasks` aplana los lotes en orden (la regla `resolveOverlap` ve sus `protect` como cualquier otro); `maskHasContent`/`hasProtection` miran dentro de los lotes.
  - `async function silhouettesFor(imageToken, people, choices, protectKept): Promise<HistoryEntry>` — pide `segmentEditorBox` por persona (remove → `segment`, keep → `protect` si `protectKept`), cada entrada con `point: anchorPoint(person)`; devuelve un `batch`.
  - Al aceptar: se agrega el lote, la herramienta pasa a `brush`, el panel muestra el resumen "N silhouettes in the erase" con chips, el texto de pulido, "Back to choosing" (saca el lote y vuelve a la lista con las mismas elecciones) y el botón de acción de siempre.

- [ ] **Step 1: Write the failing tests** — `replayMasks` con un lote dibuja sus segmentos y protecciones; deshacer (sacar la última entrada) quita el lote entero; aceptar pide una segmentación por persona y cambia a Pincel; "Back to choosing" saca el lote.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: Personas agrega siluetas al borrado en un solo paso`

### Task 6: Paso 3, Revisar y Rehacer

**Files:**
- Create: `frontend/src/modules/editor/ReviewPanel.tsx`, `frontend/src/modules/editor/zones.ts`
- Modify: `frontend/src/lib/apiTypes.ts` (`GenerationJob` :730-763), `EditorPanel.tsx`, i18n
- Test: `frontend/src/modules/editor/zones.test.ts`, `ReviewPanel.test.tsx`

**Interfaces:**
- Consumes: `GenerationJob.eraseZones` (Plan 2, Task 7); `eraseEngine` (Task 2).
- Produces:
  - `type EraseZone = { index: number; box: [number, number, number, number] | null; engine: string; guard: "face" | "person" | null; residualPerson: boolean }` en `GenerationJob.eraseZones?: EraseZone[] | null`.
  - `zones.ts` (puro): `engineLabel(engine): string` (`lama` → "LaMa", `osor` → "OSOR", `lama+osor` → "LaMa + OSOR", `hibrido` → "Híbrido", `sdxl` → "SDXL"); `zoneStatus(zone): "clean" | "retried" | "residual"`; `zonePeople(zone, people, choices): string[]` (nombres de las personas a quitar cuya caja cae ≥ 50 % en la zona); `clipMaskToBox(mask: ImageData, box): ImageData` (fuera de la caja → alfa 0).
  - El Editor guarda la máscara de borrado exportada en cada envío (`lastEraseMask`) para Rehacer.
  - `ReviewPanel({ zones, people, choices, hoveredZone, onHoverZone, onRedo, sdxlAvailable })` — fila por zona: número, quiénes, motor, estado con ícono (check / alerta) y menú Rehacer (LaMa, OSOR, LaMa + OSOR, "Describir el fondo…" deshabilitado sin SDXL con motivo). Números sobre el resultado; con una zona en hover, el escenario muestra el recorte antes/después.
  - `redoZone(zone, engine, prompt?)`: sube la base actual, sube `clipMaskToBox(lastEraseMask, zone.box)`, envía el trabajo con `eraseEngine` (y `prompt` si es SDXL); al terminar, el resultado reemplaza la vista y la fila se actualiza.

- [ ] **Step 1: Write the failing tests** — funciones puras de `zones.ts`; el panel muestra "Se vio una persona" para `guard` con `residualPerson: false` y "Todavía se ve alguien" con `residualPerson: true`; Rehacer con LaMa envía `eraseEngine: "lama"` y una máscara recortada a la caja; "Describe the background…" deshabilitado sin SDXL.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: revisar zonas del borrado y rehacer una zona`

### Task 7: Imagen base con versiones, Restaurar y Clonar

**Files:**
- Create: `frontend/src/modules/editor/retouch.ts`, `frontend/src/modules/editor/useBaseVersions.ts`
- Modify: `EditorPanel.tsx` (`handleBaseSelected` :251-286, `handleUseResultAsBase` :498-509), `EditorToolRail.tsx`, i18n
- Test: `frontend/src/modules/editor/retouch.test.ts`, `useBaseVersions.test.ts`

**Interfaces:**
- Produces:
  - `retouch.ts` (puro, sobre `ImageData`): `restoreWithMask(current: ImageData, original: ImageData, mask: ImageData): ImageData` (mezcla por el alfa de la máscara; alfa 255 → píxel original exacto); `cloneWithMask(current: ImageData, mask: ImageData, dx: number, dy: number): ImageData` (copia de `(x+dx, y+dy)` pesado por el alfa; fuera de la imagen no cambia).
  - `useBaseVersions()` → `{ original: Blob | null, current: InitImageResponse | null, push(blob: Blob): Promise<void>, undo(): boolean, reset(file: File): Promise<void> }` — `push` sube con `uploadGenerationInitImage` y guarda la versión anterior; tope de 10 versiones; `original` es la foto abierta en la sesión (no cambia con "Seguir editando este resultado").
  - Deshacer unificado: un registro de acciones `("mask" | "base")[]`; Ctrl+Z deshace la última, sea entrada de máscara o versión de base.
  - Herramientas `restore` y `clone` en el grupo "Retocar": pintan con el pincel suave (tamaño compartido con Pincel); al soltar se aplica y se sube la base nueva. Clonar: Alt+clic fija el origen; sin origen, aviso "Alt+clic para elegir de dónde copiar"; una mira muestra el origen mientras se pinta.

- [ ] **Step 1: Write the failing tests**

```ts
it("restoreWithMask returns the original pixel where the mask is opaque", () => {
  const out = restoreWithMask(solid(4, 4, [10, 10, 10]), solid(4, 4, [200, 0, 0]), maskAt(4, 4, [[1, 1]]));
  expect(pixel(out, 1, 1)).toEqual([200, 0, 0, 255]);
  expect(pixel(out, 0, 0)).toEqual([10, 10, 10, 255]);
});
it("cloneWithMask copies from the offset source", () => { /* dx=2: el píxel (0,0) toma el de (2,0) */ });
it("undo after a base edit restores the previous base before older mask entries", () => { /* registro mask, base → Ctrl+Z vuelve la base */ });
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: Restaurar y Clonar sobre la imagen base con deshacer`

### Task 8: Corrector puntual

**Files:**
- Modify: `frontend/src/services/editor.ts`, `EditorPanel.tsx`, `EditorToolRail.tsx`, i18n
- Test: `EditorPanel.test.tsx`

**Interfaces:**
- Consumes: `POST /api/v1/editor/heal` (Plan 2, Task 8); `useBaseVersions.push` (Task 7).
- Produces: `healEditorImage(imageToken: string, maskToken: string): Promise<InitImageResponse>`; herramienta `heal`: al soltar el trazo exporta su máscara, la sube, llama `healEditorImage` y la imagen nueva entra como versión de base (sin pasar por la cola de trabajos). Si la caja del trazo tiene lado > 700 px de la foto, no llama y muestra "La zona es grande: usá Quitar."

- [ ] **Step 1: Write the failing tests** — trazo chico llama `healEditorImage` y cambia la imagen; trazo grande muestra el aviso y no llama; un error del endpoint se muestra sin perder la base.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: corrector puntual en el Editor`

### Task 9: Ajustar borde

**Files:**
- Modify: `EditorPanel.tsx` (panel Acción), `frontend/src/modules/editor/editorMaskCanvas.ts`, i18n
- Test: `editorMaskCanvas.test.ts`, `EditorPanel.test.tsx`

**Interfaces:**
- Produces: deslizadores "Expandir / Contraer" (−50..50 px de la foto, paso 1, 0 por defecto) y "Suavizar" (0..30 px) debajo del motor; vista previa sobre el overlay rojo con `previewAdjust(canvas: HTMLCanvasElement, growPx: number, featherPx: number, scale: number): void` (aproximada: desenfoque + umbral en un canvas de trabajo; el backend hace la morfología exacta); los valores viajan como `maskGrowPx`/`maskFeatherPx` y vuelven a 0 al cargar otra imagen.

- [ ] **Step 1: Write the failing tests** — mover "Expand" a 12 hace que el trabajo lleve `maskGrowPx: 12`; los valores vuelven a 0 con otra imagen; `previewAdjust` con 0/0 no toca el canvas.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: ajustar el borde de la máscara antes de quitar`

### Task 10: Verificación de punta a punta (criterios 5 y 6)

**Files:**
- Create (fuera del repo): `C:/personal/upflow-eval/2026-10-30-personas/e2e/` con capturas y `RESULTADOS.md`

- [ ] **Step 1: Levantar** un servidor de desarrollo del worktree en un puerto libre (ej. 8092) con `RUNTIME_DIR` propio, `MODELS_DIR=C:/Users/santi/AppData/Local/Upflow/runtime/models`, `PERSON_DETECTOR_MODEL=C:/personal/upflow-eval/models/yolox_s.onnx` y el frontend construido o `npm run dev` apuntando a ese puerto. No usar :8090 si hay otro proceso ahí; matar procesos por puerto, nunca por línea de comando.
- [ ] **Step 2: Criterio 5 con Playwright (MCP)** — abrir `_DSC2145.jpg`, esperar la lista, aceptar la propuesta, Quitar; verificar que Revisar lista zonas con su motor; Rehacer una zona con LaMa y comprobar (diferencia de píxeles) que solo cambió dentro de su caja.
- [ ] **Step 3: Criterio 6** — Corrector puntual sobre el brillo de `_DSC2081.jpg` (< 3 s medidos); Restaurar: diferencia 0 con el original en el centro del trazo; Clonar: el desplazamiento elegido; Ctrl+Z deshace cada aplicación; Ajustar borde +N: la máscara que llega al backend crece N px (leer la máscara subida).
- [ ] **Step 4: Anotar** resultados y capturas en `RESULTADOS.md`; cualquier fallo vuelve como tarea de corrección a la tarea que posee el código.
