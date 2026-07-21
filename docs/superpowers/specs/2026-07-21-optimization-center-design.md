# Optimization Center: detección de capacidades OS/driver + diagnóstico ONNX — Design

**Fecha:** 2026-07-21
**Estado:** Approved (pendiente de plan de implementación)

## Motivación

Investigación previa (2026-07-20d, `docs/superpowers/specs/2026-07-20-gmfss-production-performance-design.md`) identificó palancas de bajo nivel sin explotar: fallback silencioso CPU-EP en ONNX Runtime, HAGS, PCIe link negociado, política de write-cache del disco, exclusión de Windows Defender sobre `runtime/`, Resizable BAR/Above4G (BIOS). Ninguna se ha medido ni expuesto al usuario. Principio de producto explícito del usuario: "software inteligente" — cada optimización se detecta, se activa si el hardware/SO la soporta, si no cae a la mejor estrategia disponible sin ella, nunca en silencio. Mismo patrón que ya usan `backend_registry.py` (runtime ncnn/onnx) y `video_encoders.py` (encoder software/hw): función pura de detección + fallback seguro.

## Alcance

Tres piezas independientes, en fases:

- **Fase 0** — código, sin dependencia de hardware específico: diagnóstico de fallback CPU-EP (0.1) + auditoría de IOBinding en los 4 motores ONNX (0.2).
- **Fase 1** — `CapabilityProbe` + panel "Optimization Center" en Settings: HAGS, PCIe link, write-cache del disco, exclusión Defender. Toggles admin vía UAC in-app.
- **Fase 2** — checklist guiado de Resizable BAR/Above4G (BIOS, no detectable por software).

**Fuera de alcance de este spec**: Fase 3 (dual-pipeline iGPU+dGPU) — arquitectura mayor, se evalúa solo si Fases 0-2 no cierran el gap medido.

**Prerrequisito de repo (no es parte del código de este spec)**: mergear a master las dos ramas ya terminadas y revisadas (`feature/gmfss-session-coordinator`, `feature/audio-tracks-subs-quality`) antes de ramar `feature/optimization-center`, para no acumular una tercera rama divergente sobre trabajo ya aprobado.

## Fase 0.1 — Diagnóstico de fallback CPU-EP

### Por qué no usar `disable_cpu_ep_fallback`

Esa opción de sesión hace que la CREACIÓN de la sesión falle (excepción) si algún nodo no tiene kernel en el EP objetivo y no puede caer a CPU — aborta, no enumera. Sirve para bloquear el fallback en producción, no para diagnosticarlo.

### Mecanismo: profiling de ONNX Runtime (API oficial, no parseo de texto libre)

```python
so = onnxruntime.SessionOptions()
so.enable_profiling = True
session = onnxruntime.InferenceSession(model_path, so, providers=[device_ep, "CPUExecutionProvider"])
synthetic_input = _zeros_for(session.get_inputs())
session.run(None, synthetic_input)
profile_path = session.end_profiling()
```

`profile_path` es un JSON de eventos; los eventos de categoría `"Node"` traen `args.op_name` y `args.provider` por nodo ejecutado. Filtrar `provider != device_ep` da la lista completa de ops que cayeron a CPU — enumeración real, no "falla en el primer nodo y para".

### Contrato del servicio

Nuevo módulo `app/services/onnx_cpu_fallback_probe.py` (o extensión pequeña de `capability_probe.py` — decidir en plan según tamaño):

- `probe_cpu_fallback(model_path: str, device_ep: str, input_shapes: dict) -> CpuFallbackReport` — función con efectos (crea sesión, corre, lee archivo), pero el parseo del JSON de profiling es una función pura separada y testeable con fixtures de JSON grabadas (no requiere GPU ni ONNX Runtime real en el test).
- `CpuFallbackReport = {model_id, device_id, hot_ops: list[str], clean: bool}`.
- Corre manual desde el panel de diagnóstico (nunca en el hot path de un job real), un modelo+device a la vez. Cacheado en memoria por `(model_id, device_id)`; "Re-scan" lo invalida.
- Cobertura: los 5 motores ONNX (`OnnxUpscaler`, `OnnxVideoUpscaler`, `ApolloRestorer`, `AudioSrRestorer`, `GmfssEngine` — sus 4 grafos) expuestos como catálogo de `(model_id, device_id)` probables a elegir desde la UI.

## Fase 0.2 — Auditoría de IOBinding

`OnnxVideoUpscaler` ya usa IOBinding (SP11). Auditar `ApolloRestorer`, `AudioSrRestorer`, `GmfssEngine`: confirmar que cada `session.run()` en el loop caliente usa `IOBinding` (evita copia CPU↔GPU implícita por llamada) o retrofit si falta, siguiendo el patrón existente. Trabajo de código real (no solo lectura); alcance final depende de lo que la auditoría encuentre — el plan detalla motor por motor.

## Fase 1 — `CapabilityProbe` + panel Optimization Center

### Módulo `app/services/capability_probe.py`

```python
class LeverStatus(str, Enum):
    ok = "ok"
    unavailable = "unavailable"
    not_applicable = "not_applicable"
    needs_admin = "needs_admin"

@dataclass(frozen=True, slots=True)
class Lever:
    id: str
    label: str
    status: LeverStatus
    detail: str
    fixable: bool
```

Cuatro probes, cada uno Windows-only con el mismo guard `sys.platform != "win32"` que ya usa `devices_service.py` (retorna `not_applicable` fuera de Windows — cero platform-branching nuevo, mismo patrón):

1. **HAGS** — lectura `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\HwSchMode` vía `winreg` estándar (sin admin). Expectativa de ganancia ~0 (research previo), se incluye igual por pedido explícito del usuario.
2. **PCIe link negociado** — `Win32_VideoController` (WMI) correlado por nombre con las GPUs no-software (mismo filtro que ya usa `devices_service._enumerate_gpu_adapter_names`) + `Get-PnpDeviceProperty DEVPKEY_PciDevice_Current{LinkSpeed,LinkWidth}` por su PNPDeviceID. Solo lectura — no hay `fixable` acá, pesca downgrades silenciosos (x8/Gen3) para diagnóstico, no para arreglar desde software.
3. **Write-cache del disco** — política vía WMI/registro (`EnableWriteCacheOnDisk` o el equivalente de "Better performance" vs "Quick removal"). Aplica directo al cuello save-bound ya diagnosticado a 720p.
4. **Exclusión Defender sobre `runtime/`** — lectura vía `Get-MpPreference` (si falla por permisos, estado `needs_admin` también para LEER, no solo para escribir).

### Elevación UAC in-app

Para HAGS-write, disk-policy-write y Defender-exclusion-add: endpoint `POST /api/v1/capabilities/{lever_id}/fix` dispara `Start-Process powershell -Verb RunAs -ArgumentList <script de una sola acción>` desde el backend. El script hace UNA cosa puntual y termina — nada interactivo. UI muestra "esperando confirmación UAC…" y hace poll a un endpoint de resultado hasta que el proceso elevado termina (o el usuario cancela el prompt de Windows, en cuyo caso el estado vuelve a `needs_admin` sin cambios).

### API

Router nuevo `app/api/capability_routes.py` (no crecer más `routes.py`, ya cerca del límite de 800 líneas por follow-up ya trackeado):
- `GET /api/v1/capabilities` → lista de `Lever` (probes read-only, cacheados).
- `POST /api/v1/capabilities/rescan` → invalida cache, vuelve a probar todo.
- `POST /api/v1/capabilities/{lever_id}/fix` → dispara la elevación UAC para levers `fixable`.
- `GET /api/v1/capabilities/onnx-diagnostics` → catálogo de `(model_id, device_id)` diagnosticables + resultados cacheados de Fase 0.1.
- `POST /api/v1/capabilities/onnx-diagnostics/{model_id}/{device_id}/scan` → corre `probe_cpu_fallback` para ese par.

### UI

Nueva sección "Optimization Center" dentro de `SettingsPage` (componente propio en `modules/settings/`, `SettingsPage.tsx` solo importa y compone — mismo patrón que ya usa con `DeviceDefault`). Una fila por lever: label + badge ✅/❌/⚠️(not_applicable)/🔒(needs_admin) + botón "Fix" cuando `fixable`. Sub-panel "Diagnostics" con tabla motor×device → ops en CPU (Fase 0.1) + estado IOBinding por motor (Fase 0.2, estático — no requiere API, es resultado de auditoría de código documentado).

## Fase 2 — Checklist BIOS (Resizable BAR / Above4G)

No detectable de forma confiable por software (vive en firmware). Componente puramente informativo: pasos guiados + campo de auto-reporte del usuario (persistido, pero **nunca gatea lógica** — es solo para que el panel recuerde lo que el usuario ya confirmó) + instrucciones de benchmark A/B documentadas (no se construye herramienta de benchmark nueva).

## Verificación de impacto (obligatoria)

Cada lever `fixable` se banca con benchmark real antes/después, mismo rigor que la auditoría de 61 agentes previa. Se documenta el número real — incluyendo si el resultado es "sin diferencia medible en este hardware" (ej. HAGS, esperado). Ninguna ganancia se afirma en README/UI sin medición.

## Testing

- **0.1**: parseo del JSON de profiling es función pura, testeada con fixtures de JSON grabadas (sin GPU real). `probe_cpu_fallback` en sí (crea sesión, corre, lee archivo) testeado con un modelo ONNX trivial CPU-only en CI — sin depender de DirectML.
- **0.2**: por motor, test que confirma uso de `IOBinding` en el loop de inferencia (inspección de la llamada, no del resultado numérico — eso ya está cubierto por los tests de parity existentes de cada motor).
- **1**: cada probe es función pura sobre su input crudo (salida de `winreg`/WMI ya parseada) — testeable sin Windows real, mockeando la fuente. Fallback a `not_applicable` en `sys.platform != "win32"` testeado con monkeypatch, mismo patrón que ya existe para `devices_service`.
- **Elevación**: test de que el script generado para cada fix es el esperado (comparación de texto/argumentos), sin ejecutar `Start-Process` real en CI.

## Fuera de alcance

- Fase 3 (dual-pipeline iGPU+dGPU) — evaluar después de medir 0-2.
- Herramienta de benchmark A/B automatizada para Resizable BAR — instrucciones documentadas, no código.
- Arreglar la causa raíz de contención DirectML entre sesiones — ya cubierto por el spec de GpuSessionCoordinator (2026-07-20), no se duplica acá.
- EP CUDA/TensorRT nativo para NVIDIA — sin relación con este spec.
