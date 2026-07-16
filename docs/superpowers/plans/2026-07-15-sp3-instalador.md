# SP3 — Instalador Inno Setup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** `setup.exe` amigable (Inno Setup) que instala Upflow en Windows sin admin, con Python embebido, para un usuario no técnico (amigo con NVIDIA RTX).

**Architecture:** Instala en `{localappdata}\Upflow`: app (backend + frontend/dist pre-buildeado) + Python embebido (3.12 x64) con pip habilitado. Accesos directos. Primer arranque: launcher adaptado usa el Python embebido para instalar deps pip (~2-3GB con torch) + descargar binarios vendored (~1GB) con progreso amigable. Desinstalador preserva `runtime/` y modelos (opción de borrarlos).

**Tech Stack:** Inno Setup 6 (ISCC), Python 3.12 embeddable + get-pip, PowerShell (launcher).

**Spec:** docs/superpowers/specs/2026-07-14-upflow-v2-ux-design.md (§SP3).

## Global Constraints
- Sin admin (install a localappdata, `PrivilegesRequired=lowest`).
- El instalador NO trae los ~1GB de binarios vendored ni los ~2-3GB de wheels — esos se obtienen en primer arranque (igual que hoy). El setup.exe queda liviano (~30-50MB: app + frontend/dist + Python embebido).
- Python embebido: el usuario NO necesita Python del sistema. El launcher usa el Python embebido bundleado.
- Mensajes al usuario en español, claros, sin stack traces.
- Desinstalador preserva datos del usuario (runtime/, modelos) por defecto; checkbox opcional para borrarlos.
- pytest sigue verde (los cambios son scripting/infra, no tocan app/ salvo quizá resolución de rutas). Commits español convencional; sin Co-Authored-By. Rama `feature/sp3-instalador`.

---

### Task 1: Python embebido + launcher adaptado + installer .iss + package-release -Installer

**Files:** Create `installer/upflow.iss`, `installer/README.md` (cómo compilar). Modify `scripts/upflow-launcher.ps1` (soportar Python embebido/bundleado, no solo del sistema), `scripts/package-release.ps1` (modo `-Installer`: descarga Python embeddable + get-pip, arma el árbol de instalación, compila con ISCC → `dist/upflow-setup-v<version>.exe`), `README.md` (sección instalación con instalador).

**Diseño Python embebido (binding):**
- El instalador incluye el Python 3.12 embeddable (`python-3.12.x-embed-amd64.zip` descomprimido) + `get-pip.py` ejecutado para habilitar pip, con `python312._pth` editado para permitir site-packages e imports. package-release.ps1 -Installer prepara este árbol antes de compilar.
- El launcher detecta el Python embebido en `{app}\python\python.exe` (bundleado por el instalador) y lo usa con prioridad; si no existe, cae al Python del sistema (modo portable/zip actual). Un `pip install -e .` o `pip install -r` contra el embebido en primer arranque instala las deps; sentinel marca "instalado".
- Primer arranque: launcher (via embebido) → pip install deps (progreso) → download-*.ps1 binarios → .env con ENABLE_INTERPOLATION=true + ENABLE_AUDIO_ENHANCE=true → arranca uvicorn → abre navegador.

**upflow.iss (binding):**
- `[Setup]`: AppName=Upflow, AppVersion leído (o pasado por -DMyAppVersion desde package-release), DefaultDirName={localappdata}\Upflow, PrivilegesRequired=lowest, no admin, DisableProgramGroupPage, OutputBaseFilename=upflow-setup-v<version>, SetupIconFile si hay icono, WizardStyle=modern.
- `[Files]`: la app (app/, frontend/dist/, scripts/, pyproject.toml, .env.example, README.md, LICENSE, Upflow.bat) + el Python embebido preparado (installer\build\python\*). Excluye tests, docs internos, .git, node_modules, .venv, runtime, vendor.
- `[Icons]`: acceso directo escritorio + menú inicio → Upflow.bat (o el launcher) con icono.
- `[Run]`: opción "iniciar Upflow al terminar" (primer arranque hace la instalación pesada — advertir en un mensaje que la primera vez descarga ~3-4GB y tarda).
- `[UninstallDelete]` / código Pascal: preservar `{localappdata}\Upflow\runtime` y modelos; página opcional (Tasks) "borrar también modelos y datos".
- Mensajes personalizados en español donde aplique.

**package-release.ps1 -Installer (binding):**
- Verifica ISCC presente (winget `JRSoftware.InnoSetup`); si falta, mensaje claro para instalarlo.
- `npm ci && npm run build` (frontend/dist).
- Descarga Python embeddable + get-pip a `installer/build/python/`, prepara pip + _pth.
- Arma `installer/build/app/` con el árbol de la app (allowlist, sin vendored/runtime).
- Compila `ISCC upflow.iss` → `dist/upflow-setup-v<version>.exe`.
- Mantiene el zip portable actual como alternativa (`-Zip` o ambos).

**Verificación (no requiere instalar en máquina limpia):**
- `ISCC` compila el .iss sin errores → produce setup.exe (instala ISCC via winget para el check).
- Lógica del launcher para Python embebido: test manual de la rama de detección (embebido presente → lo usa; ausente → sistema).
- pytest verde (si se tocó resolución de rutas en app/).

### Task 2: Smoke compile + docs + FINAL (README + GitHub release)

- **Compile smoke**: instalar ISCC (winget JRSoftware.InnoSetup), correr `package-release.ps1 -Installer`, verificar que produce `dist/upflow-setup-v<version>.exe` sin errores de compilación; inspeccionar (7z/expand) que el setup contiene la app + frontend/dist + Python embebido, sin tests/vendored/runtime. Documentar tamaño y contenido. (Instalación real en máquina limpia = manual del usuario; documentar pasos.)
- **README**: sección "Instalación (usuarios)" con el instalador como opción primaria (descargar setup.exe → siguiente-siguiente → primer arranque descarga binarios → navegador abre), zip portable como alternativa; requisitos (Windows 10/11, GPU Vulkan NVIDIA/AMD/Intel, SIN necesidad de Python del sistema con el instalador). Nota NVIDIA RTX funciona vía Vulkan/DirectML.
- **FINAL**: merge SP3 a master; regenerar release (zip + setup.exe); crear GitHub Release en santiquiroz/upflow con el setup.exe + zip adjuntos y notas (features: upscaling imagen/video, FPS boost + target 60fps, audio IA, modelos HF + selección de dispositivo, UI React). Usar `gh release create` (limpiar GITHUB_TOKEN para cuenta personal).

## Self-Review
- Cobertura §SP3: instalador Inno sin admin ✓, Python embebido ✓, shortcuts ✓, desinstalador preserva datos ✓, package-release -Installer ✓, README ✓, GitHub release ✓ (T2 FINAL).
