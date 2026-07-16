# Instalador de Upflow (Inno Setup)

`upflow.iss` genera `dist/upflow-setup-v<version>.exe`: un instalador liviano
(~30-50 MB) que trae la app + la SPA compilada + un Python 3.12 embeddable
con pip ya funcional. **No** trae los binarios vendored (Real-ESRGAN, FFmpeg,
RIFE, DeepFilterNet, ~1 GB) ni las dependencias pip pesadas (torch, onnx,
~2-3 GB) — esas se descargan en el primer arranque, igual que en el zip
portable (ver `scripts/upflow-launcher.ps1`).

## Compilar (forma normal)

Desde la raiz del repo:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package-release.ps1 -Installer
```

Esto hace todo el trabajo:

1. Verifica que `ISCC.exe` (compilador de Inno Setup) este disponible; si
   falta, muestra el comando de winget para instalarlo.
2. Compila la SPA de React (`npm ci && npm run build` en `frontend/`).
3. Descarga el Python 3.12 embeddable + `get-pip.py`, los prepara en
   `installer/build/python/` (edita `python312._pth` para habilitar
   `import site` y `Lib\site-packages`, corre `get-pip.py` para dejar pip
   funcional, e instala `setuptools`/`wheel` directo en el embebido — ver
   "Por que setuptools/wheel van pre-instalados" mas abajo).
4. Arma `installer/build/app/` con el mismo allowlist que usa el zip
   portable (`app/`, `scripts/`, `frontend/dist/`, `pyproject.toml`,
   `README.md`, `LICENSE`, `.env.example`, `Upflow.bat`).
5. Compila `upflow.iss` con `ISCC` pasando `/DMyAppVersion=<version de
   pyproject.toml>` → `dist/upflow-setup-v<version>.exe`.

Para generar **ambos** artefactos (zip portable + instalador) en una sola
corrida:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package-release.ps1 -Zip -Installer
```

Sin flags, el script mantiene el comportamiento historico: solo el zip
portable (`-Zip` implicito).

## Compilar manualmente el .iss (para validar sintaxis o iterar sobre el instalador)

`upflow.iss` espera un arbol ya armado en `installer/build/app/` y
`installer/build/python/` (eso es justamente lo que hacen los pasos 3-4 de
arriba). Si ya corriste `package-release.ps1 -Installer` al menos una vez,
podes recompilar solo el `.iss` sin repetir la descarga de Python ni el
build de frontend:

```powershell
ISCC /DMyAppVersion=0.2.0 installer\upflow.iss
```

Si `installer/build/` no existe todavia (por ejemplo, para chequear
puramente la sintaxis del `.iss` sin descargar nada), armar un arbol
stub minimo alcanza para que ISCC compile:

```powershell
New-Item -ItemType Directory -Force installer\build\app, installer\build\python
"@echo off" | Set-Content installer\build\app\Upflow.bat
"stub" | Set-Content installer\build\python\python.exe
ISCC /DMyAppVersion=0.0.0-stub installer\upflow.iss
```

`MyAppVersion` tiene un default (`0.0.0`) en el propio `.iss`, asi que
tambien se puede omitir el `/D` para una compilacion rapida de prueba.

## Instalar Inno Setup

```powershell
winget install JRSoftware.InnoSetup
```

`ISCC.exe` termina en distintas rutas segun como winget resuelva el
scope de instalacion (por usuario vs. por maquina) — el script busca en
`PATH`, `Inno Setup 6\ISCC.exe` bajo `Program Files`, `Program Files
(x86)` y `%LOCALAPPDATA%\Programs`, asi que no hace falta agregarlo al
PATH manualmente.

## Diseno del instalador (resumen)

- `DefaultDirName={localappdata}\Upflow`, `PrivilegesRequired=lowest`:
  instala en el perfil del usuario actual, nunca pide admin.
- Accesos directos en escritorio (tarea opcional, tildada por default) y
  menu inicio, apuntando a `Upflow.bat` con `WorkingDir={app}` (para que
  las rutas relativas de la app — `runtime/`, `vendor/`, `.env` — se
  resuelvan igual que en el zip portable).
- Al terminar, ofrece iniciar Upflow ya mismo; el mensaje de la pagina
  final (`[Messages] FinishedLabel`) advierte que el primer arranque baja
  ~3-4 GB y puede tardar varios minutos.
- El desinstalador **preserva `runtime/` (uploads, outputs y los modelos
  instalados desde Hugging Face, que viven en `runtime/models/`) por
  defecto**. Hay una tarea opcional sin tildar ("borrar tambien runtime\...")
  que, si se selecciona durante la instalacion, hace que el desinstalador
  borre ese directorio (`[Code] CurUninstallStepChanged` +
  `WizardIsTaskSelected`, que sigue funcionando durante el uninstall pese
  al nombre — ver el comentario en `upflow.iss`).
- `vendor/` (binarios NCNN/FFmpeg) y `python/Lib/site-packages`
  (dependencias pip instaladas en el primer arranque) quedan fuera del
  alcance de esa tarea: no los borra ni el uninstall por defecto ni la
  tarea opcional, porque Inno solo rastrea/borra lo que el instalador
  puso ahi, no lo que el launcher agrego despues. Un reset completo de
  disco requiere borrar `{localappdata}\Upflow` a mano.

## Por que setuptools/wheel van pre-instalados

`pip install -e .` normalmente usa build isolation: instala los
`[build-system] requires` de `pyproject.toml` (`setuptools`, `wheel`) en un
entorno aislado temporal e invoca el backend ahi adentro via un subproceso
con `PYTHONPATH` apuntando a ese entorno. El Python embeddable **ignora
`PYTHONPATH`** en cualquier subproceso propio — es el comportamiento
esperado de tener un archivo `._pth` (asi se logra que el deployment sea
autocontenido) — asi que esa inyeccion no hace nada y pip termina fallando
con `BackendUnavailable: Cannot import 'setuptools.build_meta'` al preparar
los metadatos del install editable. Instalar `setuptools`/`wheel`
directamente en el `site-packages` del embebido (paso 3 de arriba) y correr
`pip install --no-build-isolation -e .` en el launcher (ver
`upflow-launcher.ps1`) evita ese mecanismo roto por completo. Verificado a
mano contra un Python 3.12.10 embeddable real: sin este fix, incluso
`pip install -e .` (no solo `--dry-run`) falla en la etapa de metadata antes
de descargar ninguna dependencia.

## Version del Python embeddable

Pineada en `scripts/package-release.ps1` (`$pythonEmbedVersion`). Bump
manual si hace falta una version mas nueva — el build falla con un error
claro de descarga si la version pineada deja de estar disponible en
python.org.
