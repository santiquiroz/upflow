; Instalador de Upflow (Inno Setup 6).
; Se compila con scripts/package-release.ps1 -Installer (ISCC.exe via winget
; JRSoftware.InnoSetup). Ver installer/README.md para compilar manualmente.
;
; MyAppVersion se pasa desde package-release.ps1 con /DMyAppVersion=X.Y.Z;
; el default de abajo solo existe para poder compilar este .iss suelto
; (ISCC installer\upflow.iss) sin pasar esa variable, por ejemplo para
; validar la sintaxis con un arbol installer\build\ de prueba.
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

#define MyAppName "Upflow"
#define MyAppPublisher "Upflow"
#define MyAppURL "https://github.com/santiquiroz/upflow"

[Setup]
AppId={{2D97F739-E5F5-4495-BC53-C9702106B52C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Upflow
DefaultGroupName=Upflow
DisableProgramGroupPage=yes
; Instala en el perfil del usuario actual (localappdata): nunca pide admin.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=upflow-setup-v{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\Upflow.bat
SetupLogging=yes
; No hay SetupIconFile todavia: el repo no trae un .ico dedicado para el
; instalador. Agregar aca cuando exista (ver installer/README.md).

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Messages]
FinishedLabel=Upflow se instalo correctamente en tu computadora.%n%nIMPORTANTE: la primera vez que lo abras va a descargar ~3-4 GB (motor de upscaling, FFmpeg, RIFE y las dependencias de Python) y puede tardar varios minutos segun tu conexion a internet. Las siguientes veces arranca al instante.%n%nPodes iniciarlo ahora tildando la opcion de abajo, o mas tarde desde el acceso directo.

; Las funciones que el usuario puede elegir, descritas por lo que HABILITAN y no
; por el nombre del binario: nadie que instale por primera vez sabe que es "RIFE".
;
; Los paquetes NO viajan en el instalador -- los baja el launcher en el primer
; arranque -- asi que estos componentes se usan como pantalla de eleccion: el
; codigo de [Code] persiste lo tildado en optional-packs.txt y el launcher lo lee.
;
; Saltear algo no deja a nadie encerrado: la pantalla de Tasks de la app muestra
; cada capacidad a la que le falta su paquete, con un boton que corre el mismo
; script de descarga.
[Types]
Name: "full"; Description: "Completa (recomendada)"
Name: "minimal"; Description: "Minima: solo imagen y video"
Name: "custom"; Description: "Personalizada"; Flags: iscustom

[Components]
; `fixed` porque sin estos dos la app no hace nada: Real-ESRGAN es el motor de
; upscaling y ffmpeg lo usa todo, incluida la cadena de mejora de voz.
Name: "core"; Description: "Reescalar imagenes y video (motor + FFmpeg, ~1 GB)"; Types: full minimal custom; Flags: fixed
Name: "frames"; Description: "Generar fotogramas: duplicar o triplicar los FPS de un video (~45 MB)"; Types: full custom
Name: "denoise"; Description: "Quitar ruido de audio con un modelo de IA, mas fuerte que el filtro rapido (~20 MB)"; Types: full custom
Name: "restore"; Description: "Restaurar los agudos que perdio un MP3 o un AAC (~90 MB)"; Types: full custom
Name: "modelanime"; Description: "Generar imagenes: modelo anime (Hassaku XL Illustrious, descarga ~6.5 GB y se prepara solo la primera vez, ~20 min)"
Name: "modelphoto"; Description: "Generar imagenes: modelo fotorrealista (epiCRealism XL Last FAME, descarga ~6.5 GB y se prepara solo la primera vez, ~20 min)"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "deleteuserdata"; Description: "Al desinstalar, borrar tambien los datos y descargas (~4 GB): archivos subidos, resultados y modelos de Hugging Face (runtime\), binarios de Real-ESRGAN/FFmpeg/RIFE/DeepFilterNet (vendor\) y las dependencias de Python (torch, etc.) descargadas en el primer arranque - accion irreversible"; GroupDescription: "Datos de usuario al desinstalar:"; Flags: unchecked

[Files]
; El arbol de la app (allowlist: app/, scripts/, frontend/dist/, pyproject.toml,
; .env.example, README.md, LICENSE, Upflow.bat) ya viene armado por
; package-release.ps1 -Installer en installer\build\app\.
Source: "build\app\*"; DestDir: "{app}"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
; Python 3.12 embeddable + pip ya preparado por package-release.ps1 -Installer
; en installer\build\python\ (ver Initialize-EmbeddedPython).
Source: "build\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Upflow"; Filename: "{app}\Upflow.bat"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,Upflow}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Upflow"; Filename: "{app}\Upflow.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Upflow.bat"; Description: "{cm:LaunchProgram,Upflow}"; Flags: postinstall skipifsilent nowait

[Code]
const
  DeleteUserDataMarker = '.delete-user-data-on-uninstall';
  OptionalPacksFile = 'optional-packs.txt';

{ Persiste lo que el usuario tildo, para que el launcher lo lea en el primer
  arranque en vez de preguntarlo por consola.

  Se escribe SIEMPRE, incluso vacio: el archivo ausente significa "no hubo
  instalador" (zip portable, o una instalacion anterior a esta version) y ahi el
  launcher instala todo, que es el comportamiento historico. Un archivo vacio, en
  cambio, significa "el usuario eligio nada", y son cosas distintas. }
procedure SaveOptionalPacks;
var
  Selected: String;
  PackPath: String;
begin
  Selected := '';
  if WizardIsComponentSelected('frames') then
    Selected := Selected + 'rife' + #13#10;
  if WizardIsComponentSelected('denoise') then
    Selected := Selected + 'deepfilternet' + #13#10;
  if WizardIsComponentSelected('restore') then
    Selected := Selected + 'apollo' + #13#10;
  if WizardIsComponentSelected('modelanime') then
    Selected := Selected + 'model-anime' + #13#10;
  if WizardIsComponentSelected('modelphoto') then
    Selected := Selected + 'model-photo' + #13#10;

  PackPath := ExpandConstant('{app}\' + OptionalPacksFile);
  SaveStringToFile(PackPath, Selected, False);
end;

{ WizardIsTaskSelected lleva el flag sfNoUninstall (Setup-only): llamarla en
  CurUninstallStepChanged lanza un InternalError en runtime ("Cannot call
  WizardIsTaskSelected function during Uninstall") que ISCC NO detecta al
  compilar. Por eso persistimos la decision del task en un marker file durante
  la INSTALACION (aca WizardIsTaskSelected si es valida, es codigo de Setup) y
  la leemos en el uninstall con FileExists, sin volver a llamar la funcion. El
  marker es untracked (lo escribe este codigo, no la seccion [Files]), asi que
  Inno no lo borra solo y sigue disponible durante la desinstalacion. }
procedure CurStepChanged(CurStep: TSetupStep);
var
  MarkerPath: String;
begin
  { Idempotente: escribe el marker si el task esta tildado, lo BORRA si no.
    Sin el else, un marker de un install previo CON checkbox sobrevive a un
    reinstall/upgrade (misma carpeta de instalacion) SIN checkbox, y un
    uninstall posterior borraria los ~4 GB del usuario aunque en su ultimo
    install NO lo pidio - perdida de datos irreversible. }
  if CurStep = ssPostInstall then
  begin
    MarkerPath := ExpandConstant('{app}\' + DeleteUserDataMarker);
    if WizardIsTaskSelected('deleteuserdata') then
      SaveStringToFile(MarkerPath, '1', False)
    else
      DeleteFile(MarkerPath);

    SaveOptionalPacks;
  end;
end;

procedure DeleteDownloadedUserData;
begin
  { Borra todo lo que el primer arranque genera/descarga y que Inno NO rastrea
    (no vino del instalador). Al tildar el checkbox el usuario pidio recuperar
    los ~4 GB:
      - runtime\ : uploads/outputs/temp/video-work + modelos HF (runtime\models
        por app/config.py). Datos del usuario.
      - vendor\ : binarios NCNN/FFmpeg/RIFE/DeepFilterNet (~1 GB, download-*.ps1).
      - python\Lib\site-packages : deps pip incl. torch (~2-3 GB, pip install -e .).
    Los tres se re-obtienen solos en el proximo primer arranque si se reinstala.
    Tambien se borra .env (config generada por el launcher, untracked): asi no
    queda huerfano bloqueando la eliminacion de la carpeta de instalacion. }
  DelTree(ExpandConstant('{app}\runtime'), True, True, True);
  DelTree(ExpandConstant('{app}\vendor'), True, True, True);
  DelTree(ExpandConstant('{app}\python\Lib\site-packages'), True, True, True);
  DeleteFile(ExpandConstant('{app}\.env'));
  { El sentinel del launcher (python\.upflow-installed) es untracked: sin este
    DeleteFile quedaria huerfano y python\ no se podria eliminar por completo. }
  DeleteFile(ExpandConstant('{app}\python\.upflow-installed'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  MarkerPath: String;
begin
  { usUninstall corre ANTES de que Inno borre los archivos rastreados, con todo
    el arbol todavia presente, para que DelTree limpie site-packages completo
    (bundleado + descargado) de una sola pasada. }
  if CurUninstallStep = usUninstall then
  begin
    { Lo escribe [Code] y no [Files], asi que Inno no lo rastrea: sin este
      DeleteFile queda huerfano y la carpeta de instalacion no se puede borrar
      completa. Va afuera del if del marker porque existe en TODA instalacion. }
    DeleteFile(ExpandConstant('{app}\' + OptionalPacksFile));

    MarkerPath := ExpandConstant('{app}\' + DeleteUserDataMarker);
    if FileExists(MarkerPath) then
    begin
      DeleteDownloadedUserData;
      DeleteFile(MarkerPath);
    end;
  end;
end;
