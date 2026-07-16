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

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "deleteuserdata"; Description: "Al desinstalar, borrar tambien runtime\ (archivos subidos, resultados y modelos instalados desde Hugging Face) - accion irreversible"; GroupDescription: "Datos de usuario al desinstalar:"; Flags: unchecked

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
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  RuntimeDir: String;
begin
  { runtime\ anida uploads/outputs/temp/video-work y los modelos instalados
    desde Hugging Face (app/config.py: models_path = runtime_path /
    models_dir = runtime\models por default) - borrar runtime\ alcanza para
    cubrir "datos y modelos". vendor\ (binarios NCNN/FFmpeg) y
    python\Lib\site-packages (dependencias pip) NO se tocan aca: son
    re-descargables/re-instalables en el proximo primer arranque, no datos
    del usuario, y no forman parte del alcance de este checkbox. }
  if CurUninstallStep = usPostUninstall then
  begin
    if WizardIsTaskSelected('deleteuserdata') then
    begin
      RuntimeDir := ExpandConstant('{app}\runtime');
      if DirExists(RuntimeDir) then
        DelTree(RuntimeDir, True, True, True);
    end;
  end;
end;
