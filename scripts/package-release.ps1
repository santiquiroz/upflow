$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$pyprojectPath = Join-Path $root 'pyproject.toml'

$versionMatch = Select-String -Path $pyprojectPath -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $versionMatch) {
    throw 'No se pudo leer la version desde pyproject.toml.'
}
$version = $versionMatch.Matches[0].Groups[1].Value

$distDir = Join-Path $root 'dist'
$stagingDir = Join-Path $distDir "upflow-v$version"
$zipPath = Join-Path $distDir "upflow-v$version.zip"

New-Item -ItemType Directory -Force -Path $distDir | Out-Null

if (Test-Path $stagingDir) {
    Remove-Item -Recurse -Force $stagingDir
}
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}
New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null

# Allowlist (not a denylist of vendor/runtime/.venv/.git/tests/docs/.superpowers):
# only end-user-facing files travel in the release zip. Binaries in vendor/
# are downloaded on first launch by upflow-launcher.ps1.
$includeDirs = @('app', 'scripts')
$includeFiles = @('pyproject.toml', 'README.md', 'LICENSE', '.env.example', 'Upflow.bat')

foreach ($dir in $includeDirs) {
    $source = Join-Path $root $dir
    if (-not (Test-Path $source)) {
        throw "Carpeta esperada no encontrada: $dir"
    }
    Copy-Item -Recurse -Force $source (Join-Path $stagingDir $dir)
}

foreach ($file in $includeFiles) {
    $source = Join-Path $root $file
    if (-not (Test-Path $source)) {
        throw "Archivo esperado no encontrado: $file"
    }
    Copy-Item -Force $source (Join-Path $stagingDir $file)
}

# Bytecode cache never belongs in a distributed zip.
Get-ChildItem $stagingDir -Recurse -Directory -Filter '__pycache__' | ForEach-Object {
    Remove-Item -Recurse -Force $_.FullName
}
Get-ChildItem $stagingDir -Recurse -File -Filter '*.pyc' | ForEach-Object {
    Remove-Item -Force $_.FullName
}

# Both Compress-Archive and ZipFile.CreateFromDirectory leave literal
# backslashes in entry names on Windows/.NET Framework (Windows PowerShell
# 5.1), which violates the ZIP spec (entries must use '/') and breaks
# extraction on tools that follow it strictly. Building entries manually
# guarantees forward-slash separators regardless of PowerShell/.NET version.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipStream = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::Create)
try {
    $archive = New-Object System.IO.Compression.ZipArchive($zipStream, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -Path $stagingDir -Recurse -File | ForEach-Object {
            $relativePath = $_.FullName.Substring($stagingDir.Length + 1) -replace '\\', '/'
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $_.FullName, $relativePath, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    }
    finally {
        $archive.Dispose()
    }
}
finally {
    $zipStream.Dispose()
}

Remove-Item -Recurse -Force $stagingDir

Write-Host 'Release generado:' $zipPath
Write-Host 'Version:' $version
