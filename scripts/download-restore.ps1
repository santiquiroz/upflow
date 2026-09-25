param(
    # Cual bundle de restauracion de fotos bajar. Tiene que coincidir con
    # BUNDLE_NAMES de app/services/restore_models.py (lo exige el anti-deriva de
    # tests/test_restore_models.py). Sin Mandatory a proposito: un parametro
    # obligatorio que falta hace que PowerShell pregunte por consola, y el
    # provisioner no tiene consola.
    [ValidateSet('core', 'faces', 'colorize')]
    [string]$Bundle
)

$ErrorActionPreference = 'Stop'
# La barra de progreso de Invoke-WebRequest en 5.1 hace la descarga varias veces mas lenta.
$ProgressPreference = 'SilentlyContinue'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\restore'
$licensesDir = Join-Path $vendorDir 'licenses'

# Los ONNX salen del Release de port-restore-onnx. Esta tabla es una copia
# EXACTA de RESTORE_BUNDLES + la licencia SPDX de cada RestoreModelSpec
# (app/services/restore_models.py); tests/test_restore_models.py la evalua con
# PowerShell y la compara campo por campo.
#
# Files:        @{ Model; Precision; File; Url; Sha256; Size (bytes); Label; License (SPDX) }
# LicenseFiles: @{ Model; File; Url; Sha256; Size (bytes) }
#               El Release aplana licenses/<modelo>/<archivo> como
#               licenses--<modelo>--<archivo>; aca se copia de vuelta a
#               vendor\restore\licenses\<Model>\<File>.
#
# Vacia hasta M-06/P1-15: los modelos todavia no estan publicados.
$catalogo = @{
    'core' = @{
        Files = @()
        LicenseFiles = @()
    }
    'faces' = @{
        Files = @()
        LicenseFiles = @()
    }
    'colorize' = @{
        Files = @()
        LicenseFiles = @()
    }
}

function Get-Sha256([string]$path) {
    # .NET directo y no Get-FileHash: con un PSModulePath heredado de pwsh 7 el
    # powershell 5.1 hijo no resuelve ese cmdlet (pasado real 2026-08-09).
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($path)
        try {
            $hash = $sha256.ComputeHash($stream)
        } finally {
            $stream.Dispose()
        }
    } finally {
        $sha256.Dispose()
    }
    return ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

function Get-IntegrityError([string]$path, $entry) {
    $size = (Get-Item -LiteralPath $path).Length
    if ($size -ne $entry.Size) {
        return "$size bytes, expected $($entry.Size)"
    }
    $sha = Get-Sha256 $path
    if ($sha -ne $entry.Sha256) {
        return "SHA-256 $sha, expected $($entry.Sha256)"
    }
    return $null
}

function Test-VerifiedFile([string]$destination, $entry) {
    if (-not (Test-Path -LiteralPath $destination)) {
        return $false
    }
    $fallo = Get-IntegrityError $destination $entry
    if ($null -eq $fallo) {
        return $true
    }
    Write-Host "The existing $($entry.File) does not verify ($fallo); downloading it again."
    Remove-Item -Force -LiteralPath $destination
    return $false
}

function Install-VerifiedFile([string]$destination, $entry) {
    if (Test-VerifiedFile $destination $entry) {
        Write-Host "Already here: $($entry.File)"
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    $temporal = "$destination.download"
    Write-Host "Downloading $($entry.File)..."
    try {
        Invoke-WebRequest -Uri $entry.Url -OutFile $temporal -UseBasicParsing
        $fallo = Get-IntegrityError $temporal $entry
        if ($null -ne $fallo) {
            throw "The download of $($entry.File) did not verify: $fallo."
        }
        Move-Item -Force -LiteralPath $temporal -Destination $destination
    } finally {
        Remove-Item -Force -LiteralPath $temporal -ErrorAction SilentlyContinue
    }
}

function Get-LicenseDestination($entry) {
    return Join-Path (Join-Path $licensesDir $entry.Model) $entry.File
}

function Get-FileRecord($entry) {
    return [ordered]@{
        model = $entry.Model
        precision = $entry.Precision
        file = $entry.File
        sha256 = $entry.Sha256
        size = $entry.Size
        license = $entry.License
    }
}

function Get-LicenseRecord($entry) {
    return [ordered]@{
        model = $entry.Model
        file = "licenses/$($entry.Model)/$($entry.File)"
        sha256 = $entry.Sha256
        size = $entry.Size
    }
}

function Write-InstalledManifest([string]$path, $tabla) {
    $manifest = [ordered]@{
        bundle = $Bundle
        pack = "restore-$Bundle"
        files = @($tabla.Files | ForEach-Object { Get-FileRecord $_ })
        licenses = @($tabla.LicenseFiles | ForEach-Object { Get-LicenseRecord $_ })
    }
    $json = ConvertTo-Json -InputObject $manifest -Depth 5
    $temporal = "$path.download"
    [System.IO.File]::WriteAllText($temporal, $json, (New-Object System.Text.UTF8Encoding $false))
    Move-Item -Force -LiteralPath $temporal -Destination $path
}

if (-not $Bundle) {
    throw "Pass -Bundle core, -Bundle faces or -Bundle colorize."
}

$tabla = $catalogo[$Bundle]
if ($tabla.Files.Count -eq 0) {
    throw "restore-${Bundle}: the models are not published yet."
}

# El manifiesto es lo que la app mira para dar el pack por instalado: se borra
# antes de tocar nada y solo vuelve cuando todo lo de abajo verifico.
$manifestPath = Join-Path $vendorDir "$Bundle.installed.json"
Remove-Item -Force -LiteralPath $manifestPath -ErrorAction SilentlyContinue

foreach ($entry in $tabla.Files) {
    Install-VerifiedFile (Join-Path $vendorDir $entry.File) $entry
}
foreach ($entry in $tabla.LicenseFiles) {
    Install-VerifiedFile (Get-LicenseDestination $entry) $entry
}

Write-InstalledManifest $manifestPath $tabla

Write-Host "restore-$Bundle ready in: $vendorDir"
Write-Host "Licenses and notices for each model: $licensesDir"
