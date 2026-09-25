$ErrorActionPreference = 'Stop'
# La barra de progreso de Invoke-WebRequest en 5.1 hace la descarga varias veces mas lenta.
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# MI-GAN (Picsart) para el borrado rapido del Editor y el relleno "Fast" de
# Restore photo: rellena el hueco continuando el entorno, sin prompts ni pasos de
# difusion. La revision 406830d0 (2026-09-14) es la primera que trae el LICENSE
# en HF; el ONNX es el mismo de 1538c135. Tamano y sha256 del ONNX sacados de la
# API de HF (lfs.oid) y de las licencias bajandolas, el 2026-09-25.
$revision = '406830d0fa60666da0071c342ad2fbc8f30c5c64'
$weightsCommit = '680b0c6149c51e272412f0e4686c0d6668dba15c'

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\migan'

# Las licencias van primero: la app da el pack por instalado apenas existe el
# ONNX, asi que el ONNX es lo ultimo que aparece.
$archivos = @(
    @{
        File = 'LICENSE'
        Url = "https://huggingface.co/andraniksargsyan/migan/resolve/$revision/LICENSE"
        Sha256 = '13348eaaebc0b07b0faf3a92db04e0858737713a917d064b599c87cac05b571f'
        Size = 1082
    },
    @{
        File = 'LICENSE-WEIGHTS'
        Url = "https://raw.githubusercontent.com/Picsart-AI-Research/MI-GAN/$weightsCommit/LICENSE-WEIGHTS"
        Sha256 = '674e33456b8d03693b84ab305aa7372d93c91e68b234651b0adad106d518e6eb'
        Size = 1083
    },
    @{
        File = 'migan_pipeline_v2.onnx'
        Url = "https://huggingface.co/andraniksargsyan/migan/resolve/$revision/migan_pipeline_v2.onnx"
        Sha256 = '6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b'
        Size = 28079181
    }
)

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

function Receive-VerifiedFile([string]$destination, $entry) {
    $temporal = "$destination.download"
    Write-Host "Downloading $($entry.Url)"
    try {
        Invoke-WebRequest -Uri $entry.Url -OutFile $temporal -UseBasicParsing
        $fallo = Get-IntegrityError $temporal $entry
        if ($null -ne $fallo) {
            throw "The download of $($entry.File) did not verify: $fallo."
        }
        Move-Item -Force -LiteralPath $temporal -Destination $destination
    } catch {
        throw "MI-GAN: could not install $($entry.File): $($_.Exception.Message)"
    } finally {
        Remove-Item -Force -LiteralPath $temporal -ErrorAction SilentlyContinue
    }
}

function Install-VerifiedFile($entry) {
    $destination = Join-Path $vendorDir $entry.File
    if (Test-VerifiedFile $destination $entry) {
        Write-Host "Already here: $($entry.File)"
        return
    }
    Receive-VerifiedFile $destination $entry
}

New-Item -ItemType Directory -Force $vendorDir | Out-Null
foreach ($entry in $archivos) {
    Install-VerifiedFile $entry
}

Write-Host "MI-GAN ready in $vendorDir (MIT: LICENSE and LICENSE-WEIGHTS)"
