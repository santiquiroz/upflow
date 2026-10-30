param(
    [string]$Destino
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Destino)) {
    $Destino = Join-Path $root 'vendor\yolox'
}
$vendorDir = $Destino
$archivos = @(
    @{
        File = 'yolox_s.onnx'
        Url = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx'
        Sha256 = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
        Size = 35858002
    }
)

function Get-Sha256([string]$path) {
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
        throw "YOLOX: could not install $($entry.File): $($_.Exception.Message)"
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

Write-Host "YOLOX ready in $vendorDir (Apache-2.0)"
