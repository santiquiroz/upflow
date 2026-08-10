param(
    # Cual modelo del catalogo bajar. Tiene que coincidir con los ids de
    # app/services/engines/separation_models.py.
    [ValidateSet('inst_hq_3', 'voc_ft', 'mel_band_roformer_kim', 'reverb_hq', 'deecho_normal', 'deecho_aggressive', 'deecho_dereverb', 'denoise')]
    [string]$Model = 'inst_hq_3'
)

$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\karaoke'

# UN solo pack porque el pack lo define el DESTINO (vendor\karaoke = la carpeta
# de modelos de separacion), no la fuente: el provisioner, la capability
# audio.karaoke, el picker de la UI y KARAOKE_MODEL_DIR estan todos cableados a
# esa carpeta. Partirlo en dos scripts obligaria a duplicar todo eso para
# ganar nada.
#
# Dos familias de modelos conviven aca, con verificacion distinta a proposito:
#
# * MDX-Net (inst_hq_3 / voc_ft / reverb_hq): .onnx distribuidos por el canal
#   oficial de descargas de Ultimate Vocal Remover, con credito por autor —
#   equipo core (Anjok07 & aufr33, MIT) o contribuidores como FoxJoy (Reverb
#   HQ). Traen "hash UVR" (MD5 de los ultimos 10000 KiB, que es como UVR
#   identifica sus modelos en model_data_new.json) y ADEMAS SHA-256 del archivo
#   completo, porque el hash UVR solo cubre la cola y el mirror
#   TRvlvr/model_repo es de terceros. Verificados contra los archivos reales el
#   2026-08-09.
#
# * VR De-Echo/De-Reverb/De-Noise (deecho_*, denoise): pesos de FoxJoy
#   distribuidos por el mismo canal oficial de UVR, pero exportados a ONNX por
#   un port propio y publico (github.com/santiquiroz/port-uvr-deecho-onnx, MIT,
#   release models-v1.1). No tienen hash UVR: ese hash identifica el .pth de
#   origen, no el .onnx re-exportado (queda anotado en el catalogo como
#   procedencia). Se verifica el SHA-256 del release, tomado de su
#   manifest.json y comprobado contra los archivos descargados el 2026-08-10.
#
# * Mel-Band RoFormer (mel_band_roformer_kim): pesos de KimberleyJSN, MIT
#   declarado en su model card de HuggingFace, exportados a ONNX por un port
#   propio y publico (github.com/santiquiroz/port-bs-roformer-onnx, MIT,
#   release models-v1.0). Tampoco tiene hash UVR: no es un modelo de UVR. Se
#   verifica el SHA-256 del release, tomado de su manifest.json y comprobado
#   contra el archivo descargado el 2026-08-10. Es el unico del pack que pesa
#   casi 1 GB, y el unico ~20x mas lento que el resto: la UI lo advierte antes
#   de que se pueda elegir.
$modelos = @{
    'inst_hq_3' = @{
        File   = 'UVR-MDX-NET-Inst_HQ_3.onnx'
        Url    = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx'
        Hash   = '55657dd70583b0fedfba5f67df11d711'
        Sha256 = '317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc'
        Size   = '~67 MB'
        Label  = 'MDX-Net Inst HQ 3 (saca la instrumental)'
    }
    'voc_ft' = @{
        File   = 'UVR-MDX-NET-Voc_FT.onnx'
        Url    = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx'
        Hash   = '77d07b2667ddf05b9e3175941b4454a0'
        Sha256 = '534b2070fcc7df514b13ef660dc8cbb328679c2374d04354a5c42bb14ecce111'
        Size   = '~67 MB'
        Label  = 'MDX-Net Voc FT (saca la voz)'
    }
    'mel_band_roformer_kim' = @{
        File   = 'mel_band_roformer_kim_T801.onnx'
        Url    = 'https://github.com/santiquiroz/port-bs-roformer-onnx/releases/download/models-v1.0/mel_band_roformer_kim_T801.onnx'
        Sha256 = '1b8afd7780d8a234527748821dee6bc746d346f2088751c12fd48e8c873f625a'
        Size   = '~931 MB'
        Label  = 'Mel-Band RoFormer by KimberleyJSN (saca la voz con la maxima calidad; ~20x mas lento que Inst HQ 3)'
    }
    'reverb_hq' = @{
        File   = 'Reverb_HQ_By_FoxJoy.onnx'
        Url    = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Reverb_HQ_By_FoxJoy.onnx'
        Hash   = 'cd5b2989ad863f116c855db1dfe24e39'
        Sha256 = '233bb5c6aaa365e568659a0a81211746fa881f8f47f82d9e864fce1f7692db80'
        Size   = '~67 MB'
        Label  = 'Reverb HQ by FoxJoy (saca la cola de reverb; la pista limpia es la resta)'
    }
    'deecho_normal' = @{
        File   = 'UVR-De-Echo-Normal.onnx'
        Url    = 'https://github.com/santiquiroz/port-uvr-deecho-onnx/releases/download/models-v1.1/UVR-De-Echo-Normal.onnx'
        Sha256 = 'fc2f9df26060672b72324d6f77a046812361fd8a0dc79ba4f5258a944fc45e14'
        Size   = '~121 MB'
        Label  = 'UVR De-Echo Normal by FoxJoy (saca el eco moderado; la pista limpia es la salida directa)'
    }
    'deecho_aggressive' = @{
        File   = 'UVR-De-Echo-Aggressive.onnx'
        Url    = 'https://github.com/santiquiroz/port-uvr-deecho-onnx/releases/download/models-v1.1/UVR-De-Echo-Aggressive.onnx'
        Sha256 = 'c5f95ecf29cb0be50144ea0ab461ac920854576df47c3ede82420846f699037c'
        Size   = '~121 MB'
        Label  = 'UVR De-Echo Aggressive by FoxJoy (saca el eco fuerte; puede tocar la señal)'
    }
    'deecho_dereverb' = @{
        File   = 'UVR-DeEcho-DeReverb.onnx'
        Url    = 'https://github.com/santiquiroz/port-uvr-deecho-onnx/releases/download/models-v1.1/UVR-DeEcho-DeReverb.onnx'
        Sha256 = 'fe64dfbbeb744cf8a648a25a473ce319bbfb59771eac01f4ff47a77312839bd3'
        Size   = '~213 MB'
        Label  = 'UVR DeEcho-DeReverb by FoxJoy (saca eco Y reverb de una pasada)'
    }
    'denoise' = @{
        File   = 'UVR-DeNoise.onnx'
        Url    = 'https://github.com/santiquiroz/port-uvr-deecho-onnx/releases/download/models-v1.1/UVR-DeNoise.onnx'
        Sha256 = '3285c155a0f8f7295ad971e1fb43fdcb9d8cdbc493c28aececcddb61af26cc63'
        Size   = '~121 MB'
        Label  = 'UVR DeNoise by FoxJoy (saca el ruido de fondo; la pista limpia es la resta)'
    }
}

function Get-UvrHash([string]$path) {
    # MD5 de los ultimos 10000 KiB (o del archivo entero si es mas chico).
    $tailBytes = 10000 * 1024
    $stream = [System.IO.File]::OpenRead($path)
    try {
        if ($stream.Length -gt $tailBytes) {
            $null = $stream.Seek(-$tailBytes, [System.IO.SeekOrigin]::End)
        }
        $md5 = [System.Security.Cryptography.MD5]::Create()
        try {
            $hash = $md5.ComputeHash($stream)
        } finally {
            $md5.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
    return ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

function Get-Sha256([string]$path) {
    # .NET directo y no Get-FileHash: ese cmdlet vive en un modulo, y un server
    # lanzado con PSModulePath contaminado (pasado real 2026-08-09: relanzado
    # desde pwsh 7) deja al powershell 5.1 hijo sin poder resolverlo. .NET
    # siempre esta.
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

function Get-ModelIntegrityError([string]$path, $info) {
    # $null si pasa todas las verificaciones; si no, un texto que dice CUAL
    # fallo. El hash UVR es opcional: los .onnx del port VR no lo tienen porque
    # ese hash identifica el .pth de origen, no el grafo exportado.
    if ($info.ContainsKey('Hash')) {
        $uvr = Get-UvrHash $path
        if ($uvr -ne $info.Hash) {
            return "hash UVR $uvr, se esperaba $($info.Hash)"
        }
    }
    $sha = Get-Sha256 $path
    if ($sha -ne $info.Sha256) {
        return "SHA-256 $sha, se esperaba $($info.Sha256)"
    }
    return $null
}

function Write-KaraokeCredits([string]$directory) {
    $credits = @(
        'Separation models (vendor\karaoke)'
        ''
        'MDX-Net: UVR-MDX-NET Inst HQ 3 / UVR-MDX-NET Voc FT / Reverb HQ'
        'Ultimate Vocal Remover (Anjok07 & aufr33), MIT'
        'Reverb HQ by FoxJoy - distributed via the official UVR Download Center'
        'github.com/Anjok07/ultimatevocalremovergui'
        ''
        'VR De-Echo / De-Reverb / De-Noise: UVR-De-Echo-Normal / UVR-De-Echo-Aggressive /'
        'UVR-DeEcho-DeReverb / UVR-DeNoise'
        'Models by FoxJoy, distributed via the official UVR Download Center;'
        'ONNX port: github.com/santiquiroz/port-uvr-deecho-onnx (MIT)'
        ''
        'Mel-Band RoFormer by KimberleyJSN (MIT);'
        'ONNX port: github.com/santiquiroz/port-bs-roformer-onnx'
    ) -join [Environment]::NewLine
    Set-Content -Path (Join-Path $directory 'CREDITS.txt') -Value $credits -Encoding UTF8
}

$info = $modelos[$Model]
$destino = Join-Path $vendorDir $info.File

if (Test-Path $destino) {
    $fallo = Get-ModelIntegrityError $destino $info
    if ($null -eq $fallo) {
        Write-KaraokeCredits $vendorDir
        Write-Host "Ya esta: $($info.Label) en $destino"
        Write-Host "Borra el archivo para forzar una re-descarga."
        return
    }
    Write-Host "El archivo existente no pasa la verificacion ($fallo); se vuelve a bajar."
    Remove-Item -Force $destino
}

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null

$temporal = "$destino.download"
Write-Host "Descargando $($info.Label) ($($info.Size))..."
Invoke-WebRequest -Uri $info.Url -OutFile $temporal -UseBasicParsing

$fallo = Get-ModelIntegrityError $temporal $info
if ($null -ne $fallo) {
    Remove-Item -Force $temporal -ErrorAction SilentlyContinue
    throw "La descarga no paso la verificacion: $fallo."
}
Move-Item -Force $temporal $destino

Write-KaraokeCredits $vendorDir

Write-Host "Modelo de separacion listo en: $destino"
Write-Host 'Pesos distribuidos por el canal oficial de descargas de UVR (creditos por autor en CREDITS.txt).'
Write-Host 'El modo de separacion del apartado Audio queda habilitado con este modelo.'
