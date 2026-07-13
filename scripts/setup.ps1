param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $VenvPath)) {
    python -m venv $VenvPath
}

& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\python.exe" -m pip install -e .

Write-Host "Environment ready."
Write-Host "Next: run scripts/download-realesrgan.ps1 and then start the app with:"
Write-Host "  .venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload"
