[CmdletBinding()]
param([string]$Python = "", [string[]]$Targets = @("tests"))
$ErrorActionPreference = "Stop"
$productRepo = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $productRepo ".venv\Scripts\python.exe" }
$previousPath = $env:PATH
try {
    $env:PATH = (Join-Path $productRepo "outputs/ffmpeg-main/package") + ";" + (Join-Path $productRepo "outputs/ffprobe-main/package") + ";" + $env:PATH
    Push-Location $productRepo
    try {
        New-Item -ItemType Directory -Force -Path outputs/finalization | Out-Null
        & $Python -B -m pytest @Targets -q --junitxml=outputs/finalization/regression_final.xml
        exit $LASTEXITCODE
    } finally { Pop-Location }
} finally { $env:PATH = $previousPath }
