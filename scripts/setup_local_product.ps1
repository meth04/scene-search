[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$EnvironmentRoot = "",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [string]$ModelBundle = "",
    [switch]$SkipAssets,
    [switch]$VerifyOnly,
    [switch]$Repair
)
$ErrorActionPreference = "Stop"
$productRepo = Split-Path -Parent $PSScriptRoot
if (-not $EnvironmentRoot) { $EnvironmentRoot = $productRepo }
$EnvironmentRoot = [System.IO.Path]::GetFullPath($EnvironmentRoot)
$productInstallDevice = $Device
if ($productInstallDevice -eq "auto") {
    $productInstallDevice = "cpu"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $productInstallDevice = "cuda" }
    }
}
if (-not $VerifyOnly -and -not $SkipAssets) {
    $productAssetArgs = @((Join-Path $productRepo "scripts/install_product_assets.py"))
    if ($ModelBundle) { $productAssetArgs += @("--bundle", $ModelBundle) }
    if ($Repair) { $productAssetArgs += "--repair" }
    & $Python @productAssetArgs
    if ($LASTEXITCODE -ne 0) { throw "Runtime asset download failed. Rerun setup to resume." }
}
$productEnvironments = @{
    backend = ".venv"; worker = ".venv_embeddinggemma2"; frontend = ".venv_demo"
}
foreach ($productRole in @("backend", "worker", "frontend")) {
    $productEnvironment = Join-Path $EnvironmentRoot $productEnvironments[$productRole]
    $productPython = Join-Path $productEnvironment "Scripts\python.exe"
    $productInstallMarker = Join-Path $productEnvironment ".product-installing"
    $productCreated = $false
    if (-not (Test-Path -LiteralPath $productPython)) {
        if ($VerifyOnly) { throw "Missing $productRole environment: $productPython" }
        & $Python -c "import struct,sys; assert sys.version_info[:2] == (3,13) and struct.calcsize('P')==8, 'Python 3.13 64-bit is required by the verified product locks'"
        if ($LASTEXITCODE -ne 0) { throw "Use Python 3.13." }
        & $Python -m venv $productEnvironment
        if ($LASTEXITCODE -ne 0) { throw "Environment creation failed." }
        $productCreated = $true
        Set-Content -LiteralPath $productInstallMarker -Value $productRole -Encoding utf8
    }
    if (-not $VerifyOnly -and ($productCreated -or $Repair -or (Test-Path -LiteralPath $productInstallMarker))) {
        $productLock = Join-Path $productRepo "configs/product/requirements-$productRole.lock"
        $productWheelIndex = "https://download.pytorch.org/whl/cu126"
        if ($productInstallDevice -eq "cpu" -and $productRole -ne "frontend") {
            $productLockLines = @(Get-Content -LiteralPath $productLock | Where-Object { $_ -notmatch '^torchcodec==' } | ForEach-Object { $_ -replace '\+cu126$', '+cpu' })
            $productCpuLockDir = Join-Path $productRepo "outputs/install"
            New-Item -ItemType Directory -Force -Path $productCpuLockDir | Out-Null
            $productLock = Join-Path $productCpuLockDir "requirements-$productRole-cpu.lock"
            [System.IO.File]::WriteAllLines($productLock, [string[]]$productLockLines, [System.Text.UTF8Encoding]::new($false))
            $productWheelIndex = "https://download.pytorch.org/whl/cpu"
        }
        & $productPython -m pip install --disable-pip-version-check --extra-index-url $productWheelIndex -r $productLock
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed for $productRole. Rerun setup to resume the installation." }
        Remove-Item -LiteralPath $productInstallMarker -ErrorAction SilentlyContinue
    }
}
if (-not $VerifyOnly) {
    $productRuntimeDir = Join-Path $productRepo "outputs/product"
    New-Item -ItemType Directory -Force -Path $productRuntimeDir | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $productRuntimeDir "runtime.json"), (@{device=$Device} | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
}
$previousWorker = $env:EMBEDDINGGEMMA2_PYTHON
$previousFrontend = $env:SCENESEARCH_FRONTEND_PYTHON
try {
    $env:EMBEDDINGGEMMA2_PYTHON = Join-Path $EnvironmentRoot ".venv_embeddinggemma2\Scripts\python.exe"
    $env:SCENESEARCH_FRONTEND_PYTHON = Join-Path $EnvironmentRoot ".venv_demo\Scripts\python.exe"
    Push-Location $productRepo
    try {
        & (Join-Path $EnvironmentRoot ".venv\Scripts\python.exe") -B -m demo.product_doctor
        if ($LASTEXITCODE -ne 0) { throw "Doctor failed. Check dependency/device diagnostics; use -Device cpu if CUDA is unavailable. Use -Repair only to replace changed assets or reinstall dependencies." }
    } finally { Pop-Location }
    if ($EnvironmentRoot -eq $productRepo) {
        Write-Host "Setup verified. Start daily with: powershell -File run_demo.ps1"
    } else {
        Write-Host "Setup verified. Start run_demo.ps1 with -EnginePython, -WorkerPython and -FrontendPython pointing to the environments under $EnvironmentRoot."
    }
} finally {
    $env:EMBEDDINGGEMMA2_PYTHON = $previousWorker
    $env:SCENESEARCH_FRONTEND_PYTHON = $previousFrontend
}
