[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8501,
    [ValidateRange(1024, 65535)][int]$ApiPort = 8090,
    [string]$EnginePython = "",
    [string]$FrontendPython = "",
    [string]$WorkerPython = "",
    [switch]$Verify
)

$ErrorActionPreference = "Stop"
$demoRoot = $PSScriptRoot
$demoPython = Join-Path $demoRoot ".venv_demo\Scripts\python.exe"
if ($FrontendPython) { $demoPython = $FrontendPython }
if (-not $EnginePython) { $EnginePython = Join-Path $demoRoot ".venv\Scripts\python.exe" }
$eg2Python = Join-Path $demoRoot ".venv_embeddinggemma2\Scripts\python.exe"
if ($WorkerPython) { $eg2Python = $WorkerPython }
$demoLogs = Join-Path $demoRoot "outputs\product\logs"
$ownedProcesses = @()
$previousDemoPath = $env:PATH
$env:PATH = (Join-Path $demoRoot "data/bin/ffmpeg/bin") + ";" + $env:PATH
$previousApi = [Environment]::GetEnvironmentVariable("SCENESEARCH_DEMO_API", "Process")
$previousEg2 = [Environment]::GetEnvironmentVariable("EMBEDDINGGEMMA2_PYTHON", "Process")

function Wait-DemoEndpoint([string]$Url, $Process, [int]$Seconds = 60) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process -and $Process.HasExited) { throw "Service exited. See logs in $demoLogs" }
        try { return Invoke-RestMethod -Uri $Url -TimeoutSec 2 } catch { Start-Sleep -Milliseconds 300 }
    }
    throw "Service did not become ready: $Url. See logs in $demoLogs"
}

function Test-DemoPort([int]$Number) {
    $demoListener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Number)
    try { $demoListener.Start(); return $false } catch { return $true } finally { $demoListener.Stop() }
}

try {
    if ($Port -eq $ApiPort) { throw "Streamlit and backend need different ports." }
    if (-not (Test-Path -LiteralPath $EnginePython)) { throw "Engine Python missing: $EnginePython. Follow SETUP_ENGINE.md." }
    if (-not (Test-Path -LiteralPath $eg2Python)) { throw "EG2 worker environment missing. Follow SETUP_ENGINE.md." }
    $env:EMBEDDINGGEMMA2_PYTHON = $eg2Python
    $previousFrontend = $env:SCENESEARCH_FRONTEND_PYTHON
    $env:SCENESEARCH_FRONTEND_PYTHON = $demoPython
    Push-Location $demoRoot
    try {
        & $EnginePython -B -m demo.product_doctor
        if ($LASTEXITCODE -ne 0) { throw "Local doctor failed. Run scripts/setup_local_product.ps1." }
    } finally { Pop-Location }
    if ($Verify) { Write-Host "Verification passed. No services, media processing or browser actions were started."; return }
    if (Test-DemoPort $Port) { throw "Port $Port is occupied. Use -Port with a free port, or stop the existing UI." }
    New-Item -ItemType Directory -Force -Path $demoLogs | Out-Null
    if (-not (Test-Path -LiteralPath $demoPython)) {
        throw "Frontend environment missing. Run scripts/setup_local_product.ps1."
    }
    & $demoPython -c "import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(n) for n in ['streamlit','httpx']) else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend dependencies missing. Run scripts/setup_local_product.ps1."
    }
    $env:SCENESEARCH_DEMO_API = "http://127.0.0.1:$ApiPort"
    $env:EMBEDDINGGEMMA2_PYTHON = $eg2Python
    $apiProcess = $null
    if (Test-DemoPort $ApiPort) {
        $apiHealth = Invoke-RestMethod -Uri "$env:SCENESEARCH_DEMO_API/health" -TimeoutSec 3
        if ($apiHealth.service -ne "scenesearch-workspace" -or -not $apiHealth.library_enabled) {
            throw "Port $ApiPort belongs to another backend. Use -ApiPort with a free port."
        }
        Write-Host "Reusing existing Scene Search backend."
    } else {
        $backendScript = Join-Path $demoRoot "scripts\run_streamlit_demo_backend.py"
        $apiProcess = Start-Process -FilePath $EnginePython -ArgumentList @("-B", ('"{0}"' -f $backendScript), "--port", $ApiPort) `
            -WorkingDirectory $demoRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $demoLogs "backend.stdout.log") -RedirectStandardError (Join-Path $demoLogs "backend.stderr.log")
        $ownedProcesses += $apiProcess
        $apiHealth = Wait-DemoEndpoint "$env:SCENESEARCH_DEMO_API/health" $apiProcess 90
        if ($apiHealth.service -ne "scenesearch-workspace") { throw "Unexpected backend identity." }
    }
    $null = Wait-DemoEndpoint "$env:SCENESEARCH_DEMO_API/ready" $apiProcess 600
    $appScript = Join-Path $demoRoot "demo\streamlit_app.py"
    $uiProcess = Start-Process -FilePath $demoPython -ArgumentList @("-B", "-m", "streamlit", "run", ('"{0}"' -f $appScript), "--server.port", $Port, "--server.address", "127.0.0.1", "--server.headless", "true", "--browser.gatherUsageStats", "false") `
        -WorkingDirectory $demoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $demoLogs "streamlit.stdout.log") -RedirectStandardError (Join-Path $demoLogs "streamlit.stderr.log")
    $ownedProcesses += $uiProcess
    $null = Wait-DemoEndpoint "http://127.0.0.1:$Port/_stcore/health" $uiProcess
    Write-Host ""
    Write-Host "Scene Search: http://127.0.0.1:$Port" -ForegroundColor Cyan
    Write-Host "Video search | personal library | video uploads and processing"
    Write-Host "Logs: $demoLogs"
    if ($Verify) {
        & $demoPython -B (Join-Path $demoRoot "scripts\verify_streamlit_demo.py") --url "http://127.0.0.1:$Port" --api $env:SCENESEARCH_DEMO_API
        if ($LASTEXITCODE -ne 0) { throw "End-to-end verification failed. See outputs/streamlit_demo." }
    } else {
        Write-Host "Keep this terminal open. Ctrl+C stops the services started by this launcher."
        while (-not $uiProcess.HasExited) {
            if ($apiProcess -and $apiProcess.HasExited) { throw "Backend stopped. See backend.stderr.log." }
            Start-Sleep -Milliseconds 500
        }
        if ($uiProcess.ExitCode -ne 0) { throw "Streamlit stopped unexpectedly. See streamlit.stderr.log." }
    }
} finally {
    # Stop only the children this invocation owns; never kill a reused service.
    foreach ($child in $ownedProcesses) {
        if (-not $child.HasExited) {
            & taskkill.exe /PID $child.Id /T /F 2>$null | Out-Null
        }
    }
    [Environment]::SetEnvironmentVariable("SCENESEARCH_DEMO_API", $previousApi, "Process")
    [Environment]::SetEnvironmentVariable("EMBEDDINGGEMMA2_PYTHON", $previousEg2, "Process")
    [Environment]::SetEnvironmentVariable("SCENESEARCH_FRONTEND_PYTHON", $previousFrontend, "Process")
    $env:PATH = $previousDemoPath
}
