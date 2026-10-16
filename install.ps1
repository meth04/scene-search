[CmdletBinding()]
param(
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [string]$ModelBundle = "",
    [switch]$Repair,
    [switch]$Start
)
$ErrorActionPreference = "Stop"
$installRoot = $PSScriptRoot
$installPython = $null
function Find-ProductPython {
    $installCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs/Python/Python313/python.exe"),
        "python"
    )
    foreach ($installCandidate in $installCandidates) {
        if (-not (Get-Command $installCandidate -ErrorAction SilentlyContinue)) { continue }
        try {
            $installResolvedPython = & $installCandidate -c "import struct,sys; assert sys.version_info[:2]==(3,13) and struct.calcsize('P')==8; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $installResolvedPython) { return [string]$installResolvedPython }
        } catch { }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $installResolvedPython = & py -3.13 -c "import struct,sys; assert struct.calcsize('P')==8; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $installResolvedPython) { return [string]$installResolvedPython }
        } catch { }
    }
    return $null
}
$installPython = Find-ProductPython
if (-not $installPython) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Install Python 3.13 (64-bit) from python.org, then rerun install.ps1. See INSTALL.md."
    }
    & winget install --id Python.Python.3.13 --exact --scope user --silent --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw "Python installation failed. Install Python 3.13 manually, then rerun." }
    $installPython = Find-ProductPython
    if (-not $installPython) { throw "Python 3.13 was not found. Open a new PowerShell and rerun install.ps1." }
}
& (Join-Path $installRoot "scripts/setup_local_product.ps1") -Python $installPython -Device $Device -ModelBundle $ModelBundle -Repair:$Repair
if ($LASTEXITCODE -ne 0) { throw "Product setup failed. Rerun to resume; see INSTALL.md." }
if ($Start) { & (Join-Path $installRoot "run_demo.ps1") }
else { Write-Host "Ready. Start with: powershell -NoProfile -File run_demo.ps1" -ForegroundColor Green }
