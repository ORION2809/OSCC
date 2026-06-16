param(
    [switch]$Logs
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$Kernel = "oralpath_user/oralpath-stage2-orchid-level1"

Write-Host "[STATUS] $Kernel"
& "$RepoWin\.venv\Scripts\python.exe" -m kaggle kernels status $Kernel
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle status check failed."
}

if ($Logs) {
    Write-Host ""
    Write-Host "[LOGS] $Kernel"
    & "$RepoWin\.venv\Scripts\python.exe" -m kaggle kernels logs $Kernel
    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle logs check failed."
    }
}
