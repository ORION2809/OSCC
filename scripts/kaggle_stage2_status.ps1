param(
    [switch]$Logs
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$Kernel = "shreyassuvarna123/oralpath-stage-2-orchid-level-1-training"
$KagglePython = Join-Path $RepoWin ".venv\Scripts\python.exe"

if (-not $env:KAGGLE_API_TOKEN) {
    $token = (& $KagglePython -m kaggle auth print-access-token).Trim()
    if ($token) {
        $env:KAGGLE_API_TOKEN = $token
    }
}

Write-Host "[STATUS] $Kernel"
& $KagglePython -m kaggle kernels status $Kernel
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle status check failed."
}

if ($Logs) {
    Write-Host ""
    Write-Host "[LOGS] $Kernel"
    & $KagglePython -m kaggle kernels logs $Kernel
    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle logs check failed."
    }
}
Remove-Item Env:\KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
