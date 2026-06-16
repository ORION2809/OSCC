param(
    [switch]$Logs
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$Kernel = "oralpath_user/oralpath-stage2-orchid-level1"
$LegacyCliDir = Join-Path $env:TEMP "oralpath_kaggle_legacy_cli"
$KaggleExe = Join-Path $LegacyCliDir "Scripts\kaggle.exe"

if (-not (Test-Path $KaggleExe)) {
    py -3 -m venv $LegacyCliDir
    & (Join-Path $LegacyCliDir "Scripts\python.exe") -m pip install -q "kaggle==1.7.4.5"
}

Write-Host "[STATUS] $Kernel"
& $KaggleExe kernels status $Kernel
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle status check failed."
}

if ($Logs) {
    Write-Host ""
    Write-Host "[LOGS] $Kernel"
    & $KaggleExe kernels logs $Kernel
    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle logs check failed."
    }
}
