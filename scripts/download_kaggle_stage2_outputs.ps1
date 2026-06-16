param(
    [string]$OutputDir = "model\evaluation\reports\kaggle_stage2"
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$Kernel = "oralpath_user/oralpath-stage2-orchid-level1"
$OutputPath = Join-Path $RepoWin $OutputDir
$LegacyCliDir = Join-Path $env:TEMP "oralpath_kaggle_legacy_cli"
$KaggleExe = Join-Path $LegacyCliDir "Scripts\kaggle.exe"

if (-not (Test-Path $KaggleExe)) {
    py -3 -m venv $LegacyCliDir
    & (Join-Path $LegacyCliDir "Scripts\python.exe") -m pip install -q "kaggle==1.7.4.5"
}

New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

Write-Host "[DOWNLOAD] $Kernel -> $OutputPath"
& $KaggleExe kernels output $Kernel -p $OutputPath -o
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle output download failed."
}

Get-ChildItem $OutputPath | Select-Object Name,Length,LastWriteTime
