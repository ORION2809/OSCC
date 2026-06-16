param(
    [string]$OutputDir = "model\evaluation\reports\kaggle_stage2"
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$Kernel = "oralpath_user/oralpath-stage2-orchid-level1"
$OutputPath = Join-Path $RepoWin $OutputDir

New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

Write-Host "[DOWNLOAD] $Kernel -> $OutputPath"
& "$RepoWin\.venv\Scripts\python.exe" -m kaggle kernels output $Kernel -p $OutputPath -o
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle output download failed."
}

Get-ChildItem $OutputPath | Select-Object Name,Length,LastWriteTime
