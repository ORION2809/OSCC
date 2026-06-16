param(
    [string]$OutputDir = "model\evaluation\reports\kaggle_stage2"
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$Kernel = "shreyassuvarna123/oralpath-stage-2-orchid-level-1-training"
$OutputPath = Join-Path $RepoWin $OutputDir
$KagglePython = Join-Path $RepoWin ".venv\Scripts\python.exe"

if (-not $env:KAGGLE_API_TOKEN) {
    $token = (& $KagglePython -m kaggle auth print-access-token).Trim()
    if ($token) {
        $env:KAGGLE_API_TOKEN = $token
    }
}

New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null

Write-Host "[DOWNLOAD] $Kernel -> $OutputPath"
& $KagglePython -m kaggle kernels output $Kernel -p $OutputPath -o
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle output download failed."
}

Get-ChildItem $OutputPath | Select-Object Name,Length,LastWriteTime
Remove-Item Env:\KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
