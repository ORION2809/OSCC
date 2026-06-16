param(
    [int]$TimeoutSeconds = 43200
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$KernelDir = Join-Path $RepoWin "model\kaggle\stage2_orchid_level1"
$KagglePython = Join-Path $RepoWin ".venv\Scripts\python.exe"

function Set-KaggleApiToken {
    $token = (& $KagglePython -m kaggle auth print-access-token).Trim()
    if (-not $token) {
        throw "Kaggle OAuth token not available. Run: .venv\Scripts\python.exe -m kaggle auth login"
    }
    $env:KAGGLE_API_TOKEN = $token
}

if (-not (Test-Path (Join-Path $HOME ".kaggle\kaggle.json"))) {
    throw "Missing Kaggle credentials at $HOME\.kaggle\kaggle.json"
}

Write-Host "[1/3] Verifying Kaggle CLI..."
Set-KaggleApiToken
& $KagglePython -m kaggle --version
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle CLI check failed."
}

Write-Host "[2/3] Verifying ORCHID Kaggle dataset is visible..."
$DatasetOutput = & $KagglePython -m kaggle datasets files nazmulxdxd/orchid-oscc-classification 2>&1
$DatasetExitCode = $LASTEXITCODE
$DatasetOutput | Select-Object -First 8
if ($DatasetExitCode -ne 0) {
    throw "Unable to access nazmulxdxd/orchid-oscc-classification. Check Kaggle account/dataset access."
}

Write-Host "[3/3] Pushing Kaggle Stage 2 kernel..."
& $KagglePython -m kaggle kernels push -p $KernelDir --accelerator gpu -t $TimeoutSeconds
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle kernel push failed."
}

Write-Host "[DONE] Kaggle Stage 2 submitted."
Write-Host "Check status:"
Write-Host "  .\scripts\kaggle_stage2_status.ps1"
Remove-Item Env:\KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
