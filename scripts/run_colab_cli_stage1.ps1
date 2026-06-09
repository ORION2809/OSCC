param(
    [switch]$FullTraining
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$RepoWsl = (wsl -d Ubuntu -- bash -lc "wslpath -a '$($RepoWin -replace '\\','\\')'").Trim()
$Session = "oralpath-stage1"
$KaggleWin = Join-Path $HOME ".kaggle\kaggle.json"

function Invoke-WslColab {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    wsl -d Ubuntu -- bash -lc $Command
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed with exit code ${LASTEXITCODE}: $Command"
    }
}

if (-not (Test-Path $KaggleWin)) {
    throw "Missing Kaggle credentials at $KaggleWin"
}

Write-Host "[1/7] Checking Colab CLI in WSL..."
Invoke-WslColab "source ~/.local/bin/env && colab version"

Write-Host "[2/7] Checking Colab auth..."
Invoke-WslColab "source ~/.local/bin/env && colab --auth=adc sessions"

Write-Host "[3/7] Starting Colab T4 session..."
Invoke-WslColab "source ~/.local/bin/env && (colab --auth=adc status -s $Session >/dev/null 2>&1 || colab --auth=adc new -s $Session --gpu T4)"

Write-Host "[4/7] Installing remote base packages..."
Invoke-WslColab "source ~/.local/bin/env && colab --auth=adc install -s $Session git kaggle"

Write-Host "[5/7] Uploading Kaggle credentials..."
Invoke-WslColab "source ~/.local/bin/env && colab --auth=adc upload -s $Session /mnt/c/Users/ShreyasSuvarna/.kaggle/kaggle.json /content/kaggle.json"

Write-Host "[6/7] Running Stage 1 remote job..."
$full = if ($FullTraining) { "ORALPATH_FULL_STAGE1=1 " } else { "" }
Invoke-WslColab "source ~/.local/bin/env && cd '$RepoWsl' && ${full}colab --auth=adc exec -s $Session -f scripts/colab_cli_stage1_job.py"

Write-Host "[7/7] Downloading reports/checkpoints if present..."
New-Item -ItemType Directory -Force -Path (Join-Path $RepoWin "model\evaluation\reports\colab_cli") | Out-Null
wsl -d Ubuntu -- bash -lc "source ~/.local/bin/env && colab --auth=adc download -s $Session /content/oralpath/model/training/stage1_detection/logs/stage1_report.json '$RepoWsl/model/evaluation/reports/colab_cli/stage1_report.json' || true"
wsl -d Ubuntu -- bash -lc "source ~/.local/bin/env && colab --auth=adc download -s $Session /content/oralpath/model/training/stage1_detection/checkpoints/stage1_best.pt '$RepoWsl/model/evaluation/reports/colab_cli/stage1_best.pt' || true"

Write-Host "[DONE] Session is still running for inspection. Stop it with:"
Write-Host "  wsl -d Ubuntu -- bash -lc `"source ~/.local/bin/env && colab --auth=adc stop -s $Session`""
