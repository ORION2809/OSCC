param(
    [switch]$FullTraining,
    [int]$FullEpochs = 0
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$RepoWsl = (wsl -d Ubuntu -- bash -lc "wslpath -a '$($RepoWin -replace '\\','\\')'").Trim()
$Session = "oralpath-stage1"
$ColabAuth = "oauth2"
$KaggleWin = Join-Path $HOME ".kaggle\kaggle.json"
$HfTokenWin = Join-Path $env:TEMP "oralpath_hf_token.secret"
$FullFlagWin = Join-Path $env:TEMP "oralpath_full_stage1.flag"
$FullEpochsWin = Join-Path $env:TEMP "oralpath_full_stage1_epochs.txt"
$ResumeStateWin = Join-Path $RepoWin "model\evaluation\reports\colab_cli\stage1_last.pt"

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
Invoke-WslColab "source ~/.local/bin/env && colab --auth=$ColabAuth sessions"

Write-Host "[3/7] Starting Colab T4 session..."
$sessionList = wsl -d Ubuntu -- bash -lc "source ~/.local/bin/env && colab --auth=$ColabAuth sessions"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to list Colab sessions."
}
if ($sessionList -match [regex]::Escape($Session)) {
    Write-Host "[OK] Reusing existing Colab session: $Session"
} else {
    Invoke-WslColab "source ~/.local/bin/env && colab --auth=$ColabAuth new -s $Session --gpu T4"
}
Invoke-WslColab "source ~/.local/bin/env && colab --auth=$ColabAuth status -s $Session"
$sessionUrl = (wsl -d Ubuntu -- bash -lc "source ~/.local/bin/env && colab --auth=$ColabAuth url -s $Session").Trim()
if ($sessionUrl -match "^https?://") {
    Write-Host "[3/7] Opening Colab session URL in browser for runtime keepalive..."
    Start-Process $sessionUrl
} else {
    Write-Host "[3/7] Could not get browser URL for Colab session."
}

Write-Host "[4/7] Remote package install is handled inside the job..."

Write-Host "[5/7] Uploading Kaggle credentials..."
Invoke-WslColab "source ~/.local/bin/env && colab --auth=$ColabAuth upload -s $Session /mnt/c/Users/ShreyasSuvarna/.kaggle/kaggle.json /content/kaggle.json"
if ($env:HF_TOKEN) {
    Set-Content -Path $HfTokenWin -Value $env:HF_TOKEN -NoNewline -Encoding ascii
    $HfTokenWsl = (wsl -d Ubuntu -- bash -lc "wslpath -a '$($HfTokenWin -replace '\\','\\')'").Trim()
    try {
        Write-Host "[5/7] Uploading Hugging Face token..."
        Invoke-WslColab "source ~/.local/bin/env && colab --auth=$ColabAuth upload -s $Session '$HfTokenWsl' /content/hf_token.secret"
    } finally {
        Remove-Item -LiteralPath $HfTokenWin -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "[5/7] HF_TOKEN is not set locally; UNI may fall back if gated."
}
if ($FullTraining) {
    Set-Content -Path $FullFlagWin -Value "1" -NoNewline -Encoding ascii
    $FullFlagWsl = (wsl -d Ubuntu -- bash -lc "wslpath -a '$($FullFlagWin -replace '\\','\\')'").Trim()
    try {
        Write-Host "[5/7] Uploading full-training flag..."
        Invoke-WslColab "source ~/.local/bin/env && colab --auth=$ColabAuth upload -s $Session '$FullFlagWsl' /content/full_stage1.flag"
    } finally {
        Remove-Item -LiteralPath $FullFlagWin -Force -ErrorAction SilentlyContinue
    }
    if ($FullEpochs -gt 0) {
        Set-Content -Path $FullEpochsWin -Value $FullEpochs -NoNewline -Encoding ascii
        $FullEpochsWsl = (wsl -d Ubuntu -- bash -lc "wslpath -a '$($FullEpochsWin -replace '\\','\\')'").Trim()
        try {
            Write-Host "[5/7] Uploading full-training epoch limit: $FullEpochs..."
            Invoke-WslColab "source ~/.local/bin/env && colab --auth=$ColabAuth upload -s $Session '$FullEpochsWsl' /content/full_stage1_epochs.txt"
        } finally {
            Remove-Item -LiteralPath $FullEpochsWin -Force -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path $ResumeStateWin) {
        $ResumeStateWsl = (wsl -d Ubuntu -- bash -lc "wslpath -a '$($ResumeStateWin -replace '\\','\\')'").Trim()
        Write-Host "[5/7] Uploading resumable Stage 1 state..."
        Invoke-WslColab "source ~/.local/bin/env && colab --auth=$ColabAuth upload -s $Session '$ResumeStateWsl' /content/stage1_resume.pt"
    }
}

Write-Host "[6/7] Running Stage 1 remote job..."
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    wsl -d Ubuntu -- bash -lc "source ~/.local/bin/env && cd '$RepoWsl' && colab --auth=$ColabAuth exec -s $Session --timeout 7200 -f scripts/colab_cli_stage1_job.py"
    $ExecExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}
if ($ExecExitCode -ne 0) {
    Write-Warning "Colab exec returned exit code ${ExecExitCode}; continuing to artifact download."
}

Write-Host "[7/7] Downloading reports/checkpoints if present..."
New-Item -ItemType Directory -Force -Path (Join-Path $RepoWin "model\evaluation\reports\colab_cli") | Out-Null
wsl -d Ubuntu -- bash -lc "source ~/.local/bin/env && colab --auth=$ColabAuth download -s $Session /content/oralpath/model/training/stage1_detection/logs/stage1_report.json '$RepoWsl/model/evaluation/reports/colab_cli/stage1_report.json' || true"
wsl -d Ubuntu -- bash -lc "source ~/.local/bin/env && colab --auth=$ColabAuth download -s $Session /content/oralpath/model/training/stage1_detection/checkpoints/stage1_last.pt '$RepoWsl/model/evaluation/reports/colab_cli/stage1_last.pt' || true"
wsl -d Ubuntu -- bash -lc "source ~/.local/bin/env && colab --auth=$ColabAuth download -s $Session /content/oralpath/model/training/stage1_detection/checkpoints/stage1_best.pt '$RepoWsl/model/evaluation/reports/colab_cli/stage1_best.pt' || true"

Write-Host "[DONE] Session is still running for inspection. Stop it with:"
Write-Host "  wsl -d Ubuntu -- bash -lc `"source ~/.local/bin/env && colab --auth=$ColabAuth stop -s $Session`""
