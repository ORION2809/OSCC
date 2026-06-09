param(
    [int]$TargetEpoch = 50,
    [int]$MaxChunks = 0
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$StatePath = Join-Path $RepoWin "model\evaluation\reports\colab_cli\stage1_last.pt"
$LogDir = Join-Path $RepoWin "model\evaluation\reports\colab_cli"
$HfTokenPath = Join-Path $HOME ".cache\huggingface\token"

function Get-Stage1Epoch {
    if (-not (Test-Path $StatePath)) {
        return 0
    }

    $Code = "import torch; s=torch.load(r'$StatePath', map_location='cpu'); print(int(s.get('epoch', 0)))"
    $Output = py -3 -c $Code
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to read Stage 1 state epoch from $StatePath"
    }

    return [int]($Output | Select-Object -Last 1)
}

if (-not $env:HF_TOKEN -and (Test-Path $HfTokenPath)) {
    $env:HF_TOKEN = (Get-Content $HfTokenPath -Raw).Trim()
}

if (-not $env:HF_TOKEN) {
    throw "HF_TOKEN is not set and no Hugging Face cache token was found at $HfTokenPath"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ChunksRun = 0
while ($true) {
    $CurrentEpoch = Get-Stage1Epoch
    if ($CurrentEpoch -ge $TargetEpoch) {
        Write-Host "[DONE] Stage 1 already reached epoch $CurrentEpoch / $TargetEpoch"
        break
    }

    if ($MaxChunks -gt 0 -and $ChunksRun -ge $MaxChunks) {
        Write-Host "[STOP] Reached MaxChunks=$MaxChunks at epoch $CurrentEpoch / $TargetEpoch"
        break
    }

    $NextEpoch = $CurrentEpoch + 1
    $LogPath = Join-Path $LogDir ("stage1_chunk_epoch_{0:D2}.log" -f $NextEpoch)
    Write-Host "[CHUNK] Running Stage 1 epoch $NextEpoch / $TargetEpoch"
    & "$PSScriptRoot\run_colab_cli_stage1.ps1" -FullTraining -FullEpochs 1 *> $LogPath

    $AfterEpoch = Get-Stage1Epoch
    if ($AfterEpoch -lt $NextEpoch) {
        throw "Stage 1 did not advance to epoch $NextEpoch. Current state epoch: $AfterEpoch. See $LogPath"
    }

    $ChunksRun += 1
}
