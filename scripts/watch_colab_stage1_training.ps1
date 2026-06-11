param(
    [int]$TargetEpoch = 50,
    [int]$CheckSeconds = 300
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$LogDir = Join-Path $RepoWin "model\evaluation\reports\colab_cli"
$StatePath = Join-Path $LogDir "stage1_last.pt"
$RunnerScript = Join-Path $RepoWin "scripts\run_colab_cli_stage1_chunks.ps1"
$RunnerLog = Join-Path $LogDir "stage1_chunks_to_50_runner.log"
$WatchdogLog = Join-Path $LogDir "stage1_watchdog.log"
$HfTokenPath = Join-Path $HOME ".cache\huggingface\token"

function Write-WatchdogLog {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date).ToString("s"), $Message
    Add-Content -Path $WatchdogLog -Value $Line -Encoding utf8
    Write-Host $Line
}

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

function Get-RunnerProcess {
    $EscapedScript = $RunnerScript.Replace("\", "\\")
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -ieq "powershell.exe" -and
            $_.CommandLine -like "*run_colab_cli_stage1_chunks.ps1*" -and
            $_.CommandLine -notlike "*watch_colab_stage1_training.ps1*"
        } |
        Select-Object -First 1
}

function Start-Runner {
    if (-not $env:HF_TOKEN -and (Test-Path $HfTokenPath)) {
        $env:HF_TOKEN = (Get-Content $HfTokenPath -Raw).Trim()
    }

    $Command = "Set-Location '$RepoWin'; & '$RunnerScript' -TargetEpoch $TargetEpoch *> '$RunnerLog'"
    $Args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $Command)
    $Process = Start-Process -FilePath "powershell.exe" -ArgumentList $Args -WindowStyle Hidden -PassThru
    Write-WatchdogLog "Started Stage 1 runner PID=$($Process.Id)"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Write-WatchdogLog "Watchdog started. TargetEpoch=$TargetEpoch CheckSeconds=$CheckSeconds"

while ($true) {
    try {
        $Epoch = Get-Stage1Epoch
        if ($Epoch -ge $TargetEpoch) {
            Write-WatchdogLog "Training complete at epoch $Epoch / $TargetEpoch. Watchdog exiting."
            break
        }

        $Runner = Get-RunnerProcess
        if ($Runner) {
            Write-WatchdogLog "Runner alive PID=$($Runner.ProcessId). Current epoch=$Epoch / $TargetEpoch."
        } else {
            Write-WatchdogLog "Runner not found at epoch $Epoch / $TargetEpoch. Restarting."
            Start-Runner
        }
    } catch {
        Write-WatchdogLog "Watchdog error: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds $CheckSeconds
}
