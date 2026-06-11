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
$LastEpochSeen = -1
$UnchangedChecks = 0
$MaxUnchangedChecks = 3

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
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -ieq "powershell.exe" -and
            $_.CommandLine -like "*run_colab_cli_stage1_chunks.ps1*" -and
            $_.CommandLine -notlike "*watch_colab_stage1_training.ps1*"
        } |
        Select-Object -First 1
}

function Stop-RunnerTree {
    param([uint32]$RootPid)

    $Children = Get-CimInstance Win32_Process |
        Where-Object { $_.ParentProcessId -eq $RootPid } |
        Select-Object -ExpandProperty ProcessId

    foreach ($ChildPid in $Children) {
        Stop-RunnerTree -RootPid $ChildPid
    }

    Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
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
            if ($Epoch -eq $LastEpochSeen) {
                $UnchangedChecks += 1
            } else {
                $UnchangedChecks = 0
                $LastEpochSeen = $Epoch
            }

            Write-WatchdogLog "Runner alive PID=$($Runner.ProcessId). Current epoch=$Epoch / $TargetEpoch. UnchangedChecks=$UnchangedChecks."

            if ($UnchangedChecks -ge $MaxUnchangedChecks) {
                Write-WatchdogLog "Epoch unchanged for $UnchangedChecks checks. Restarting runner tree PID=$($Runner.ProcessId)."
                Stop-RunnerTree -RootPid $Runner.ProcessId
                Start-Sleep -Seconds 5
                Start-Runner
                $UnchangedChecks = 0
            }
        } else {
            Write-WatchdogLog "Runner not found at epoch $Epoch / $TargetEpoch. Restarting."
            Start-Runner
            $LastEpochSeen = $Epoch
            $UnchangedChecks = 0
        }
    } catch {
        Write-WatchdogLog "Watchdog error: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds $CheckSeconds
}
