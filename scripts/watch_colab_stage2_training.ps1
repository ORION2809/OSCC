param(
    [int]$TargetEpoch = 50,
    [int]$CheckSeconds = 300
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$LogDir = Join-Path $RepoWin "model\evaluation\reports\colab_cli"
$StatePath = Join-Path $LogDir "stage2_last.pt"
$RunnerScript = Join-Path $RepoWin "scripts\run_colab_cli_stage2_chunks.ps1"
$RunnerLog = Join-Path $LogDir "stage2_chunks_to_50_runner.log"
$WatchdogLog = Join-Path $LogDir "stage2_watchdog.log"
$HfTokenPath = Join-Path $HOME ".cache\huggingface\token"
$LastEpochSeen = -1
$LastActivitySeen = $null
$UnchangedChecks = 0
$MaxUnchangedChecks = 3

function Write-WatchdogLog {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date).ToString("s"), $Message
    Add-Content -Path $WatchdogLog -Value $Line -Encoding utf8
    Write-Host $Line
}

function Get-Stage2Epoch {
    if (-not (Test-Path $StatePath)) {
        return 0
    }

    $Code = "import torch; s=torch.load(r'$StatePath', map_location='cpu'); print(int(s.get('epoch', 0)))"
    $Output = py -3 -c $Code
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to read Stage 2 state epoch from $StatePath"
    }

    return [int]($Output | Select-Object -Last 1)
}

function Get-RunnerProcess {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -ieq "powershell.exe" -and
            $_.CommandLine -like "*run_colab_cli_stage2_chunks.ps1*" -and
            $_.CommandLine -notlike "*watch_colab_stage2_training.ps1*"
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
    Write-WatchdogLog "Started Stage 2 runner PID=$($Process.Id)"
}

function Get-RunnerActivityStamp {
    $paths = @($RunnerLog)
    $paths += Get-ChildItem -Path $LogDir -Filter "stage2_chunk_epoch_*.log" -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName

    $latest = $null
    foreach ($path in $paths) {
        if (-not (Test-Path $path)) {
            continue
        }
        $item = Get-Item $path
        if ($null -eq $latest -or $item.LastWriteTimeUtc -gt $latest) {
            $latest = $item.LastWriteTimeUtc
        }
    }
    return $latest
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Write-WatchdogLog "Watchdog started. TargetEpoch=$TargetEpoch CheckSeconds=$CheckSeconds"

while ($true) {
    try {
        $Epoch = Get-Stage2Epoch
        if ($Epoch -ge $TargetEpoch) {
            Write-WatchdogLog "Training complete at epoch $Epoch / $TargetEpoch. Watchdog exiting."
            break
        }

        $Runner = Get-RunnerProcess
        if ($Runner) {
            $Activity = Get-RunnerActivityStamp
            $ActivityChanged = $false
            if ($null -ne $Activity -and $Activity -ne $LastActivitySeen) {
                $ActivityChanged = $true
                $LastActivitySeen = $Activity
            }

            if ($Epoch -eq $LastEpochSeen -and -not $ActivityChanged) {
                $UnchangedChecks += 1
            } else {
                $UnchangedChecks = 0
                $LastEpochSeen = $Epoch
            }

            Write-WatchdogLog "Runner alive PID=$($Runner.ProcessId). Current epoch=$Epoch / $TargetEpoch. LogActivityChanged=$ActivityChanged. UnchangedChecks=$UnchangedChecks."

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
