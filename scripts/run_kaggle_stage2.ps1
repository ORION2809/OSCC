param(
    [int]$TimeoutSeconds = 43200,
    [switch]$SkipPreflight,
    [int]$PreflightPollSeconds = 30,
    [int]$PreflightTimeoutMinutes = 30
)

$ErrorActionPreference = "Stop"

$RepoWin = (Resolve-Path "$PSScriptRoot\..").Path
$KernelDir = Join-Path $RepoWin "model\kaggle\stage2_orchid_level1"
$PreflightKernelDir = Join-Path $RepoWin "model\kaggle\stage2_orchid_level1_preflight"
$KagglePython = Join-Path $RepoWin ".venv\Scripts\python.exe"
$KernelSlug = "shreyassuvarna123/oralpath-stage-2-orchid-level-1-training"
$PreflightKernelSlug = "shreyassuvarna123/oralpath-stage-2-orchid-level-1-preflight"

function Set-KaggleApiToken {
    $token = (& $KagglePython -m kaggle auth print-access-token).Trim()
    if (-not $token) {
        throw "Kaggle OAuth token not available. Run: .venv\Scripts\python.exe -m kaggle auth login"
    }
    $env:KAGGLE_API_TOKEN = $token
}

function Prepare-KaggleBundle {
    param(
        [Parameter(Mandatory = $true)]
        [string]$KernelPath
    )

    $bundleRoot = Join-Path $KernelPath "bundle"
    if (Test-Path $bundleRoot) {
        Remove-Item -LiteralPath $bundleRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $bundleRoot | Out-Null

    $files = @(
        @{ Source = "model\data\preprocessing\dataset_loader.py"; Target = "model\data\preprocessing\dataset_loader.py" }
        @{ Source = "model\data\manifests\orchid_kaggle.json"; Target = "model\data\manifests\orchid_kaggle.json" }
        @{ Source = "model\training\stage2_grading\train.py"; Target = "model\training\stage2_grading\train.py" }
        @{ Source = "model\training\stage2_grading\config.kaggle.yaml"; Target = "model\training\stage2_grading\config.kaggle.yaml" }
    )

    foreach ($file in $files) {
        $sourcePath = Join-Path $RepoWin $file.Source
        $targetPath = Join-Path $bundleRoot $file.Target
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetPath) | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
    }
}

function Get-KernelStatusValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Kernel
    )

    $output = & $KagglePython -m kaggle kernels status $Kernel 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle status check failed for $Kernel`n$output"
    }

    $text = ($output | Out-String)
    if ($text -match 'KernelWorkerStatus\.([A-Z]+)') {
        return $Matches[1]
    }
    throw "Unable to parse Kaggle kernel status for $Kernel`n$text"
}

function Wait-ForKernelCompletion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Kernel,
        [Parameter(Mandatory = $true)]
        [int]$PollSeconds,
        [Parameter(Mandatory = $true)]
        [int]$TimeoutMinutes
    )

    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while ((Get-Date) -lt $deadline) {
        $status = Get-KernelStatusValue -Kernel $Kernel
        Write-Host "[WAIT] $Kernel => $status"
        if ($status -eq "COMPLETE") {
            return
        }
        if ($status -eq "ERROR") {
            Write-Host "[WAIT] Preflight failed. Fetching logs..."
            & $KagglePython -m kaggle kernels logs $Kernel
            throw "Kaggle kernel failed: $Kernel"
        }
        Start-Sleep -Seconds $PollSeconds
    }

    throw "Timed out waiting for Kaggle kernel completion: $Kernel"
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

Write-Host "[3/4] Preparing bundled code for Kaggle kernels..."
Prepare-KaggleBundle -KernelPath $KernelDir
Prepare-KaggleBundle -KernelPath $PreflightKernelDir

if (-not $SkipPreflight) {
    Write-Host "[4/4] Pushing Kaggle Stage 2 preflight kernel..."
    & $KagglePython -m kaggle kernels push -p $PreflightKernelDir --accelerator gpu -t 3600
    if ($LASTEXITCODE -ne 0) {
        throw "Kaggle preflight kernel push failed."
    }

    Wait-ForKernelCompletion -Kernel $PreflightKernelSlug -PollSeconds $PreflightPollSeconds -TimeoutMinutes $PreflightTimeoutMinutes
    Write-Host "[PRECHECK] Kaggle preflight completed successfully."
} else {
    Write-Host "[PRECHECK] Skipping preflight as requested."
}

Write-Host "[FINAL] Pushing Kaggle Stage 2 training kernel..."
& $KagglePython -m kaggle kernels push -p $KernelDir --accelerator gpu -t $TimeoutSeconds
if ($LASTEXITCODE -ne 0) {
    throw "Kaggle training kernel push failed."
}

Write-Host "[DONE] Kaggle Stage 2 training submitted."
Write-Host "Check status:"
Write-Host "  .\scripts\kaggle_stage2_status.ps1"
Remove-Item Env:\KAGGLE_API_TOKEN -ErrorAction SilentlyContinue
