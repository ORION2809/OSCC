# Colab CLI Setup

Google released an official Colab CLI in June 2026. It lets terminal agents provision Colab GPUs/TPUs and execute scripts remotely.

## Current Machine Status

Native Windows does not work because the official CLI imports Linux-only modules. The project is configured to run it through Ubuntu WSL instead.

Installed in Ubuntu WSL:

- `uv`
- `google-colab-cli`
- Google Cloud CLI

## One-Time Google Auth

Google OAuth consent cannot be automated. Run this once in PowerShell:

```powershell
.\scripts\auth_colab_cli_adc.ps1
```

It prints a Google login URL. Open it, approve access, and paste the returned code into the terminal. After that, the agent can run Colab CLI jobs through WSL.

## Stage 1 Smoke Job

After auth:

```powershell
.\scripts\run_colab_cli_stage1.ps1
```

This provisions a T4 session, uploads Kaggle credentials from `~/.kaggle/kaggle.json`, downloads Stage 1 data into Colab temporary disk, runs dry-run and one-batch smoke training, and downloads reports.

## Full Stage 1 Training

```powershell
.\scripts\run_colab_cli_stage1.ps1 -FullTraining
```

## Stop Session

```powershell
wsl -d Ubuntu -- bash -lc "source ~/.local/bin/env && colab --auth=adc stop -s oralpath-stage1"
```

## Notes

- Datasets are stored on Colab temporary `/content` disk, not Google Drive.
- The Kaggle API key has already appeared in chat; rotate it after testing.
- The CLI source and Google launch post confirm support for remote execution via `colab exec`, accelerator provisioning via `colab new --gpu T4`, artifact download via `colab download`, and agent workflows.
