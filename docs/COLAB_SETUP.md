# Colab + VS Code Setup

## Current Status

The official Google Colab VS Code extension is the right tool for training work in this project. It lets us edit notebooks in VS Code while running cells on Colab GPU/TPU runtimes.

Local VS Code already has these required extensions installed:

| Extension | Purpose |
|---|---|
| `google.colab` | Connect VS Code notebooks to Colab runtimes |
| `ms-toolsai.jupyter` | Notebook editing and execution support |
| `ms-python.python` | Python language support |
| `ms-python.vscode-pylance` | Python analysis and completions |

The repo also includes `.vscode/extensions.json` so VS Code will recommend the same setup on any new machine.

## How To Use It

1. Open this repo in VS Code.
2. Open `model/notebooks/oralpath_colab_bootstrap.ipynb`.
3. Use the notebook kernel/runtime picker and choose a Colab runtime.
4. Sign in with Google when prompted by the Colab extension.
5. Pick a GPU runtime for model training.
6. Run the notebook setup cells.

## Training Workflow

Use VS Code for editing and source control. Use Colab only for GPU execution.

Do not use Playwright or browser automation to control Colab from a CLI. The Colab VS Code extension assigns runtimes through VS Code's notebook kernel flow, so browser automation is brittle and usually fights the extension instead of helping.

The intended loop is:

1. Edit training scripts locally in `model/training/`.
2. Commit or push the repo when the code is ready for a GPU run.
3. Open a project notebook from `model/notebooks/`.
4. Connect it to a Colab runtime.
5. In the notebook, clone or update the repo inside `/content/oralpath`.
6. Install `requirements.txt`.
7. Run the training command.
8. Save metrics, checkpoints, and exported models to Google Drive.

Only datasets, checkpoints, run logs, and exported models should live in Google Drive. The source code should stay in git and be cloned into the Colab runtime.

## Runtime Paths

Inside Colab, use this convention:

| Path | Purpose |
|---|---|
| `/content/oralpath` | Repo checkout |
| `/content/drive/MyDrive/oralpath/data` | Datasets |
| `/content/drive/MyDrive/oralpath/runs` | Training outputs |
| `/content/drive/MyDrive/oralpath/checkpoints` | Model checkpoints |
| `/content/drive/MyDrive/oralpath/exports` | ONNX / TFLite exports |

Do not store datasets inside git.
Do not upload the whole project folder to Drive for every run.

## Login And Server Assignment

If the Colab extension shows "no assigned servers", that does not mean the setup failed. It means no notebook has been connected to a Colab runtime yet.

Use this flow:

1. Open `model/notebooks/oralpath_colab_bootstrap.ipynb`.
2. Click the notebook kernel picker in the top right.
3. Select a Colab runtime/kernel.
4. Complete Google sign-in when VS Code prompts you.
5. Choose GPU if the runtime picker asks for hardware.
6. Run the notebook cells.

The server assignment happens from the notebook kernel picker, not from Playwright and not from a terminal command.

## What Kimi / Codex Should And Should Not Do

Good automation:

- Edit `.py` training scripts.
- Edit `.ipynb` launcher notebooks.
- Update configs and manifests.
- Run local dry-run checks.
- Generate commands for the Colab notebook.

Bad automation:

- Trying to click through Google login with Playwright.
- Trying to control the Colab VS Code extension from a CLI.
- Hardcoding Hugging Face, Kaggle, or Google tokens in repo files.
- Uploading the full repo to Drive on every run.

## First Model Run

The first Colab run should target Stage 1 only:

```bash
python model/data/preprocessing/dataset_loader.py
python model/training/stage1_detection/train.py --config model/training/stage1_detection/config.yaml
```

Stage 2 and Stage 3 should wait until Stage 1 loading, metrics, and checkpoint handling are stable.

## Notes

- Keep notebooks thin. Training logic belongs in `.py` files under `model/training/`.
- Use notebooks for setup, GPU checks, experiment launch, and result inspection.
- Save every run output under Google Drive so Colab disconnects do not destroy results.
- The official extension is still notebook-centered, so the most reliable flow is to clone/update the repo inside the Colab runtime before each serious run.

## References

- Google Developers Blog: https://developers.googleblog.com/google-colab-is-coming-to-vscode/
- VS Code Marketplace: https://marketplace.visualstudio.com/items?itemName=google.colab
