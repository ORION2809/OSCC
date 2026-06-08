# OralPath Notebooks

Use these notebooks through the Google Colab VS Code extension.

Recommended order:

1. `oralpath_colab_bootstrap.ipynb` - Connect to Colab, mount Drive, clone/update repo, install dependencies, verify GPU.
2. Run `python scripts/colab_runtime_check.py` inside the Colab runtime.
3. Run `python model/training/stage1_detection/train.py --config model/training/stage1_detection/config.yaml --max-batches 2` as a smoke test.
4. Run the same Stage 1 command without `--max-batches` for a real experiment.
5. Stage 2 experiment notebook - Add after ORCHID integration is stable.
6. Export notebook - Add after the selected model checkpoint is ready.

Keep notebooks as launchers. Put reusable code in `model/` scripts.

Do not use deleted one-shot/upload helpers or Playwright browser automation for Colab. The VS Code Colab extension should own login and runtime assignment.
