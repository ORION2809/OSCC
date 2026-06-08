import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _repo_text_files():
    excluded_parts = {
        ".venv",
        ".playwright-mcp",
        "__pycache__",
        ".git",
    }
    excluded_roots = {
        ROOT / "model" / "data" / "raw",
        ROOT / "model" / "data" / "processed",
        ROOT / "model" / "external",
    }

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in excluded_parts for part in path.parts):
            continue
        if any(path.is_relative_to(root) for root in excluded_roots):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".zip"}:
            continue
        yield path


def test_no_huggingface_tokens_are_committed():
    token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
    leaks = []
    for path in _repo_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if token_pattern.search(text):
            leaks.append(str(path.relative_to(ROOT)))

    assert leaks == []


def test_colab_runbook_rejects_playwright_control():
    setup_doc = (ROOT / "docs" / "COLAB_SETUP.md").read_text(encoding="utf-8")

    assert "Do not use Playwright" in setup_doc
    assert "not from Playwright and not from a terminal command" in setup_doc
    assert "Do not upload the whole project folder to Drive" in setup_doc
