"""Smoke tests for learning-repo layout (no GPU, no upstream required)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "learning",
    "exercises",
    "extensions",
    "tests",
    "configs/learning",
    "runs",
    "scripts/lab",
    "upstream",
]

REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    "learning/00_project_map.md",
    "learning/experiment_ledger.md",
    "learning/improvement_backlog.md",
    "scripts/lab/count_params.py",
]


def test_required_dirs_exist():
    missing = [d for d in REQUIRED_DIRS if not (ROOT / d).is_dir()]
    assert not missing, f"missing dirs: {missing}"


def test_required_files_exist():
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
    assert not missing, f"missing files: {missing}"


def test_gitignore_excludes_weights():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for token in ("*.pth", "*.pt", "runs/"):
        assert token in text
