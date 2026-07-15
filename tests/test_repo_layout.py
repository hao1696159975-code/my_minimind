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
    ".gitmodules",
    "learning/00_project_map.md",
    "learning/experiment_ledger.md",
    "learning/improvement_backlog.md",
    "scripts/lab/count_params.py",
]

# Present when submodule is initialized (CI/local after --recurse-submodules)
UPSTREAM_CORE_FILES = [
    "upstream/model/model_minimind.py",
    "upstream/model/model_lora.py",
    "upstream/dataset/lm_dataset.py",
    "upstream/trainer/train_pretrain.py",
    "upstream/eval_llm.py",
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


def test_gitmodules_points_to_official_minimind():
    text = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert 'path = upstream' in text
    assert "jingyaogong/minimind" in text


def test_upstream_core_files_if_submodule_checked_out():
    """Skip soft-fail only when submodule not initialized; fail if partially broken."""
    marker = ROOT / "upstream" / "model" / "model_minimind.py"
    gitkeep_only = (ROOT / "upstream" / ".gitkeep").is_file() and not marker.is_file()
    if gitkeep_only:
        # Old broken scaffold — should not happen after Phase0 fix commit
        raise AssertionError(
            "upstream/ only has .gitkeep; run: git submodule update --init --recursive"
        )
    # If upstream dir has any real content expectation, require core files
    if (ROOT / "upstream" / "README.md").is_file() or marker.is_file():
        missing = [f for f in UPSTREAM_CORE_FILES if not (ROOT / f).is_file()]
        assert not missing, f"submodule incomplete, missing: {missing}"
