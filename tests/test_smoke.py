"""Smoke tests: keep fast, deterministic, CPU-only."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

CI = os.environ.get("CI", "").lower() in {"1", "true", "yes"}


def test_configs_parse() -> None:
    paths = [
        Path("model_config.yaml"),
        Path("train_config.yaml"),
        Path("configs/moe_30m_debug.yaml"),
        Path("configs/moe_200m.yaml"),
        Path("configs/moe_400m.yaml"),
    ]
    for path in paths:
        assert path.exists(), f"Missing config: {path}"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None


def test_compileall_project_sources() -> None:
    targets = [
        "run_train.py",
        "prepare_data.py",
        "generate.py",
        "evaluate.py",
        "wandb_dataset.py",
        "src/custom_moe.py",
        "tools",
    ]
    cmd = [sys.executable, "-m", "compileall", "-q", "-f", *targets]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout


def test_generate_has_single_streamer_definition() -> None:
    text = Path("generate.py").read_text(encoding="utf-8")
    assert text.count("class AsyncTokenStreamer") == 1


def test_import_entrypoints() -> None:
    # Import-only smoke: should not require data/checkpoints.
    # Add project root to sys.path to allow imports of top-level scripts
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import run_train
    import prepare_data
    import generate
    import evaluate
    import wandb_dataset
    from src import custom_moe


@pytest.mark.skipif(CI, reason="CI runners are GPU-free and may not support bitsandbytes runtime")
def test_env_sanity_check_local() -> None:
    env = os.environ.copy()
    env.pop("SKIP_BNB_RUNTIME", None)

    # Check if we are running under uv or a venv
    # If using 'uv run pytest', sys.executable points to the venv python.
    # However, if just 'pytest' is called, it might use the global python.
    # We want to force using the 'python' that is in the current environment's path if possible,
    # OR explicitly use the one that launched this test process.

    # But wait, earlier logs showed sys.executable was G:\anaconda\python.exe when it failed,
    # which suggests pytest was running from the base conda env, NOT the uv venv.
    # To fix this, we should rely on 'uv run' to execute the script if possible, or use sys.executable.

    # If we are in a uv managed project, `uv run` handles the environment correctly.
    # Let's try to run via `uv run python ...` first.

    try:
        # Try running with uv first (assuming uv is in PATH)
        cmd = ["uv", "run", "python", "tools/env_sanity_check.py"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    except FileNotFoundError:
        # Fallback to sys.executable if uv is not found
        cmd = [sys.executable, "tools/env_sanity_check.py"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    assert result.returncode == 0, result.stderr or result.stdout
