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
        "custom_moe.py",
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
    __import__("run_train")
    __import__("prepare_data")
    __import__("generate")
    __import__("evaluate")
    __import__("wandb_dataset")
    __import__("custom_moe")


@pytest.mark.skipif(CI, reason="CI runners are GPU-free and may not support bitsandbytes runtime")
def test_env_sanity_check_local() -> None:
    env = os.environ.copy()
    env.pop("SKIP_BNB_RUNTIME", None)
    result = subprocess.run(
        [sys.executable, "tools/env_sanity_check.py"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
