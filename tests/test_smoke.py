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
        Path("configs/moe_30m_debug.yaml"),
        Path("configs/moe_200m.yaml"),
        Path("configs/moe_400m.yaml"),
        Path("configs/kaggle_t4_ddp.yaml"),
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
        "src",
        "scripts",
    ]
    cmd = [sys.executable, "-m", "compileall", "-q", "-f", *targets]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout


def test_generate_has_single_streamer_definition() -> None:
    text = Path("generate.py").read_text(encoding="utf-8")
    assert text.count("class AsyncTokenStreamer") == 1


def test_import_entrypoints() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("litgpt")

    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import evaluate
    import generate
    import prepare_data
    import run_train
    from src.litgpt_moe import utils

    assert evaluate
    assert generate
    assert prepare_data
    assert run_train
    assert utils


@pytest.mark.skipif(CI, reason="CI runners are GPU-free and may not support bitsandbytes runtime")
def test_env_sanity_check_local() -> None:
    env = os.environ.copy()
    env.pop("SKIP_BNB_RUNTIME", None)

    sanity_script = Path("scripts/verify_flash.py")
    if not sanity_script.exists():
        pytest.skip("scripts/verify_flash.py not found")

    try:
        # Try running with uv first (assuming uv is in PATH)
        cmd = ["uv", "run", "python", str(sanity_script)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    except FileNotFoundError:
        # Fallback to sys.executable if uv is not found
        cmd = [sys.executable, str(sanity_script)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    # In sandboxed environments uv cache can be inaccessible; fallback to plain python.
    if result.returncode != 0 and "Failed to initialize cache" in (result.stderr or ""):
        cmd = [sys.executable, str(sanity_script)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    # Skip instead of failing when core runtime deps are unavailable in lightweight environments.
    if result.returncode != 0 and ("No module named 'torch'" in (result.stderr or "") or "ModuleNotFoundError: No module named 'torch'" in (result.stderr or "")):
        pytest.skip("torch is not installed in this environment")

    assert result.returncode == 0, result.stderr or result.stdout
