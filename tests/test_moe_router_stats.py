from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
VENV_SITE = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
LITGPT_ROOT = VENV_SITE / "litgpt"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if "litgpt" not in sys.modules:
    package = types.ModuleType("litgpt")
    package.__path__ = [str(LITGPT_ROOT)]
    sys.modules["litgpt"] = package

if "litgpt.scripts" not in sys.modules:
    scripts_pkg = types.ModuleType("litgpt.scripts")
    scripts_pkg.__path__ = [str(LITGPT_ROOT / "scripts")]
    sys.modules["litgpt.scripts"] = scripts_pkg

config_module = _load_module("litgpt.config", LITGPT_ROOT / "config.py")
_load_module("litgpt.scripts.convert_hf_checkpoint", LITGPT_ROOT / "scripts" / "convert_hf_checkpoint.py")
model_module = _load_module("litgpt.model", LITGPT_ROOT / "model.py")
setattr(sys.modules["litgpt"], "model", model_module)

Config = config_module.Config
GPT = model_module.GPT
LLaMAMoE = model_module.LLaMAMoE


def _make_moe_config(**overrides) -> Config:
    base = {
        "name": "MoE-Router-Stats-Test",
        "bias": False,
        "block_size": 16,
        "mlp_class_name": "LLaMAMoE",
        "moe_intermediate_size": 64,
        "n_embd": 32,
        "n_expert": 4,
        "n_expert_per_token": 2,
        "n_head": 4,
        "n_layer": 2,
        "n_query_groups": 2,
        "norm_class_name": "RMSNorm",
        "norm_eps": 1e-5,
        "padded_vocab_size": 64,
        "parallel_residual": False,
        "rope_base": 10000,
        "vocab_size": 64,
    }
    merged = {**base, **overrides}
    return Config(**merged)


def _get_moe_module(model: GPT) -> LLaMAMoE:
    for module in model.modules():
        if isinstance(module, LLaMAMoE):
            return module
    raise AssertionError("Expected LLaMAMoE module not found")


def test_router_stats_enabled() -> None:
    config = _make_moe_config(moe_router_stats=True, moe_aux_loss_weight=0.01)
    model = GPT(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)

    assert logits.shape == (2, 8, config.padded_vocab_size)

    moe_module = _get_moe_module(model)
    stats = moe_module.router_stats

    assert stats is not None
    aux_loss = stats["aux_loss"]
    assert aux_loss.ndim == 0
    assert aux_loss.item() >= 0

    load = stats["load"]
    assert load.sum().item() == input_ids.numel() * config.n_expert_per_token

    importance = stats["importance"]
    assert importance.sum().item() > 0
    assert importance.shape[0] == config.n_expert


def test_router_stats_disabled() -> None:
    config = _make_moe_config(moe_router_stats=False, moe_aux_loss_weight=0.0)
    model = GPT(config)

    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    _ = model(input_ids)

    moe_module = _get_moe_module(model)
    assert moe_module.router_stats is None
