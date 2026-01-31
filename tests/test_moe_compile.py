import sys
import os
import torch
import pytest
from litgpt.config import Config

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.custom_moe import FixedLLaMAMoE

def test_moe_compile():
    """Test that FixedLLaMAMoE can be compiled and run with torch.compile."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    # Use a small config for speed
    config = Config(
        name="MoE-200M",
        block_size=128,
        n_layer=2,
        n_embd=128,
        n_head=4,
        n_query_groups=2,
        mlp_class_name="LLaMAMoE",
        moe_intermediate_size=256,
        n_expert=4,
        n_expert_per_token=2,
        padded_vocab_size=1000,
        vocab_size=1000,
        bias=False,
        parallel_residual=False,
        rope_base=10000,
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
    )

    model = FixedLLaMAMoE(config)

    # Move to CUDA if available, but CPU is fine for this test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Input tensor
    B, T, C = 2, 64, config.n_embd
    x = torch.randn(B, T, C, device=device)

    # Eager run
    with torch.no_grad():
        y_eager = model(x)

    # Compile
    # Note: dynamic=True might be needed to avoid excessive graph breaks with nonzero,
    # but FixedLLaMAMoE should work even with default settings (with graph breaks).
    compiled_model = torch.compile(model)

    # Warmup
    with torch.no_grad():
        y_compiled = compiled_model(x)

    # Verification run
    with torch.no_grad():
        y_compiled = compiled_model(x)

    # Check match
    diff = (y_eager - y_compiled).abs().max()
    assert diff < 1e-4, f"Output mismatch: max diff {diff}"

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
