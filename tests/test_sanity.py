import sys
import os
import pytest

# Add project root to sys.path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Simple sanity check to ensure modules can be imported."""
    try:
        import custom_moe
        import prepare_data
        # run_train has side effects (patches torch.compile, env vars), but importing it should be fine
        # if dependencies are met.
        import run_train
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")
    except Exception as e:
        pytest.fail(f"Module import raised exception: {e}")

def test_custom_moe_classes():
    """Verify custom_moe classes exist."""
    import custom_moe
    assert hasattr(custom_moe, 'FixedLLaMAMoE')
    assert hasattr(custom_moe, 'SimplifiedLLaMAMoE')
    assert hasattr(custom_moe, 'BatchedMoE')
