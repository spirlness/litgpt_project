from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Mock dependencies to avoid importing them if they are missing or heavy
# But since we installed them, we can import run_train directly,
# but we need to mock internal calls to avoid running real training.

import run_train

@patch("run_train.load_yaml")
@patch("run_train.L.Fabric")
@patch("run_train.torch.compile")
@patch("run_train.FixedTextFiles")
@patch("run_train.Tokenizer")
@patch("run_train.GPT")
@patch("run_train.build_optimizer")
@patch("run_train.patch_cudagraph_for_compile")
@patch("pathlib.Path.exists")
def test_compile_disable_for_moe(
    mock_path_exists,
    mock_patch_cudagraph,
    mock_build_opt,
    mock_gpt,
    mock_tokenizer,
    mock_fixed_text_files,
    mock_compile,
    mock_fabric_cls,
    mock_load_yaml,
):
    # Ensure Path.exists returns True so tokenizer check passes
    mock_path_exists.return_value = True

    # Setup mocks
    mock_fabric_instance = MagicMock()
    mock_fabric_cls.return_value = mock_fabric_instance
    mock_fabric_instance.is_global_zero = True
    mock_fabric_instance.world_size = 1
    mock_fabric_instance.device = torch.device("cpu")

    # Mock data loader to return empty iterator to stop training loop immediately
    mock_data = MagicMock()
    mock_data.train_dataloader.return_value = []
    mock_fixed_text_files.return_value = mock_data

    # Mock tokenizer
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer.return_value = mock_tokenizer_instance

    # Mock GPT model
    mock_model = MagicMock()
    mock_gpt.return_value = mock_model
    mock_fabric_instance.setup.return_value = (mock_model, MagicMock())

    # Case 1: MoE model (n_expert=8) and compile=True requested
    # We expect compile to be DISABLED
    mock_load_yaml.side_effect = [
        {"n_expert": 8}, # model_cfg
        {"optimization": {"compile": True}, "train": {}, "data": {"init_args": {"train_data_path": "foo"}}} # train_cfg
    ]

    args = MagicMock()
    args.compile = None
    args.model_config = Path("model_config.yaml")
    args.train_config = Path("train_config.yaml")
    args.compile_mode = None
    args.compile_dynamic = None
    args.compile_fullgraph = None
    args.flash_attention = None
    args.flash_attention_force = None

    # Run train
    try:
        run_train.train(args.model_config, args.train_config, args)
    except Exception as e:
        print(f"Exception during training: {e}")
        # Ignore errors after compile check (e.g. data loading)
        pass

    # Verify torch.compile was NOT called
    mock_compile.assert_not_called()
    mock_patch_cudagraph.assert_not_called()
    print("Test passed: torch.compile was disabled for MoE model.")

    # Reset mocks
    mock_compile.reset_mock()
    mock_patch_cudagraph.reset_mock()

    # Case 2: Non-MoE model (n_expert=0) and compile=True requested
    # We expect compile to be CALLED
    mock_load_yaml.side_effect = [
        {"n_expert": 0, "name": "GPT-Small"}, # model_cfg
        {"optimization": {"compile": True}, "train": {}, "data": {"init_args": {"train_data_path": "foo"}}} # train_cfg
    ]

    try:
        run_train.train(args.model_config, args.train_config, args)
    except Exception:
        pass

    # Verify torch.compile WAS called
    mock_compile.assert_called()
    mock_patch_cudagraph.assert_called()
    print("Test passed: torch.compile was enabled for non-MoE model.")

if __name__ == "__main__":
    # Manually run the test function if executed directly
    # Need to handle the patch decorators
    try:
        # This is hacky to run decorated function without pytest
        # We'll just rely on pytest invocation in the plan
        pass
    except Exception as e:
        print(f"Error: {e}")
