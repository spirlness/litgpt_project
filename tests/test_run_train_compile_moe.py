import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock modules before import to handle environments without these packages
sys.modules["lightning"] = MagicMock()
sys.modules["litgpt"] = MagicMock()
sys.modules["litgpt.config"] = MagicMock()
sys.modules["litgpt.model"] = MagicMock()
sys.modules["litgpt.tokenizer"] = MagicMock()
sys.modules["torch_xla"] = MagicMock()
sys.modules["torch_xla.core"] = MagicMock()
sys.modules["torch_xla.core.xla_model"] = MagicMock()
sys.modules["yaml"] = MagicMock()

# Mock torch if not available
try:
    import torch
except ImportError:
    torch = MagicMock()
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["torch.utils"] = MagicMock()
    sys.modules["torch.utils.checkpoint"] = MagicMock()

    # Basic mocks for torch functions used in tests/run_train
    torch.tensor.side_effect = lambda x, *args, **kwargs: MagicMock()
    torch.device.return_value = MagicMock()

    mock_tensor = MagicMock()
    mock_tensor.numel.return_value = 1024
    mock_tensor.__getitem__.return_value = mock_tensor
    mock_tensor.contiguous.return_value = mock_tensor
    mock_tensor.view.return_value = mock_tensor

    torch.randint.return_value = mock_tensor
    torch.zeros.return_value = MagicMock()
    torch.stack.return_value = MagicMock()

# Mock src modules
sys.modules["src"] = MagicMock()
sys.modules["src.fixed_text_files"] = MagicMock()
sys.modules["src.utils"] = MagicMock()

# Now import run_train
import run_train  # noqa: E402


class TestRunTrainCompileMoE(unittest.TestCase):
    @patch("run_train.load_yaml")
    @patch("run_train.L.Fabric")
    @patch("run_train.torch.compile")
    @patch("run_train.GPT")
    @patch("run_train.Tokenizer")
    @patch("run_train.FixedTextFiles")
    @patch("run_train.build_optimizer")
    @patch("run_train.Path")
    @patch("run_train.F.cross_entropy")
    def test_moe_compile_enabled_no_cudagraph(self, mock_cross_entropy, mock_path, mock_build_opt, mock_data, mock_tok, mock_gpt, mock_compile, mock_fabric, mock_load_yaml):
        # Setup mocks
        mock_gpt.return_value.config.moe_aux_loss_weight = 0.0
        mock_gpt.return_value.router_stats = {}

        # Make compile return the model itself (identity)
        mock_compile.side_effect = lambda m, *args, **kwargs: m

        # Mock fabric.all_reduce return value
        mock_fabric.return_value.all_reduce.return_value.item.return_value = 0.0

        # Mock optimizer param groups
        mock_build_opt.return_value.param_groups = [{"lr": 0.001}]

        mock_cross_entropy.return_value = torch.tensor(0.0)
        mock_path.return_value.exists.return_value = True

        # First call for model_config, second for train_config
        mock_load_yaml.side_effect = [
            {"model_config": {"n_expert": 8, "block_size": 1024}}, # MoE model
            {"optimization": {"compile": True}, "train": {"max_tokens": 10}, "data": {"init_args": {"train_data_path": "dummy"}}} # Compile requested
        ]

        # Mock fabric instance
        fabric_instance = mock_fabric.return_value
        fabric_instance.global_rank = 0
        fabric_instance.world_size = 1
        fabric_instance.device = torch.device("cpu")
        fabric_instance.setup.side_effect = lambda m, o: (m, o)
        fabric_instance.setup_dataloaders.return_value = MagicMock() # dataloader

        # Mock data loader iterator
        mock_dataloader = fabric_instance.setup_dataloaders.return_value
        # Mock batch: [input_ids, targets]
        mock_batch = torch.randint(0, 100, (1, 1025))
        mock_dataloader.__iter__.return_value = iter([mock_batch])

        # Mock args
        args = MagicMock()
        args.compile = None # Allow config to control it
        args.model_config = "model_config.yaml"
        args.train_config = "train_config.yaml"
        args.compile_mode = None
        args.compile_dynamic = None
        args.compile_fullgraph = None
        args.flash_attention = None
        args.flash_attention_force = None

        # Run train
        try:
            run_train.train(args.model_config, args.train_config, args)
        except StopIteration:
            pass # Iterator exhausted
        except Exception as e:
            # If unexpected error, let it raise to debug
            raise e

        # Assert compile was NOT called (this is the desired behavior for MoE)
        # Verified: the fix in run_train.py correctly disables compile for n_expert > 0
        mock_compile.assert_not_called()

    @patch("run_train.load_yaml")
    @patch("run_train.L.Fabric")
    @patch("run_train.torch.compile")
    @patch("run_train.GPT")
    @patch("run_train.Tokenizer")
    @patch("run_train.FixedTextFiles")
    @patch("run_train.build_optimizer")
    @patch("run_train.Path")
    @patch("run_train.F.cross_entropy")
    def test_non_moe_compile_enabled(self, mock_cross_entropy, mock_path, mock_build_opt, mock_data, mock_tok, mock_gpt, mock_compile, mock_fabric, mock_load_yaml):
        # Setup mocks for non-MoE
        mock_gpt.return_value.config.moe_aux_loss_weight = 0.0
        mock_gpt.return_value.router_stats = {}

        # Make compile return the model itself
        mock_compile.side_effect = lambda m, *args, **kwargs: m

        mock_fabric.return_value.all_reduce.return_value.item.return_value = 0.0
        mock_build_opt.return_value.param_groups = [{"lr": 0.001}]

        mock_cross_entropy.return_value = torch.tensor(0.0)
        mock_path.return_value.exists.return_value = True

        mock_load_yaml.side_effect = [
            {"model_config": {"n_expert": 0, "block_size": 1024}}, # Non-MoE
            {"optimization": {"compile": True}, "train": {"max_tokens": 10}, "data": {"init_args": {"train_data_path": "dummy"}}}
        ]

        fabric_instance = mock_fabric.return_value
        fabric_instance.global_rank = 0
        fabric_instance.world_size = 1
        fabric_instance.device = torch.device("cpu")
        fabric_instance.setup.side_effect = lambda m, o: (m, o)
        fabric_instance.setup_dataloaders.return_value = MagicMock()
        mock_dataloader = fabric_instance.setup_dataloaders.return_value
        mock_batch = torch.randint(0, 100, (1, 1025))
        mock_dataloader.__iter__.return_value = iter([mock_batch])

        args = MagicMock()
        args.compile = None
        args.model_config = "model_config.yaml"
        args.train_config = "train_config.yaml"
        args.compile_mode = None
        args.compile_dynamic = None
        args.compile_fullgraph = None
        args.flash_attention = None
        args.flash_attention_force = None

        try:
            run_train.train(args.model_config, args.train_config, args)
        except StopIteration:
            pass

        # Assert compile WAS called (this is standard behavior)
        mock_compile.assert_called_once()
        # Assert patch_cudagraph_for_compile WAS called for non-MoE
        sys.modules["src.utils"].patch_cudagraph_for_compile.assert_called_once()

if __name__ == "__main__":
    unittest.main()
