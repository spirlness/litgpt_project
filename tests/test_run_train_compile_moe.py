import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestRunTrainCompileMoE(unittest.TestCase):
    def setUp(self):
        # Create a dictionary of mocked modules
        self.mock_modules = {
            "lightning": MagicMock(),
            "litgpt": MagicMock(),
            "litgpt.config": MagicMock(),
            "litgpt.model": MagicMock(),
            "litgpt.tokenizer": MagicMock(),
            "torch_xla": MagicMock(),
            "torch_xla.core": MagicMock(),
            "torch_xla.core.xla_model": MagicMock(),
            "src": MagicMock(),
            "src.fixed_text_files": MagicMock(),
            "src.utils": MagicMock(),
        }

        # Start patching sys.modules
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        # Import run_train inside the patched environment
        if "run_train" in sys.modules:
            importlib.reload(sys.modules["run_train"])
        else:
            importlib.import_module("run_train")
        self.run_train = sys.modules["run_train"]

    def tearDown(self):
        self.patcher.stop()
        # Clean up run_train from sys.modules to avoid side effects
        if "run_train" in sys.modules:
            del sys.modules["run_train"]

    @patch("run_train.load_yaml")
    @patch("run_train.L.Fabric")
    @patch("run_train.torch.compile")
    @patch("run_train.GPT")
    @patch("run_train.Tokenizer")
    @patch("run_train.FixedTextFiles")
    @patch("run_train.build_optimizer")
    @patch("run_train.Path")
    @patch("run_train.F.cross_entropy")
    def test_moe_compile_disabled(self, mock_cross_entropy, mock_path, mock_build_opt, mock_data, mock_tok, mock_gpt, mock_compile, mock_fabric, mock_load_yaml):
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
            self.run_train.train(args.model_config, args.train_config, args)
        except StopIteration:
            pass # Iterator exhausted
        except Exception as e:
            # If unexpected error, let it raise to debug
            raise e

        # Assert compile was NOT called (this is the desired behavior for MoE)
        # Currently this assertion should FAIL because the fix is not implemented
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
            self.run_train.train(args.model_config, args.train_config, args)
        except StopIteration:
            pass

        # Assert compile WAS called (this is standard behavior)
        mock_compile.assert_called_once()

if __name__ == "__main__":
    unittest.main()
