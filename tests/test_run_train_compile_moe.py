import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure run_train can be imported
sys.path.append(os.getcwd())

class TestRunTrainCompileMoE(unittest.TestCase):

    @patch("run_train.load_yaml")
    @patch("run_train.L.Fabric")
    @patch("run_train.torch.compile")
    @patch("run_train.GPT")
    @patch("run_train.Tokenizer")
    @patch("run_train.FixedTextFiles")
    @patch("run_train.build_optimizer")
    @patch("run_train.Path") # Mock Path to avoid file system access
    def test_moe_compile_enabled(self, mock_path, mock_build_opt, mock_data, mock_tok, mock_gpt, mock_compile, mock_fabric, mock_load_yaml):
        import run_train

        # Setup mocks
        # Model config with MoE
        model_cfg = {"n_expert": 8, "block_size": 1024, "model_config": {"n_expert": 8, "block_size": 1024}}
        # Train config with compile enabled
        train_cfg = {
            "optimization": {"compile": True},
            "train": {"max_tokens": 10},
            "data": {"init_args": {"train_data_path": "dummy"}}
        }

        mock_load_yaml.side_effect = [model_cfg, train_cfg]

        # Mock Path existence
        mock_path.return_value.exists.return_value = True

        # Mock Fabric setup
        fabric_instance = mock_fabric.return_value
        fabric_instance.global_rank = 0
        fabric_instance.world_size = 1

        # Mock model returned by GPT
        mock_model = MagicMock()
        mock_gpt.return_value = mock_model

        # Mock model returned by fabric.setup
        # Case 1: Standard model (no _forward_module)
        wrapped_model = MagicMock()
        del wrapped_model._forward_module # Ensure it doesn't have it
        fabric_instance.setup.return_value = (wrapped_model, MagicMock())

        # Mock args
        args = MagicMock()
        args.compile = True
        args.model_config = "model_config.yaml"
        args.train_config = "train_config.yaml"
        args.compile_mode = "default"
        args.compile_dynamic = False
        args.compile_fullgraph = False
        args.flash_attention = None
        args.flash_attention_force = None

        # Run train (will crash later due to mocks but we check compile call)
        try:
            run_train.train(args.model_config, args.train_config, args)
        except Exception:
            pass

        # Assert compile WAS called on wrapped_model
        mock_compile.assert_called_with(wrapped_model, mode="default", dynamic=False, fullgraph=False)

    @patch("run_train.load_yaml")
    @patch("run_train.L.Fabric")
    @patch("run_train.torch.compile")
    @patch("run_train.GPT")
    @patch("run_train.Tokenizer")
    @patch("run_train.FixedTextFiles")
    @patch("run_train.build_optimizer")
    @patch("run_train.Path")
    def test_moe_compile_forward_module(self, mock_path, mock_build_opt, mock_data, mock_tok, mock_gpt, mock_compile, mock_fabric, mock_load_yaml):
        import run_train

        # Setup mocks
        model_cfg = {"n_expert": 8, "block_size": 1024, "model_config": {"n_expert": 8, "block_size": 1024}}
        train_cfg = {
            "optimization": {"compile": True},
            "train": {"max_tokens": 10},
            "data": {"init_args": {"train_data_path": "dummy"}}
        }

        mock_load_yaml.side_effect = [model_cfg, train_cfg]
        mock_path.return_value.exists.return_value = True

        fabric_instance = mock_fabric.return_value
        fabric_instance.global_rank = 0
        fabric_instance.world_size = 1

        mock_model = MagicMock()
        mock_gpt.return_value = mock_model

        # Mock model returned by fabric.setup WITH _forward_module
        wrapped_model = MagicMock()
        inner_module = MagicMock()
        wrapped_model._forward_module = inner_module
        fabric_instance.setup.return_value = (wrapped_model, MagicMock())

        # Setup compile return value
        compiled_inner = MagicMock()
        mock_compile.return_value = compiled_inner

        args = MagicMock()
        args.compile = True
        args.model_config = "model_config.yaml"
        args.train_config = "train_config.yaml"
        args.compile_mode = "default"
        args.compile_dynamic = False
        args.compile_fullgraph = False
        args.flash_attention = None
        args.flash_attention_force = None

        try:
            run_train.train(args.model_config, args.train_config, args)
        except Exception:
            pass

        # Assert compile WAS called on inner_module
        mock_compile.assert_called_with(inner_module, mode="default", dynamic=False, fullgraph=False)

        # Assert _forward_module was updated
        self.assertEqual(wrapped_model._forward_module, compiled_inner)

if __name__ == "__main__":
    unittest.main()
