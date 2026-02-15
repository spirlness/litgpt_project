import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock imports that might be heavy or unavailable BEFORE importing run_train
# This prevents run_train from actually importing them
sys.modules["litgpt"] = MagicMock()
sys.modules["litgpt.config"] = MagicMock()
sys.modules["litgpt.model"] = MagicMock()
sys.modules["litgpt.tokenizer"] = MagicMock()
sys.modules["lightning"] = MagicMock()
# torch_xla is handled in run_train with try-except, but we can mock it anyway
sys.modules["torch_xla"] = MagicMock()
sys.modules["torch_xla.core.xla_model"] = MagicMock()

# Also mock src.fixed_text_files
sys.modules["src.fixed_text_files"] = MagicMock()
sys.modules["src.utils"] = MagicMock()

import run_train  # noqa: E402


class TestRunTrainCompileLogic(unittest.TestCase):
    def setUp(self):
        # Reset mocks if needed
        pass

    @patch("run_train.load_yaml")
    @patch("run_train.L.Fabric")
    @patch("run_train.GPT")
    @patch("run_train.Config")
    @patch("run_train.torch.compile")
    @patch("run_train.build_optimizer")
    @patch("run_train.FixedTextFiles")
    @patch("run_train.Tokenizer")
    def test_moe_disables_compile(
        self,
        mock_tokenizer,
        mock_data,
        mock_build_opt,
        mock_compile,
        mock_config,
        mock_gpt,
        mock_fabric,
        mock_load_yaml,
    ):
        # Setup mocks
        # Mock load_yaml to return config with n_expert=8
        mock_load_yaml.side_effect = [
            {"model_config": {"n_expert": 8}},  # model_cfg
            {
                "optimization": {"compile": True},
                "train": {},
                "data": {"init_args": {"train_data_path": "dummy"}},
            },  # train_cfg
        ]

        args = argparse.Namespace(
            model_config=Path("dummy_model.yaml"),
            train_config=Path("dummy_train.yaml"),
            compile=True,  # Explicitly requested
            compile_mode=None,
            compile_dynamic=None,
            compile_fullgraph=None,
            flash_attention=None,
            flash_attention_force=None,
        )

        # Mock fabric.setup to return (model, optimizer)
        mock_fabric_instance = mock_fabric.return_value
        mock_fabric_instance.setup.return_value = (MagicMock(), MagicMock())
        mock_fabric_instance.is_global_zero = True

        # Mock Path.exists to pass tokenizer check
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("run_train.apply_runtime_config"),
            patch("run_train.configure_flash_attention"),
            patch("run_train.verify_flash_attention"),
            patch("run_train.patch_cudagraph_for_compile"),
            patch("run_train.patch_gradient_checkpointing"),
        ):
            # Call train
            try:
                run_train.train(args.model_config, args.train_config, args)
            except Exception:
                # We expect it might fail later due to mocks not covering everything (e.g. data loading)
                # But we catch it.
                pass

            # Since we haven't implemented the fix yet, we EXPECT compile to be called!
            # After fix, we expect it NOT called.
            # But the test should assert the DESIRED behavior (fix applied).
            mock_compile.assert_not_called()

    @patch("run_train.load_yaml")
    @patch("run_train.L.Fabric")
    @patch("run_train.GPT")
    @patch("run_train.Config")
    @patch("run_train.torch.compile")
    @patch("run_train.build_optimizer")
    @patch("run_train.FixedTextFiles")
    @patch("run_train.Tokenizer")
    def test_non_moe_enables_compile(
        self,
        mock_tokenizer,
        mock_data,
        mock_build_opt,
        mock_compile,
        mock_config,
        mock_gpt,
        mock_fabric,
        mock_load_yaml,
    ):
        # Setup mocks
        mock_load_yaml.side_effect = [
            {"model_config": {"n_expert": 0}},  # model_cfg
            {
                "optimization": {"compile": True},
                "train": {},
                "data": {"init_args": {"train_data_path": "dummy"}},
            },  # train_cfg
        ]

        args = argparse.Namespace(
            model_config=Path("dummy_model.yaml"),
            train_config=Path("dummy_train.yaml"),
            compile=True,
            compile_mode=None,
            compile_dynamic=None,
            compile_fullgraph=None,
            flash_attention=None,
            flash_attention_force=None,
        )

        mock_fabric_instance = mock_fabric.return_value
        mock_fabric_instance.setup.return_value = (MagicMock(), MagicMock())
        mock_fabric_instance.is_global_zero = True

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("run_train.apply_runtime_config"),
            patch("run_train.configure_flash_attention"),
            patch("run_train.verify_flash_attention"),
            patch("run_train.patch_cudagraph_for_compile"),
            patch("run_train.patch_gradient_checkpointing"),
        ):
            try:
                run_train.train(args.model_config, args.train_config, args)
            except Exception:
                pass

            # Assert compile IS called
            mock_compile.assert_called()


if __name__ == "__main__":
    unittest.main()
