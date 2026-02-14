import sys
import unittest
import importlib
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add root to path so we can import run_train
sys.path.append(str(Path(__file__).parent.parent))

class TestRunTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a patcher for sys.modules
        cls.modules_patcher = patch.dict(sys.modules, {
            "lightning": MagicMock(),
            "lightning.Fabric": MagicMock(),
            "litgpt": MagicMock(),
            "litgpt.config": MagicMock(),
            "litgpt.model": MagicMock(),
            "litgpt.tokenizer": MagicMock(),
            "src": MagicMock(),
            "src.fixed_text_files": MagicMock(),
            "src.utils": MagicMock(),
        })
        cls.modules_patcher.start()

        # Ensure run_train is imported with these mocks
        if "run_train" in sys.modules:
            del sys.modules["run_train"]
        import run_train
        cls.run_train = run_train

    @classmethod
    def tearDownClass(cls):
        # Stop patching
        cls.modules_patcher.stop()
        if "run_train" in sys.modules:
            del sys.modules["run_train"]

    def setUp(self):
        # Reset mocks that are properties of the module (imported mocks)
        # run_train.L is a Mock
        self.run_train.L.Fabric.reset_mock()
        # run_train.GPT is a Mock
        self.run_train.GPT.reset_mock()
        # run_train.build_optimizer defines logic but uses mocked imports
        # Wait, run_train.build_optimizer is a function defined in run_train.py
        # So we should patch it if we want to mock it.
        pass

    def test_compile_disabled_for_moe(self):
        """Test that torch.compile is disabled for MoE models even if requested."""
        mock_fabric = self.run_train.L.Fabric
        mock_gpt = self.run_train.GPT

        # We patch local functions: load_yaml, build_optimizer
        # We also patch torch.compile
        with patch("run_train.load_yaml") as mock_load_yaml, \
             patch("run_train.build_optimizer") as mock_opt, \
             patch("run_train.torch.compile") as mock_compile, \
             patch("pathlib.Path.exists", return_value=True) as mock_exists, \
             patch("run_train.patch_cudagraph_for_compile") as mock_patch_cuda:

            # Scenario: MoE model (n_expert=8) and compile=True requested
            model_cfg = {"model_config": {"n_expert": 8, "block_size": 128}}
            train_cfg = {"train": {}, "data": {}, "optimizer": {}}

            mock_load_yaml.side_effect = [model_cfg, train_cfg]

            args = MagicMock()
            args.compile = True
            args.compile_mode = "default"
            args.compile_dynamic = False
            args.compile_fullgraph = False
            args.flash_attention = False
            args.flash_attention_force = False

            fabric_instance = mock_fabric.return_value
            fabric_instance.is_global_zero = True
            fabric_instance.world_size = 1
            fabric_instance.device = MagicMock()
            fabric_instance.setup.return_value = (mock_gpt.return_value, mock_opt.return_value)

            try:
                self.run_train.train(Path("model.yaml"), Path("train.yaml"), args)
            except ValueError:
                pass
            except Exception:
                import traceback
                traceback.print_exc()
                pass

            # Assert compile was NOT called
            mock_compile.assert_not_called()
            mock_patch_cuda.assert_not_called()

    def test_compile_enabled_for_dense(self):
        """Test that torch.compile is enabled for dense models if requested."""
        mock_fabric = self.run_train.L.Fabric
        mock_gpt = self.run_train.GPT

        with patch("run_train.load_yaml") as mock_load_yaml, \
             patch("run_train.build_optimizer") as mock_opt, \
             patch("run_train.torch.compile") as mock_compile, \
             patch("pathlib.Path.exists", return_value=True) as mock_exists, \
             patch("run_train.patch_cudagraph_for_compile") as mock_patch_cuda:

            model_cfg = {"model_config": {"block_size": 128}}
            train_cfg = {"train": {}, "data": {}, "optimizer": {}}

            mock_load_yaml.side_effect = [model_cfg, train_cfg]

            args = MagicMock()
            args.compile = True
            args.compile_mode = "default"
            args.compile_dynamic = False
            args.compile_fullgraph = False
            args.flash_attention = False
            args.flash_attention_force = False

            fabric_instance = mock_fabric.return_value
            fabric_instance.is_global_zero = True
            fabric_instance.world_size = 1
            fabric_instance.setup.return_value = (mock_gpt.return_value, mock_opt.return_value)

            try:
                self.run_train.train(Path("model.yaml"), Path("train.yaml"), args)
            except ValueError:
                pass
            except Exception:
                import traceback
                traceback.print_exc()
                pass

            # Assert compile WAS called
            mock_compile.assert_called()
            mock_patch_cuda.assert_called()

if __name__ == "__main__":
    unittest.main()
