import concurrent.futures
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestRunTrainUpload(unittest.TestCase):
    def setUp(self):
        # Setup mocks dictionary
        self.mock_modules = {
            "lightning": MagicMock(),
            "litgpt": MagicMock(),
            "litgpt.config": MagicMock(),
            "litgpt.model": MagicMock(),
            "litgpt.tokenizer": MagicMock(),
            "torch_xla": MagicMock(),
            "torch_xla.core": MagicMock(),
            "torch_xla.core.xla_model": MagicMock(),
            "yaml": MagicMock(),
            "src": MagicMock(),
            "src.litgpt_moe": MagicMock(),
            "src.litgpt_moe.config": MagicMock(),
            "src.litgpt_moe.fixed_text_files": MagicMock(),
            "src.litgpt_moe.utils": MagicMock(),
        }

        # We mock torch consistently
        mock_torch = MagicMock()
        mock_torch.__path__ = []
        mock_torch.__spec__ = None

        class MockTensor(MagicMock):
            def numel(self):
                return 1024
            def __getitem__(self, idx):
                return self
            def contiguous(self):
                return self
            def view(self, *args):
                return self
            def detach(self):
                return self
            def to(self, *args, **kwargs):
                return self
            def __ge__(self, other):
                return True
            def __gt__(self, other):
                return True
            def __le__(self, other):
                return True
            def __lt__(self, other):
                return True

        mock_torch.Tensor = MockTensor
        mock_torch.randint.return_value = MockTensor()
        mock_torch.zeros.return_value = MockTensor()
        mock_torch.stack.return_value = MockTensor()

        self.mock_modules["torch"] = mock_torch
        self.mock_modules["torch.nn"] = MagicMock()
        self.mock_modules["torch.nn.functional"] = MagicMock()
        self.mock_modules["torch.utils"] = MagicMock()
        self.mock_modules["torch.utils.checkpoint"] = MagicMock()

        # Start patcher
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        # Add current directory to path
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())

        # Import run_train
        import importlib

        import run_train
        importlib.reload(run_train)
        self.run_train = run_train

    def tearDown(self):
        self.patcher.stop()
        if "run_train" in sys.modules:
            del sys.modules["run_train"]

    def test_executor_is_process_pool(self):
        executor = self.run_train._get_upload_executor()
        self.assertIsInstance(executor, concurrent.futures.ProcessPoolExecutor)

    def test_save_checkpoint_submits_to_executor(self):
        mock_fabric = MagicMock()
        mock_fabric.is_global_zero = True

        # Mock executor inside run_train
        mock_executor = MagicMock()

        # We need to mock _get_upload_executor
        with patch.object(self.run_train, "_get_upload_executor", return_value=mock_executor):
            self.run_train.save_checkpoint(
                fabric=mock_fabric,
                out_dir=MagicMock(),
                step=10,
                total_tokens=1000,
                model=MagicMock(),
                optimizer=MagicMock(),
                upload_to_hf=True,
                hf_repo_id="test/repo"
            )
            mock_executor.submit.assert_called_once()
            # Verify the call arguments
            args, _ = mock_executor.submit.call_args
            self.assertEqual(args[0], self.run_train._upload_and_cleanup)

if __name__ == "__main__":
    unittest.main()
