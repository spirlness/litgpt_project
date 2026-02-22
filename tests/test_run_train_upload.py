
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import time

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

        # Mock torch
        mock_torch = MagicMock()
        mock_torch.__path__ = []
        mock_torch.__spec__ = None

        # Fix TypeError: isinstance() arg 2 must be a type
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

            # Allow comparison with int just in case
            def __ge__(self, other):
                return True
            def __gt__(self, other):
                return True
            def __le__(self, other):
                return True
            def __lt__(self, other):
                return True

        mock_torch.Tensor = MockTensor
        self.mock_modules["torch"] = mock_torch
        self.mock_modules["torch.nn"] = MagicMock()
        self.mock_modules["torch.nn.functional"] = MagicMock()
        self.mock_modules["torch.utils"] = MagicMock()
        self.mock_modules["torch.utils.checkpoint"] = MagicMock()

        # Start patcher
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        # Import run_train safely
        import importlib
        try:
            import run_train
            importlib.reload(run_train)
        except ImportError:
            pass

        self.run_train = run_train

    def tearDown(self):
        self.patcher.stop()
        if "run_train" in sys.modules:
            del sys.modules["run_train"]

    def test_save_checkpoint_is_async(self):
        """Test that save_checkpoint returns immediately even if upload is slow."""

        # Mock dependencies
        fabric = MagicMock()
        fabric.is_global_zero = True
        out_dir = Path("checkpoints")
        step = 1
        total_tokens = 100
        model = MagicMock()
        optimizer = MagicMock()

        # Mock HfApi inside run_train._upload_and_cleanup
        # Since _upload_and_cleanup imports locally, we need to patch sys.modules['huggingface_hub']

        mock_hf_hub = MagicMock()
        mock_api = MagicMock()
        mock_hf_hub.HfApi.return_value = mock_api

        # Simulate slow upload
        upload_called = False
        def slow_upload(*args, **kwargs):
            nonlocal upload_called
            time.sleep(0.5) # Simulate delay
            upload_called = True

        mock_api.upload_folder.side_effect = slow_upload

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
            # Ensure executor is initialized
            self.run_train._shutdown_upload_executor() # Reset if needed

            start_time = time.time()
            self.run_train.save_checkpoint(
                fabric,
                out_dir,
                step,
                total_tokens,
                model,
                optimizer,
                upload_to_hf=True,
                hf_repo_id="test/repo"
            )
            end_time = time.time()

            duration = end_time - start_time

            # Assert save_checkpoint returned quickly (< 0.1s)
            self.assertLess(duration, 0.1, f"save_checkpoint took too long: {duration:.4f}s")

            # Assert upload hasn't finished yet (proving async)
            self.assertFalse(upload_called, "Upload finished before save_checkpoint returned (synchronous!)")

            # Wait for upload to finish
            time.sleep(0.6)
            self.assertTrue(upload_called, "Upload task was not executed")

            # Verify upload called with correct args
            mock_api.upload_folder.assert_called_once()

            # Shutdown executor
            self.run_train._shutdown_upload_executor()

if __name__ == "__main__":
    unittest.main()
