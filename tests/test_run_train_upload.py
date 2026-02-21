import concurrent.futures
import os
import sys
import time
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

        # Mock torch
        mock_torch = MagicMock()
        mock_torch.__path__ = []
        mock_torch.__spec__ = None

        # Functions returning tensors
        mock_torch.zeros.return_value = MagicMock()

        self.mock_modules["torch"] = mock_torch
        self.mock_modules["torch.nn"] = MagicMock()
        self.mock_modules["torch.nn.functional"] = MagicMock()
        self.mock_modules["torch.utils"] = MagicMock()
        self.mock_modules["torch.utils.checkpoint"] = MagicMock()
        self.mock_modules["torch.optim"] = MagicMock()
        self.mock_modules["torch.utils.data"] = MagicMock()

        # Start patcher
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        # Import run_train safely
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())

        # If run_train was already imported, reload it to apply mocks
        import importlib

        import run_train
        importlib.reload(run_train)
        self.run_train = run_train

    def tearDown(self):
        self.patcher.stop()
        if "run_train" in sys.modules:
            del sys.modules["run_train"]

    def test_save_checkpoint_async_upload(self):
        """Test that save_checkpoint performs upload asynchronously."""

        # Setup mocks for save_checkpoint arguments
        fabric = MagicMock()
        fabric.is_global_zero = True
        out_dir = MagicMock()
        out_dir.__truediv__.return_value = MagicMock() # checkpoint_dir
        step = 1
        total_tokens = 100
        model = MagicMock()
        optimizer = MagicMock()
        hf_repo_id = "test/repo"

        # Mock HfApi to simulate slow upload
        with patch("huggingface_hub.HfApi") as MockHfApi:
            mock_api = MockHfApi.return_value

            # Event to signal upload start and allow test to verify async return
            upload_started = concurrent.futures.Future()

            def slow_upload(*args, **kwargs):
                upload_started.set_result(True)
                time.sleep(0.5) # Simulate work

            mock_api.upload_folder.side_effect = slow_upload

            # We need to ensure run_train imports our mocked HfApi
            # run_train imports HfApi inside _upload_and_cleanup
            # Since we patched sys.modules, run_train should pick it up if we patch huggingface_hub in sys.modules

            with patch.dict("sys.modules", {"huggingface_hub": MagicMock(HfApi=MockHfApi)}):

                print("Calling save_checkpoint...")
                start_time = time.time()
                self.run_train.save_checkpoint(
                    fabric,
                    out_dir,
                    step,
                    total_tokens,
                    model,
                    optimizer,
                    upload_to_hf=True,
                    hf_repo_id=hf_repo_id,
                )
                end_time = time.time()
                duration = end_time - start_time
                print(f"save_checkpoint took {duration:.4f} seconds")

                # Assert it returned quickly (e.g. < 0.1s)
                self.assertLess(duration, 0.1, "save_checkpoint took too long, likely synchronous")

                # Wait for upload to start to confirm it was triggered
                try:
                    upload_started.result(timeout=2.0)
                except concurrent.futures.TimeoutError:
                    self.fail("Upload task was not started within timeout")

                # Verify upload_folder was called
                # We need to wait for the thread to finish to be sure
                executor = self.run_train._get_upload_executor()
                executor.shutdown(wait=True)

                mock_api.upload_folder.assert_called_once()

if __name__ == "__main__":
    unittest.main()
