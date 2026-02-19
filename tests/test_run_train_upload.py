import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add src to sys.path
sys.path.append(str(Path.cwd()))

# Mock dependencies before importing run_train
# We need to mock these modules because run_train imports them at the top level
sys.modules["lightning"] = MagicMock()
sys.modules["lightning.fabric"] = MagicMock()
sys.modules["litgpt"] = MagicMock()
sys.modules["litgpt.model"] = MagicMock()
sys.modules["litgpt.tokenizer"] = MagicMock()
sys.modules["src.litgpt_moe.config"] = MagicMock()
sys.modules["src.litgpt_moe.fixed_text_files"] = MagicMock()
sys.modules["src.litgpt_moe.utils"] = MagicMock()

import run_train  # noqa: E402


class TestAsyncUpload(unittest.TestCase):
    def setUp(self):
        # Reset executor to ensure a clean state
        run_train._shutdown_upload_executor()

    def tearDown(self):
        run_train._shutdown_upload_executor()
        # Ensure we don't leak patches
        if hasattr(self, "_original_upload"):
            run_train._upload_and_cleanup = self._original_upload

    def test_save_checkpoint_is_non_blocking(self):
        # Setup mocks
        fabric = MagicMock()
        fabric.is_global_zero = True
        out_dir = Path("tmp_checkpoints")
        step = 10
        total_tokens = 1000
        model = MagicMock()
        optimizer = MagicMock()
        hf_repo_id = "test/repo"

        # Manually patch the function on the imported module to be absolutely sure
        self._original_upload = run_train._upload_and_cleanup
        mock_upload = MagicMock()

        # Make the mock upload slow to simulate network latency
        def slow_upload(*args, **kwargs):
            time.sleep(1.0)

        mock_upload.side_effect = slow_upload
        run_train._upload_and_cleanup = mock_upload

        try:
            start_time = time.time()
            run_train.save_checkpoint(
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
            # save_checkpoint should return almost immediately, much faster than the 1s sleep
            self.assertLess(duration, 0.5, "save_checkpoint blocked the main thread!")

            # Wait for the background thread to finish
            executor = run_train._get_upload_executor()
            executor.shutdown(wait=True)

            # Verify mock_upload was called exactly once
            mock_upload.assert_called_once()
            args, _ = mock_upload.call_args
            self.assertEqual(args[1], hf_repo_id)
            self.assertEqual(args[2], step)

        finally:
            # Restore original
            run_train._upload_and_cleanup = self._original_upload

    def test_upload_and_cleanup_function(self):
        # Test the actual upload function logic (mocking HfApi)
        with unittest.mock.patch.dict(sys.modules, {"huggingface_hub": MagicMock()}):
            import huggingface_hub

            mock_api = MagicMock()
            huggingface_hub.HfApi.return_value = mock_api

            checkpoint_dir = Path("ckpt")
            repo_id = "repo"
            step = 1
            out_dir = Path("out")

            # Mock glob to avoid FS access
            with unittest.mock.patch.object(Path, "glob", return_value=[]):
                # Use the ORIGINAL function here, not the mocked one from test_save_checkpoint
                # But since this test doesn't modify run_train globally via setUp, it should be fine
                # unless parallel execution happens (unittest runs sequentially by default)
                run_train._upload_and_cleanup(checkpoint_dir, repo_id, step, out_dir)

            mock_api.upload_folder.assert_called_once()

            # Verify arguments passed to upload_folder
            args, kwargs = mock_api.upload_folder.call_args
            self.assertEqual(kwargs["repo_id"], repo_id)
            self.assertIn(f"step-{step:08d}", kwargs["path_in_repo"])


if __name__ == "__main__":
    unittest.main()
