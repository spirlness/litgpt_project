import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path so we can import run_train
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_train  # noqa: E402


def test_save_checkpoint_async_upload():
    mock_fabric = MagicMock()
    mock_fabric.is_global_zero = True
    mock_out_dir = Path("checkpoints")
    mock_model = MagicMock()
    mock_optimizer = MagicMock()

    # Reset the global executor to ensure _get_upload_executor creates a new one or uses the mock if patched correctly
    run_train._UPLOAD_EXECUTOR = None

    # Patch the ThreadPoolExecutor class in concurrent.futures to return a mock executor
    # This is more robust than patching _get_upload_executor if run_train is already imported/initialized
    with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_cls:
        mock_executor = MagicMock()
        mock_executor_cls.return_value = mock_executor

        # Call save_checkpoint with upload enabled
        run_train.save_checkpoint(
            fabric=mock_fabric,
            out_dir=mock_out_dir,
            step=1,
            total_tokens=100,
            model=mock_model,
            optimizer=mock_optimizer,
            upload_to_hf=True,
            hf_repo_id="test/repo"
        )

        # Assert that submit was called on the executor instance
        mock_executor.submit.assert_called_once()

        # Verify arguments passed to submit
        args, _ = mock_executor.submit.call_args
        # The first argument should be _upload_and_cleanup function
        assert args[0] == run_train._upload_and_cleanup
        # Following arguments should match
        assert args[2] == "test/repo"
        assert args[3] == 1

def test_upload_and_cleanup_calls_hf_api():
    # Test that _upload_and_cleanup calls HfApi.upload_folder
    mock_checkpoint_dir = Path("checkpoints/step-00000001")
    mock_out_dir = Path("checkpoints")
    repo_id = "test/repo"
    step = 1

    # We mock huggingface_hub.HfApi because it's imported inside the function
    with patch("huggingface_hub.HfApi") as mock_hf_api_cls:
        mock_api_instance = MagicMock()
        mock_hf_api_cls.return_value = mock_api_instance

        run_train._upload_and_cleanup(mock_checkpoint_dir, repo_id, step, mock_out_dir)

        mock_api_instance.upload_folder.assert_called_once()
        _, kwargs = mock_api_instance.upload_folder.call_args
        assert kwargs["repo_id"] == repo_id
        assert kwargs["folder_path"] == str(mock_checkpoint_dir)
        assert kwargs["path_in_repo"] == f"step-{step:08d}"

def test_shutdown_upload_executor_waits():
    # Verify that shutdown is called with wait=True to prevent data loss on exit

    # Set the global executor to a mock
    mock_executor = MagicMock()
    run_train._UPLOAD_EXECUTOR = mock_executor

    with patch("builtins.print") as mock_print:
        try:
            run_train._shutdown_upload_executor()

            # Verify shutdown called with wait=True
            mock_executor.shutdown.assert_called_with(wait=True)
            # Verify prints
            mock_print.assert_any_call("Waiting for pending uploads to finish...")
            mock_print.assert_any_call("Uploads finished.")
        finally:
            # Reset global state
            run_train._UPLOAD_EXECUTOR = None
