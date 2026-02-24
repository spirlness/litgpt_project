import sys
import unittest
from unittest.mock import MagicMock, patch

class TestRunTrainUpload(unittest.TestCase):
    def setUp(self):
        # Create a patcher for sys.modules
        self.modules_patcher = patch.dict(sys.modules, {
            "lightning": MagicMock(),
            "litgpt": MagicMock(),
            "litgpt.model": MagicMock(),
            "litgpt.tokenizer": MagicMock(),
            "src.litgpt_moe.config": MagicMock(),
            "src.litgpt_moe.fixed_text_files": MagicMock(),
            "src.litgpt_moe.utils": MagicMock(),
            "torch_xla": MagicMock(),
            "torch_xla.core.xla_model": MagicMock(),
            "torch": MagicMock(),
            "torch.nn": MagicMock(),
            "torch.nn.functional": MagicMock(),
            "torch.optim": MagicMock(),
            "torch.utils": MagicMock(),
            "torch.utils.checkpoint": MagicMock(),
            "torch.distributed": MagicMock(),
            "yaml": MagicMock(),
        })
        self.modules_patcher.start()

        # Ensure run_train is not in sys.modules to force re-import with mocks
        if "run_train" in sys.modules:
            del sys.modules["run_train"]

        import run_train
        self.run_train = run_train

        # Reset the global executor before each test
        self.run_train._UPLOAD_EXECUTOR = None

    def tearDown(self):
        # Clean up executor if any
        if self.run_train._UPLOAD_EXECUTOR:
            try:
                self.run_train._UPLOAD_EXECUTOR.shutdown(wait=False)
            except:
                pass
        self.run_train._UPLOAD_EXECUTOR = None

        # Stop patching sys.modules
        self.modules_patcher.stop()

        # Clean up run_train from sys.modules to avoid pollution
        if "run_train" in sys.modules:
            del sys.modules["run_train"]

    @patch("concurrent.futures.ProcessPoolExecutor")
    @patch("multiprocessing.get_context")
    def test_upload_executor_uses_process_pool(self, mock_get_context, mock_executor_cls):
        # Setup mock context
        mock_ctx = MagicMock()
        mock_get_context.return_value = mock_ctx

        # Setup mock executor instance
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value = mock_executor_instance

        # Call the function
        executor = self.run_train._get_upload_executor()

        # Verify get_context was called with 'spawn'
        mock_get_context.assert_called_with("spawn")

        # Verify ProcessPoolExecutor was initialized with the context
        mock_executor_cls.assert_called_with(max_workers=1, mp_context=mock_ctx)

        # Verify the returned object is the mock executor instance
        self.assertEqual(executor, mock_executor_instance)

        # Verify it's stored in global variable
        self.assertEqual(self.run_train._UPLOAD_EXECUTOR, mock_executor_instance)

    def test_shutdown_upload_executor(self):
        # specific test for shutdown logic
        mock_executor = MagicMock()
        self.run_train._UPLOAD_EXECUTOR = mock_executor

        self.run_train._shutdown_upload_executor()

        # Verify shutdown is called with wait=True
        mock_executor.shutdown.assert_called_once_with(wait=True)
        self.assertIsNone(self.run_train._UPLOAD_EXECUTOR)

if __name__ == "__main__":
    unittest.main()
