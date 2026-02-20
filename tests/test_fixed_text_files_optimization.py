import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestFixedTextFilesOptimization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.train_dir = Path(self.temp_dir) / "train_data"
        self.train_dir.mkdir()

        # Create some dummy .txt files
        self.files = ["a.txt", "b.txt", "c.txt"]
        for f in self.files:
            (self.train_dir / f).touch()

        # Create a subdirectory that should be ignored (scandir is not recursive)
        (self.train_dir / "subdir").mkdir()
        (self.train_dir / "subdir" / "d.txt").touch()

        # Setup Mocks for sys.modules
        self.mock_modules = {
            "torch": MagicMock(),
            "torch.utils": MagicMock(),
            "torch.utils.data": MagicMock(),
            "litdata": MagicMock(),
            "litdata.streaming": MagicMock(),
            "litgpt": MagicMock(),
            "litgpt.data": MagicMock(),
            "litgpt.tokenizer": MagicMock(),
        }

        # Make torch mock package-like
        self.mock_modules["torch"].__path__ = []
        self.mock_modules["torch"].__spec__ = None
        self.mock_modules["torch.nn"] = MagicMock()
        self.mock_modules["torch.nn.functional"] = MagicMock()

        # Mock DataModule
        class MockDataModule:
            def __init__(self):
                pass

            def prepare_data(self):
                pass

            def setup(self, stage=None):
                pass

            def connect(self, **kwargs):
                pass

        self.mock_modules["litgpt.data"].DataModule = MockDataModule
        self.mock_modules["torch.utils.data"].DataLoader = MagicMock()

        # Start sys.modules patcher
        self.patcher = patch.dict(sys.modules, self.mock_modules)
        self.patcher.start()

        # Import the module under test with patched dependencies
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())

        # We need to force reload/import to pick up patched modules
        # If it was already imported, reload it. If not, import it.
        try:
            import src.litgpt_moe.fixed_text_files

            importlib.reload(src.litgpt_moe.fixed_text_files)
        except ImportError:
            # Should not happen if path is correct
            pass

        from src.litgpt_moe.fixed_text_files import FixedTextFiles

        self.FixedTextFiles = FixedTextFiles

    def tearDown(self):
        # Stop patcher to restore original modules
        self.patcher.stop()
        shutil.rmtree(self.temp_dir)

        # Un-import the module under test so it doesn't pollute subsequent tests
        # with references to mocks
        if "src.litgpt_moe.fixed_text_files" in sys.modules:
            del sys.modules["src.litgpt_moe.fixed_text_files"]

    def test_prepare_data_files_enumeration(self):
        # Setup the module
        module = self.FixedTextFiles(train_data_path=self.train_dir, num_workers=4)
        module.tokenizer = MagicMock()  # Mock tokenizer

        # Mock validate_tokenizer in the module namespace or patch
        with patch("src.litgpt_moe.fixed_text_files.validate_tokenizer"):
            # Mock litdata.optimize
            with patch("litdata.optimize") as mock_optimize:
                module.prepare_data()

                # Verify optimize was called
                self.assertTrue(mock_optimize.called)

                # Inspect the 'inputs' argument passed to optimize
                calls = mock_optimize.call_args_list
                self.assertEqual(len(calls), 2)

                # First call is for train split
                args1, kwargs1 = calls[0]
                inputs1 = kwargs1["inputs"]
                expected_train = sorted([str(self.train_dir / f) for f in ["b.txt", "c.txt"]])
                self.assertEqual(inputs1, expected_train)

                # Second call is for val split
                args2, kwargs2 = calls[1]
                inputs2 = kwargs2["inputs"]
                expected_val = [str(self.train_dir / "a.txt")]
                self.assertEqual(inputs2, expected_val)

                # Verify num_workers usage (should be 4)
                self.assertEqual(kwargs1["num_workers"], 4)
                self.assertEqual(kwargs2["num_workers"], 4)

    def test_prepare_data_with_explicit_val_path(self):
        val_dir = Path(self.temp_dir) / "val_data"
        val_dir.mkdir()
        (val_dir / "val.txt").touch()

        module = self.FixedTextFiles(train_data_path=self.train_dir, val_data_path=val_dir, num_workers=2)
        module.tokenizer = MagicMock()

        with patch("src.litgpt_moe.fixed_text_files.validate_tokenizer"):
            with patch("litdata.optimize") as mock_optimize:
                module.prepare_data()

                calls = mock_optimize.call_args_list
                self.assertEqual(len(calls), 2)

                # Train call
                args1, kwargs1 = calls[0]
                inputs1 = kwargs1["inputs"]
                # Here we don't split, so all files in train_dir are train files
                expected_train = sorted([str(self.train_dir / f) for f in self.files])
                self.assertEqual(inputs1, expected_train)

                # Val call
                args2, kwargs2 = calls[1]
                inputs2 = kwargs2["inputs"]
                expected_val = [str(val_dir / "val.txt")]
                self.assertEqual(inputs2, expected_val)


if __name__ == "__main__":
    unittest.main()
