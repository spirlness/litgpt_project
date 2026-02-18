
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock all dependencies
sys.modules["torch"] = MagicMock()
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["litdata"] = MagicMock()
sys.modules["litdata.streaming"] = MagicMock()
sys.modules["litgpt"] = MagicMock()
sys.modules["litgpt.data"] = MagicMock()
sys.modules["litgpt.tokenizer"] = MagicMock()

# Mock the DataModule class so FixedTextFiles can inherit from it
class MockDataModule:
    def __init__(self):
        pass
    def prepare_data(self):
        pass
    def setup(self, stage=None):
        pass
    def connect(self, **kwargs):
        pass

# Setup mocks
sys.modules["litgpt.data"].DataModule = MockDataModule
sys.modules["torch.utils.data"].DataLoader = MagicMock()

# Now import the module under test
# We need to make sure src is importable
sys.path.append(os.getcwd())
from src.litgpt_moe.fixed_text_files import FixedTextFiles  # noqa: E402


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

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_prepare_data_files_enumeration(self):
        # Setup the module
        module = FixedTextFiles(train_data_path=self.train_dir, num_workers=4)
        module.tokenizer = MagicMock() # Mock tokenizer

        # Mock validate_tokenizer in the module namespace or patch
        with patch("src.litgpt_moe.fixed_text_files.validate_tokenizer"):
            # Mock litdata.optimize
            with patch("litdata.optimize") as mock_optimize:
                # Need to mock os.path.isdir for output paths to force processing
                # But module.out_path_train check uses Path(...).is_dir()
                # So we can just rely on non-existence (since we created empty temp dir)
                # But wait, module creates out_path_train = train_data_path / "train"
                # If train_data_path / "train" doesn't exist, it proceeds.
                # In setUp we made train_dir.
                # out_path_train will be train_dir / "train".
                # We haven't created "train" subdir, so it doesn't exist. Good.

                module.prepare_data()

                # Verify optimize was called
                self.assertTrue(mock_optimize.called)

                # Inspect the 'inputs' argument passed to optimize
                calls = mock_optimize.call_args_list
                self.assertEqual(len(calls), 2)

                # First call is for train split
                # train_files was sorted(["a.txt", "b.txt", "c.txt"])
                # val_files, *train_files = train_files
                # val_files -> a.txt
                # train_files -> b.txt, c.txt

                # First optimize call is for train
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

        module = FixedTextFiles(train_data_path=self.train_dir, val_data_path=val_dir, num_workers=2)
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
