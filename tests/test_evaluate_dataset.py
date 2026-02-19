import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import warnings
from evaluate import TextDataset

class TestEvaluateDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.file_path = Path(self.temp_dir) / "test.bin"

        # Create dummy uint16 data
        self.data_len = 10000
        self.data = np.random.randint(0, 65535, size=self.data_len, dtype=np.uint16)
        self.data.tofile(self.file_path)

        self.block_size = 128
        self.dataset = TextDataset(self.file_path, self.block_size)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_getitem_correctness(self):
        # Compare loaded data with original numpy data
        # We need to manually perform the "standard" loading to compare

        # Pick random indices
        indices = np.random.randint(0, len(self.dataset), 10)

        for idx in indices:
            x, y = self.dataset[idx]

            # Expected x: copy, cast to int64
            expected_x = self.data[idx : idx + self.block_size].astype(np.int64)
            expected_y = self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64)

            # Verify shapes
            self.assertEqual(x.shape, (self.block_size,))
            self.assertEqual(y.shape, (self.block_size,))

            # Verify dtypes
            self.assertEqual(x.dtype, torch.long)
            self.assertEqual(y.dtype, torch.long)

            # Verify values
            np.testing.assert_array_equal(x.numpy(), expected_x)
            np.testing.assert_array_equal(y.numpy(), expected_y)

    def test_zero_copy_optimization(self):
        # This test ensures that we are indeed using zero-copy (or at least valid logic)
        # It's hard to test "zero-copy-ness" without checking memory addresses,
        # but we can check if it handles negative values correctly if we were to use int16 view

        # Create data with values > 32767 to test uint16 -> int16 casting issues
        high_val_data = np.array([32768, 65535, 0, 1], dtype=np.uint16)
        high_val_path = Path(self.temp_dir) / "high_val.bin"
        high_val_data.tofile(high_val_path)

        ds = TextDataset(high_val_path, block_size=2)
        x, y = ds[0]

        # x should be [32768, 65535]
        # if simply viewed as int16, 32768 is -32768, 65535 is -1
        # so we check if the values are correct (unsigned)

        expected_x = np.array([32768, 65535], dtype=np.int64)
        np.testing.assert_array_equal(x.numpy(), expected_x)

if __name__ == "__main__":
    unittest.main()
