import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# Add project root to sys.path to allow importing from evaluate.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluate import TextDataset  # noqa: E402


class TestEvaluateDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.file_path = Path(self.test_dir) / "test_data.bin"

        # Create dummy data with interesting values
        # 0, 1, ..., 65535
        # We want to test boundary conditions around 32767/32768
        self.data = np.array([0, 1, 32767, 32768, 65534, 65535], dtype=np.uint16)
        with open(self.file_path, "wb") as f:
            f.write(self.data.tobytes())

        self.block_size = 2
        self.dataset = TextDataset(self.file_path, block_size=self.block_size)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_getitem_returns_int16(self):
        """Test that __getitem__ returns int16 tensors (view of uint16)."""
        x, y = self.dataset[0]

        # Verify dtype
        self.assertEqual(x.dtype, torch.int16, "x should be int16")
        self.assertEqual(y.dtype, torch.int16, "y should be int16")

    def test_value_recovery(self):
        """Test that values are correctly recovered when cast to long and masked."""
        # Index 2: [32767, 32768] -> x
        # Index 3: [32768, 65534] -> y (if block_size=2)

        # Let's test specific indices to control expected values
        # Data: [0, 1, 32767, 32768, 65534, 65535]
        # idx=2: x=[32767, 32768], y=[32768, 65534]

        x, y = self.dataset[2]

        # Verify raw values (as int16)
        # 32767 in uint16 is 32767 in int16
        # 32768 in uint16 is -32768 in int16
        self.assertEqual(x[0].item(), 32767)
        self.assertEqual(x[1].item(), -32768)

        # Verify recovery logic
        x_long = x.to(torch.long).bitwise_and(0xffff)
        y_long = y.to(torch.long).bitwise_and(0xffff)

        expected_x = torch.tensor([32767, 32768], dtype=torch.long)
        expected_y = torch.tensor([32768, 65534], dtype=torch.long)

        torch.testing.assert_close(x_long, expected_x)
        torch.testing.assert_close(y_long, expected_y)

if __name__ == "__main__":
    unittest.main()
