import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
from pathlib import Path
import os
from dataclasses import dataclass

# Try to import litgpt. If it fails, mock it.
try:
    import litgpt
    import litgpt.config
    import litgpt.model
except ImportError:
    # Mock litgpt dependencies before importing evaluate

    @dataclass
    class MockConfig:
        block_size: int = 1024
        vocab_size: int = 50257
        n_layer: int = 2
        n_head: int = 2
        n_embd: int = 32
        def __post_init__(self):
            pass

    mock_litgpt = MagicMock()
    mock_litgpt.config = MagicMock()
    mock_litgpt.config.Config = MockConfig
    mock_litgpt.Config = MockConfig
    mock_litgpt.model = MagicMock()
    mock_litgpt.model.GPT = MagicMock()

    sys.modules["litgpt"] = mock_litgpt
    sys.modules["litgpt.config"] = mock_litgpt.config
    sys.modules["litgpt.model"] = mock_litgpt.model

# Now import evaluate
try:
    from evaluate import TextDataset
except ImportError as e:
    # Fallback for when running from root
    sys.path.append(os.getcwd())
    from evaluate import TextDataset

class TestTextDataset(unittest.TestCase):
    def setUp(self):
        self.test_file = Path("test_data.bin")
        self.block_size = 10
        # Create dummy data
        self.data_len = 20
        self.data = np.arange(self.data_len, dtype=np.uint16)
        with open(self.test_file, "wb") as f:
            self.data.tofile(f)

    def tearDown(self):
        if self.test_file.exists():
            self.test_file.unlink()

    def test_getitem_types_and_values(self):
        dataset = TextDataset(self.test_file, self.block_size)
        idx = 0
        x, y = dataset[idx]

        # Verify x is int32 (Optimization)
        self.assertEqual(x.dtype, torch.int32, "x should be int32 (IntTensor)")
        # Verify y is int64
        self.assertEqual(y.dtype, torch.int64, "y should be int64 (LongTensor)")

        # Verify values
        expected_x = self.data[idx : idx + self.block_size].astype(np.int64)
        expected_y = self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64)

        torch.testing.assert_close(x.long(), torch.from_numpy(expected_x))
        torch.testing.assert_close(y, torch.from_numpy(expected_y))

    def test_len(self):
        dataset = TextDataset(self.test_file, self.block_size)
        self.assertEqual(len(dataset), self.data_len - self.block_size)

if __name__ == "__main__":
    unittest.main()
