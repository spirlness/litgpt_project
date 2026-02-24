import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import sys
import os

# Ensure evaluate.py is importable
sys.path.append(os.getcwd())
from evaluate import TextDataset

def test_text_dataset_zero_copy_correctness():
    # Create temporary file with known values
    # We want to test edge cases around 32767 (0x7FFF)
    # 0, 1, ..., 32767, 32768, ..., 65535
    values = [0, 1, 32767, 32768, 65535]
    # Create a larger array to test block size
    data = np.array(values * 10, dtype=np.uint16)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        data.tofile(f)
        filepath = Path(f.name)

    try:
        block_size = len(values)
        dataset = TextDataset(filepath, block_size=block_size)

        # Test __getitem__
        x, y = dataset[0]

        # Check dtype
        assert x.dtype == torch.int16
        assert y.dtype == torch.int16

        # Verify values
        # Expected x: values[0:block_size] as int16
        expected_x_uint16 = torch.tensor(values, dtype=torch.long)

        # Recover unsigned values
        recovered_x = x.long() & 0xffff
        recovered_y = y.long() & 0xffff

        assert torch.equal(recovered_x, expected_x_uint16)

        # Verify specific values for int16 representation
        # 32768 (0x8000) as int16 should be -32768
        # 65535 (0xFFFF) as int16 should be -1
        assert x[3].item() == -32768
        assert x[4].item() == -1

        # Verify y is shifted by 1
        expected_y_uint16 = torch.tensor(data[1:1+block_size].astype(np.int64), dtype=torch.long)
        assert torch.equal(recovered_y, expected_y_uint16)

    finally:
        if filepath.exists():
            filepath.unlink()

def test_text_dataset_dataloader_collation():
    # Verify DataLoader collation works with int16 tensors
    values = np.arange(100, dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        values.tofile(f)
        filepath = Path(f.name)

    try:
        dataset = TextDataset(filepath, block_size=10)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

        batch = next(iter(dataloader))
        x, y = batch

        assert x.shape == (2, 10)
        assert x.dtype == torch.int16

        # Recover
        recovered_x = x.long() & 0xffff

        expected_x0 = torch.from_numpy(values[0:10].astype(np.int64))
        expected_x1 = torch.from_numpy(values[1:11].astype(np.int64))

        assert torch.equal(recovered_x[0], expected_x0)
        # Note: In TextDataset, y is shifted by 1. But for batch collation, standard DataLoader takes items sequentially?
        # TextDataset[1] takes from index 1.
        # batch[1] is dataset[1]
        # dataset[1] x is data[1:1+block_size]

        assert torch.equal(recovered_x[1], expected_x1)

    finally:
        if filepath.exists():
            filepath.unlink()
