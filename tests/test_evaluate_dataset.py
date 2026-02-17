import os
import sys

import numpy as np
import pytest
import torch

# Add root to sys.path
sys.path.append(os.getcwd())

from evaluate import TextDataset


@pytest.fixture
def dummy_data_file(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    f = d / "data.bin"
    # Create random uint16 data
    data = np.random.randint(0, 256, size=1024, dtype=np.uint16)
    data.tofile(f)
    return f, data

def test_text_dataset_getitem(dummy_data_file):
    f_path, raw_data = dummy_data_file
    block_size = 10
    ds = TextDataset(f_path, block_size=block_size)

    # Check length
    assert len(ds) == len(raw_data) - block_size

    # Check item content
    idx = 0
    x, y = ds[idx]

    # Expected x: raw_data[idx : idx + block_size] as int16 view
    expected_x_np = raw_data[idx : idx + block_size]
    # In my implementation, TextDataset returns int16 view
    # So values > 32767 will be negative
    # But for 0-255 (which I generated), they are same.

    # Verify x is int16 tensor
    assert x.dtype == torch.int16
    assert y.dtype == torch.int16

    # Verify values match (when cast to long and masked)
    x_long = x.long()
    x_long.bitwise_and_(0xffff)

    expected_x_long = torch.from_numpy(expected_x_np.astype(np.int64))

    assert torch.all(x_long == expected_x_long)

def test_text_dataset_large_values(tmp_path):
    # Test with values that would be negative in int16
    f = tmp_path / "large_data.bin"
    # Create data with 60000 (which is negative in int16)
    data = np.array([60000, 10, 20], dtype=np.uint16)
    data.tofile(f)

    ds = TextDataset(f, block_size=2)
    x, y = ds[0]

    # x should contain [60000, 10]
    # But represented as int16: [ -5536, 10 ]

    assert x.dtype == torch.int16
    assert x[0].item() == -5536 # 60000 - 65536
    assert x[1].item() == 10

    # Verify reconstruction
    x_rec = x.long()
    x_rec.bitwise_and_(0xffff)

    assert x_rec[0].item() == 60000
    assert x_rec[1].item() == 10
