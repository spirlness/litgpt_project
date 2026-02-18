import sys
from pathlib import Path
import numpy as np
import torch
import pytest

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluate import TextDataset

def test_text_dataset_loading(tmp_path):
    # Create dummy data with values covering full uint16 range
    data_file = tmp_path / "test_data.bin"
    # Sequence: 0, 1, ..., 65535, 0, 1, ...
    # Create enough data
    data_len = 1000
    data = np.arange(data_len, dtype=np.int32) % 65536
    # Add some specific large values
    data[-10:] = [65535, 32768, 32767, 0, 1, 2, 3, 4, 5, 6]
    data = data.astype(np.uint16)
    data.tofile(data_file)

    block_size = 10
    dataset = TextDataset(data_file, block_size=block_size)

    # Test length
    assert len(dataset) == len(data) - block_size

    # Test getitem for random index
    idx = 100
    x, y = dataset[idx]

    # Expected
    expected_x = torch.tensor(data[idx:idx+block_size].astype(np.int64), dtype=torch.long)
    expected_y = torch.tensor(data[idx+1:idx+1+block_size].astype(np.int64), dtype=torch.long)

    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)

    # Test values > 32767 to ensure unsigned interpretation is correct
    # Find index of 32768
    indices = np.where(data == 32768)[0]
    if len(indices) > 0:
        idx = indices[0]
        # Make sure we can read a full block
        if idx + block_size < len(data):
            x, y = dataset[idx]
            assert x[0].item() == 32768
            assert x.dtype == torch.int64
            assert (x >= 0).all()

    # Find index of 65535
    indices = np.where(data == 65535)[0]
    if len(indices) > 0:
        idx = indices[0]
        if idx + block_size < len(data):
            x, y = dataset[idx]
            assert x[0].item() == 65535
