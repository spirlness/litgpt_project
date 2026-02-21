import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from evaluate import TextDataset


def test_text_dataset_zero_copy(tmp_path):
    # Create dummy data
    data_path = tmp_path / "data.bin"
    # Create uint16 data with values > 32767
    data = np.array([0, 100, 32767, 32768, 65535, 10, 20], dtype=np.uint16)
    data.tofile(data_path)

    # Init dataset
    block_size = 5
    dataset = TextDataset(data_path, block_size=block_size)

    assert len(dataset) == len(data) - block_size

    # Get item
    idx = 0
    x, y = dataset[idx]

    # Verify x and y are int16 tensors
    assert x.dtype == torch.int16
    assert y.dtype == torch.int16

    # Verify content logic (casting back to long and correcting)
    x_long = x.long()
    x_corrected = x_long & 0xffff

    y_long = y.long()
    y_corrected = y_long & 0xffff

    expected_x = torch.tensor([0, 100, 32767, 32768, 65535], dtype=torch.long)
    expected_y = torch.tensor([100, 32767, 32768, 65535, 10], dtype=torch.long)

    assert torch.equal(x_corrected, expected_x)
    assert torch.equal(y_corrected, expected_y)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
