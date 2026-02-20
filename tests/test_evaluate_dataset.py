import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from evaluate import TextDataset


def test_text_dataset(tmp_path):
    # Create dummy data
    file_path = tmp_path / "data.bin"
    data_len = 100
    block_size = 10
    # Create random data including values > 32767 to test signed/unsigned issues
    # Ensure strict deterministic sequence for reproducibility if needed, but random is fine here
    rng = np.random.default_rng(42)
    data = rng.integers(0, 65535, size=data_len, dtype=np.uint16)
    data.tofile(file_path)

    dataset = TextDataset(file_path, block_size=block_size)

    assert len(dataset) == data_len - block_size

    # Test a few items
    indices = [0, 1, 10, len(dataset) - 1]

    for i in indices:
        x, y = dataset[i]

        # Check correctness of values after potential conversion
        if x.dtype == torch.int16:
            x_val = x.to(torch.long).bitwise_and_(0xFFFF)
            y_val = y.to(torch.long).bitwise_and_(0xFFFF)
        else:
            x_val = x
            y_val = y

        expected_x = torch.from_numpy(data[i : i + block_size].astype(np.int64))
        expected_y = torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))

        assert torch.equal(x_val, expected_x)
        assert torch.equal(y_val, expected_y)
