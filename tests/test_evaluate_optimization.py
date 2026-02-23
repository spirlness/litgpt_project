import sys
import os
import torch
import numpy as np
from pathlib import Path
import pytest

# Add root to sys.path to import evaluate
sys.path.append(os.getcwd())
from evaluate import TextDataset

def test_text_dataset_zero_copy_optimization(tmp_path):
    # Create a dummy bin file with specific values
    # Include values > 32767 to test int16 wrapping
    # 0, 1, 32767 (max pos int16), 32768 (min neg int16), 65535 (-1 int16)
    data = np.array([0, 1, 32767, 32768, 65535], dtype=np.uint16)
    file_path = tmp_path / "test_data.bin"
    with open(file_path, "wb") as f:
        f.write(data.tobytes())

    block_size = 2
    dataset = TextDataset(file_path, block_size=block_size)

    # __getitem__ returns (x, y) where x is data[idx:idx+block_size], y is data[idx+1:idx+1+block_size]
    # idx=0 -> x=[0, 1], y=[1, 32767]
    # idx=2 -> x=[32767, 32768], y=[32768, 65535]

    # Test case 1: simple values
    x, y = dataset[0]
    # x should be int16 tensor
    assert x.dtype == torch.int16
    assert y.dtype == torch.int16

    # Reconstruct
    x_rec = x.long().bitwise_and_(0xffff)
    y_rec = y.long().bitwise_and_(0xffff)

    assert torch.equal(x_rec, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(y_rec, torch.tensor([1, 32767], dtype=torch.long))

    # Test case 2: values > 32767 (negative in int16)
    x, y = dataset[2]
    # x=[32767, 32768] -> int16: [32767, -32768]
    assert x[0].item() == 32767
    assert x[1].item() == -32768

    x_rec = x.long().bitwise_and_(0xffff)
    y_rec = y.long().bitwise_and_(0xffff)

    assert torch.equal(x_rec, torch.tensor([32767, 32768], dtype=torch.long))
    assert torch.equal(y_rec, torch.tensor([32768, 65535], dtype=torch.long))

    # Check 65535 -> -1 in int16
    # y from idx=2 is [32768, 65535]
    # int16: [-32768, -1]
    assert y[1].item() == -1

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
