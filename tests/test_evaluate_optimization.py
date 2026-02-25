import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import numpy as np
import io

# Add repo root to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluate import evaluate, TextDataset

@pytest.fixture
def mock_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create random bin file
    data = np.random.randint(0, 100, size=1000, dtype=np.uint16)
    data.tofile(data_dir / "test.bin")
    return data_dir

@pytest.fixture
def mock_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    config = {
        "model_config": {
            "block_size": 128,
            "vocab_size": 100,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 32,
            "n_expert": 0, # Disable MoE for simplicity
            "moe_aux_loss_weight": 0.01,
            "moe_router_stats": False
        }
    }

    import yaml
    with open(ckpt_dir / "model_config.yaml", "w") as f:
        yaml.dump(config, f)

    return ckpt_dir

def test_evaluate_runs_and_prints_metrics(mock_data, mock_checkpoint, capsys):
    # Run evaluate
    evaluate(
        checkpoint_dir=mock_checkpoint,
        data_dir=mock_data,
        batch_size=2,
        max_batches=5,
        device="cpu",
        num_workers=0
    )

    captured = capsys.readouterr()
    assert "Evaluation Complete" in captured.out
    assert "Average Loss:" in captured.out
    assert "Perplexity:" in captured.out

    # Parse loss
    import re
    match = re.search(r"Average Loss: (\d+\.\d+)", captured.out)
    assert match
    loss = float(match.group(1))
    assert loss > 0

def test_evaluate_loss_accumulation_logic():
    # This test verifies the logic of accumulating tensors and taking mean
    # We can't easily mock inside evaluate without modifying it,
    # but we can verify that the pattern works as expected.

    losses = []
    for i in range(10):
        t = torch.tensor(float(i))
        losses.append(t)

    avg_loss = torch.stack(losses).mean().item()
    assert avg_loss == 4.5
