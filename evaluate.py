import torch
from pathlib import Path
import sys
import yaml
import time
from litgpt import GPT, Config
from litgpt.tokenizer import Tokenizer
import numpy as np


def evaluate(
    checkpoint_dir: Path = Path("checkpoints"),
    data_dir: Path = Path("data/custom_text"),
    batch_size: int = 8,
    max_batches: int = 100,
    device: str = "auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = Path("litgpt_config.yaml")
    if not config_path.exists():
        config_path = checkpoint_dir / "litgpt_config.yaml"

    if not config_path.exists():
        print("Config not found.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
        model_config_dict = full_config.get("model_config", {})

    config = Config(**model_config_dict)

    model = GPT(config)

    ckpt_path = None
    ckpts = list(checkpoint_dir.glob("**/*.pth"))
    if ckpts:
        ckpt_path = sorted(ckpts)[-1]
        print(f"Evaluating checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
    else:
        print("Warning: Evaluating random weights!")

    model.to(device)
    model.eval()

    val_data_dir = data_dir
    # Check for bin files recursively
    val_files = list(val_data_dir.rglob("*.bin"))

    if not val_files:
        print(f"No validation .bin files found in {val_data_dir}")
        return

    print(f"Found validation files: {[f.name for f in val_files]}")

    data = np.memmap(val_files[0], dtype=np.uint16, mode="r")
    total_len = len(data)
    block_size = config.block_size

    print(f"Validation data size: {total_len} tokens")

    losses = []

    print("Starting evaluation...")
    with torch.no_grad():
        for i in range(max_batches):
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack(
                [
                    torch.from_numpy((data[i : i + block_size]).astype(np.int64))
                    for i in ix
                ]
            )
            y = torch.stack(
                [
                    torch.from_numpy(
                        (data[i + 1 : i + 1 + block_size]).astype(np.int64)
                    )
                    for i in ix
                ]
            )

            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            losses.append(loss.item())

            if (i + 1) % 10 == 0:
                print(f"Batch {i + 1}/{max_batches}, Current Loss: {loss.item():.4f}")

    avg_loss = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print("=" * 40)
    print(f"Evaluation Complete")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    evaluate()
