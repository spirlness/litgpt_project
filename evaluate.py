import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from litgpt import GPT, Config
from torch.utils.data import DataLoader, Dataset, RandomSampler


class TextDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.file_path = file_path
        self.block_size = block_size
        self.data = np.memmap(self.file_path, dtype=np.uint16, mode="r")
        # Ensure we have enough data for at least one block
        self.total_len = max(0, len(self.data) - self.block_size - 1)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Retrieve chunk and convert to int64 for PyTorch
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


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

    model_config_path = None
    candidates = [
        checkpoint_dir / "final" / "model_config.yaml",
        checkpoint_dir / "model_config.yaml",
        Path("model_config.yaml"),
    ]
    step_configs = sorted(checkpoint_dir.glob("step-*/model_config.yaml"))
    if step_configs:
        candidates.insert(1, step_configs[-1])

    for path in candidates:
        if path.exists():
            model_config_path = path
            break

    if model_config_path is None:
        print(f"Config not found (model_config.yaml). Searched under: {checkpoint_dir}")
        return

    with open(model_config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    model_config_dict = loaded.get("model_config", loaded)
    config = Config(**model_config_dict)

    model = GPT(config)

    ckpt_path = checkpoint_dir / "final" / "lit_model.pth"
    if not ckpt_path.exists():
        ckpts = list(checkpoint_dir.glob("**/*.pth"))
        if ckpts:
            ckpt_path = sorted(ckpts)[-1]
        else:
            ckpt_path = None

    if ckpt_path is not None:
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

    # We use the first file for validation as before
    data_file = val_files[0]

    # Check size first to determine block_size adjustments
    temp_data = np.memmap(data_file, dtype=np.uint16, mode="r")
    total_len = len(temp_data)
    if total_len < 2:
        print(f"Validation data too small: {total_len} tokens")
        return

    # Some smoke-test datasets can be shorter than the model context length.
    block_size = min(config.block_size, total_len - 1)

    print(f"Validation data size: {total_len} tokens")

    # Initialize Dataset and DataLoader
    dataset = TextDataset(data_file, block_size)

    if len(dataset) <= 0:
         print(f"Dataset length {len(dataset)} is too small for block size {block_size}")
         return

    # Use RandomSampler with replacement to mimic original behavior and handle arbitrary max_batches
    sampler = RandomSampler(dataset, replacement=True, num_samples=max_batches * batch_size)

    num_workers = os.cpu_count() or 1

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=True if num_workers > 0 else False
    )

    losses = []

    print(f"Starting evaluation with {num_workers} workers...")
    with torch.no_grad():
        # Loop over dataloader. The sampler ensures we get max_batches * batch_size samples.
        # So the dataloader will yield max_batches batches.
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())

            if (i + 1) % 10 == 0:
                print(f"Batch {i + 1}/{max_batches}, Current Loss: {loss.item():.4f}")

    avg_loss = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print("=" * 40)
    print("Evaluation Complete")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a LitGPT checkpoint on tokenized .bin shards")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/custom_text"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
    )
