import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from litgpt.model import GPT
from torch.utils.data import DataLoader, Dataset

from src.litgpt_moe.config import MoEConfig


class TextDataset(Dataset):
    def __init__(self, file_path: Path, block_size: int):
        self.file_path = file_path
        self.block_size = block_size
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64))
        return x, y


def evaluate(
    checkpoint_dir: Path = Path("checkpoints"),
    data_dir: Path = Path("data/custom_text"),
    batch_size: int = 8,
    max_batches: int = 100,
    device: str = "auto",
    num_workers: int = 4,
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

    # Extract MoE specific args that are not in LitGPT Config
    moe_args = {}
    for key in ["moe_aux_loss_weight", "moe_router_stats"]:
        if key in model_config_dict:
            moe_args[key] = model_config_dict.pop(key)

    # config = Config(**model_config_dict)
    #
    # # Inject MoE args back into config object
    # for key, value in moe_args.items():
    #     setattr(config, key, value)

    # Re-integrate popped MoE args and use MoEConfig
    model_config_dict.update(moe_args)
    config = MoEConfig(**model_config_dict)

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

    # Check total length first
    temp_data = np.memmap(val_files[0], dtype=np.uint16, mode="r")
    total_len = len(temp_data)
    del temp_data

    if total_len < 2:
        print(f"Validation data too small: {total_len} tokens")
        return

    # Some smoke-test datasets can be shorter than the model context length.
    block_size = min(config.block_size, total_len - 1)

    print(f"Validation data size: {total_len} tokens")

    dataset = TextDataset(val_files[0], block_size=block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False
    )

    losses = []
    print("Starting evaluation...")

    data_iter = iter(dataloader)

    with torch.no_grad():
        for i in range(max_batches):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.detach())

            if (i + 1) % 10 == 0:
                print(f"Batch {i + 1}/{max_batches}, Current Loss: {loss.item():.4f}")

    # Accumulate losses on device and compute mean at the end to avoid per-step synchronization
    losses = torch.stack(losses)
    avg_loss = losses.mean().item()
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
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    evaluate(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
        num_workers=args.num_workers,
    )
