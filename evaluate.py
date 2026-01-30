import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from litgpt import GPT, Config


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

    data = np.memmap(val_files[0], dtype=np.uint16, mode="r")
    total_len = len(data)
    if total_len < 2:
        print(f"Validation data too small: {total_len} tokens")
        return

    # Some smoke-test datasets can be shorter than the model context length.
    block_size = min(config.block_size, total_len - 1)
    max_start = total_len - block_size - 1
    if max_start < 0:
        max_start = 0

    print(f"Validation data size: {total_len} tokens")

    losses = []

    print("Starting evaluation...")
    with torch.no_grad():
        for i in range(max_batches):
            ix = torch.randint(0, max_start + 1, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

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
