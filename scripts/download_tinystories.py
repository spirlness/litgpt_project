from pathlib import Path

import datasets
from tqdm import tqdm


def download_tinystories_streaming(output_dir: Path, num_train: int = 10000, num_val: int = 2000):
    """
    Download TinyStories dataset from HuggingFace using streaming and save as text files.

    Args:
        output_dir: Directory to save the dataset
        num_train: Number of training stories to download (default 10000)
        num_val: Number of validation stories to download (default 2000)
    """
    print("Downloading TinyStories dataset from HuggingFace (streaming)...")
    print(f"Training stories: {num_train}")
    print(f"Validation stories: {num_val}")

    train_output = output_dir / "train"
    val_output = output_dir / "val"

    train_output.mkdir(parents=True, exist_ok=True)
    val_output.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    train_count = 0
    val_count = 0

    total = num_train + num_val
    print(f"\nProcessing {total} stories...\n")

    with tqdm(total=total, desc="Downloading stories") as pbar:
        for i, story in enumerate(dataset):
            text = story["text"]

            if i < num_train:
                output_file = train_output / f"story_{i:07d}.txt"
                train_count += 1
            elif i < total:
                output_file = val_output / f"story_{val_count:07d}.txt"
                val_count += 1
            else:
                break

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

            pbar.update(1)

    print(f"\n✓ Saved {train_count} training stories to {train_output}")
    print(f"✓ Saved {val_count} validation stories to {val_output}")
    print("\nDataset ready for training!")


if __name__ == "__main__":
    output_dir = Path("data/custom_text")

    print("=" * 60)
    print("TinyStories Dataset Downloader (Streaming)")
    print("=" * 60)

    download_tinystories_streaming(output_dir)
