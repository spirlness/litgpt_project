import yaml
from pathlib import Path
import os
import sys
import requests
import tarfile
import gzip
import shutil

# Set environment variables for data processing
os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = "0"
os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = "1"

if __name__ == "__main__":
    # Load model config
    with open("configs/moe_200m.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Model config: {config['name']}")
    print(f"Vocab size: {config['padded_vocab_size']}")
    print(f"Context length: {config['block_size']}")
    print(
        f"MoE experts: {config['n_expert']}, active per token: {config['n_expert_per_token']}"
    )

    # Import LitGPT components
    from litgpt.data import TextFiles, DataModule
    from litgpt.tokenizer import Tokenizer

    # Download and prepare tokenizer first
    tokenizer_dir = Path("data/tokenizer")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # Use TinyLlama tokenizer (has BOS)
    tokenizer_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json"
    tokenizer_config_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer_config.json"

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    config_path = tokenizer_dir / "tokenizer_config.json"

    # Download tokenizer files if missing
    print(f"Downloading TinyLlama tokenizer to {tokenizer_dir}...")
    if not tokenizer_path.exists():
        print("Downloading tokenizer.json...")
        response = requests.get(tokenizer_url, stream=True)
        response.raise_for_status()
        with open(tokenizer_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    if not config_path.exists():
        print("Downloading tokenizer_config.json...")
        response = requests.get(tokenizer_config_url, stream=True)
        response.raise_for_status()
        with open(config_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Tokenizer download complete!")

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = Tokenizer(tokenizer_dir)

    # Print vocab size info
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Initialize TextFiles data module
    data_path = Path("data/custom_text")
    # data_path.mkdir(parents=True, exist_ok=True) # Already created
    data_module = TextFiles(
        train_data_path=data_path / "train",
        val_data_path=data_path / "val",
        num_workers=1,
    )

    # Connect with tokenizer and batch size
    data_module.connect(tokenizer=tokenizer, batch_size=8, max_seq_length=2047)

    # Prepare data (downloads and tokenizes)
    print("\nPreparing tokenized data...")
    print(f"Data will be saved to: {data_path}")
    print("This may take a while...")
    data_module.prepare_data()

    print("\n" + "=" * 60)
    print("Data preparation complete!")

    print(f"Training data: {data_module.data_path_train}")
    print(f"Validation data: {data_module.data_path_val}")
    print("=" * 60)
