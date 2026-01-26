import yaml
from pathlib import Path
import os
import sys
import requests
import tarfile
import gzip
import shutil
import argparse

from wandb_dataset import log_dataset_to_wandb


def _ensure_data_directories(data_path: Path) -> None:
    data_path.mkdir(parents=True, exist_ok=True)
    (data_path / "train").mkdir(parents=True, exist_ok=True)
    (data_path / "val").mkdir(parents=True, exist_ok=True)




# Set environment variables for data processing
os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = "0"
os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = str(os.cpu_count() or 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare tokenized dataset for LitGPT TextFiles")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/custom_text"),
        help="Dataset root directory (will contain train/ and val/)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT"),
        help="If set, upload dataset as a W&B Artifact to this project",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="Optional W&B entity/team",
    )
    parser.add_argument(
        "--wandb-artifact",
        type=str,
        default=os.environ.get("WANDB_DATA_ARTIFACT", "dataset-custom_text"),
        help="W&B artifact name (no spaces)",
    )
    parser.add_argument(
        "--wandb-alias",
        action="append",
        default=os.environ.get("WANDB_DATA_ALIASES", "latest").split(",") if os.environ.get("WANDB_DATA_ALIASES") else ["latest"],
        help="Artifact alias (repeatable). Default: latest",
    )
    parser.add_argument(
        "--wandb-tag",
        action="append",
        default=os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else [],
        help="Run tag (repeatable)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.environ.get("WANDB_RUN_NAME"),
        help="Optional W&B run name",
    )
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        default=bool(os.environ.get("WANDB_LOG_DATASET")),
        help="Enable dataset upload to W&B (or set WANDB_LOG_DATASET=1)",
    )
    args = parser.parse_args()

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
    data_path = args.data_dir
    _ensure_data_directories(data_path)
    data_module = TextFiles(
        train_data_path=data_path / "train",
        val_data_path=data_path / "val",
        num_workers=os.cpu_count() or 1,
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

    print(f"Training data: {data_module.out_path_train}")
    print(f"Validation data: {data_module.out_path_val}")
    print("=" * 60)

    if args.log_to_wandb:
        if not args.wandb_project:
            raise SystemExit("--log-to-wandb requires --wandb-project or WANDB_PROJECT")
        print("\nUploading dataset to Weights & Biases as an Artifact...")
        log_dataset_to_wandb(
            data_dir=data_path,
            project=args.wandb_project,
            entity=args.wandb_entity,
            artifact_name=args.wandb_artifact,
            aliases=[a for a in args.wandb_alias if a],
            tags=[t for t in args.wandb_tag if t],
            run_name=args.wandb_run_name,
        )
        print("W&B dataset Artifact upload complete.")
