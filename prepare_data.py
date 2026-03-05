import argparse
from pathlib import Path

import requests
import yaml
from litgpt.tokenizer import Tokenizer

from src.litgpt_moe.fixed_text_files import FixedTextFiles

DEFAULT_TOKENIZER_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MODEL_CONFIG = Path("configs/moe_200m.yaml")


def _ensure_data_directories(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "val").mkdir(parents=True, exist_ok=True)


def _download_if_missing(url: str, output_path: Path, timeout: int) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0:
        return False
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def _ensure_tokenizer(tokenizer_dir: Path, tokenizer_repo: str, timeout: int) -> None:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://huggingface.co/{tokenizer_repo}/resolve/main"
    files = ("tokenizer.json", "tokenizer_config.json")
    downloaded = []
    for file_name in files:
        url = f"{base_url}/{file_name}"
        target = tokenizer_dir / file_name
        did_download = _download_if_missing(url, target, timeout=timeout)
        downloaded.append((file_name, did_download))

    for file_name, did_download in downloaded:
        status = "downloaded" if did_download else "cached"
        print(f"Tokenizer file {file_name}: {status}")


def _load_model_config(model_config_path: Path) -> dict:
    with model_config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if "model_config" in loaded:
        loaded = loaded["model_config"]
    return loaded


def _resolve_max_seq_length(model_config: dict, override: int | None) -> int:
    if override is not None:
        return max(int(override), 1)
    block_size = int(model_config.get("block_size", 2048))
    return max(block_size - 1, 1)


def _count_txt_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix == ".txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tokenized dataset for LitGPT MoE training")
    parser.add_argument("--data-dir", type=Path, default=Path("data/custom_text"))
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--tokenizer-repo", type=str, default=DEFAULT_TOKENIZER_REPO)
    parser.add_argument("--download-timeout", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prepare-num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _ensure_data_directories(args.data_dir)
    train_data_path = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    val_data_path = val_dir if _count_txt_files(val_dir) > 0 else None

    model_config = _load_model_config(args.model_config)
    max_seq_length = _resolve_max_seq_length(model_config, args.max_seq_length)

    print(f"Model config: {model_config.get('name', 'unknown')}")
    print(f"Tokenizer repo: {args.tokenizer_repo}")
    print(f"Data dir: {args.data_dir}")
    print(f"Train .txt files: {_count_txt_files(train_data_path)}")
    print(f"Val .txt files: {_count_txt_files(val_dir)}")
    print(f"Target max_seq_length: {max_seq_length}")

    _ensure_tokenizer(args.tokenizer_dir, args.tokenizer_repo, timeout=max(int(args.download_timeout), 1))
    tokenizer = Tokenizer(args.tokenizer_dir)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    data_module = FixedTextFiles(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        num_workers=max(int(args.num_workers), 0),
        prepare_num_workers=args.prepare_num_workers,
    )
    data_module.connect(
        tokenizer=tokenizer,
        batch_size=max(int(args.batch_size), 1),
        max_seq_length=max_seq_length,
    )

    print("\nPreparing tokenized data...")
    data_module.prepare_data()
    print("\nData preparation complete")
    print(f"Training data path: {data_module.out_path_train}")
    print(f"Validation data path: {data_module.out_path_val}")


if __name__ == "__main__":
    main()
