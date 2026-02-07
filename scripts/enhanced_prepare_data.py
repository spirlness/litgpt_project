#!/usr/bin/env python3
"""
增强版数据处理脚本，确保在数据处理完成后自动生成index.json文件
"""

import argparse
import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any

import requests
import yaml

log_dataset_to_wandb = None
HAS_WANDB_DATASET = False

try:
    import wandb_dataset  # type: ignore

    log_dataset_to_wandb = wandb_dataset.log_dataset_to_wandb
    HAS_WANDB_DATASET = True
except ImportError:
    pass


def _ensure_data_directories(data_path: Path) -> None:
    """确保数据目录存在"""
    data_path.mkdir(parents=True, exist_ok=True)
    (data_path / "train").mkdir(parents=True, exist_ok=True)
    (data_path / "val").mkdir(parents=True, exist_ok=True)


def _count_bin_files(directory: Path) -> int:
    """统计目录中bin文件的数量"""
    return len(list(directory.glob("*.bin")))


def _generate_index_json(output_dir: Path, chunk_pattern: str = "chunk-*-*.bin", estimated_samples: int = 1000) -> None:
    """
    为处理后的数据生成index.json文件

    Args:
        output_dir: 处理后数据的输出目录
        chunk_pattern: chunk文件的模式
        estimated_samples: 每个chunk的估计样本数
    """
    # 查找所有chunk文件
    chunk_files = sorted(output_dir.glob(chunk_pattern))

    if not chunk_files:
        print(f"警告: 在 {output_dir} 中未找到chunk文件")
        return

    # 创建chunks列表
    chunks = []
    for i, chunk_file in enumerate(chunk_files):
        chunk_info = {
            "chunk_index": i,
            "filename": chunk_file.name,
            "dim": None,  # 对于tokenized数据，通常为None
            "num_samples": estimated_samples,  # 估计的样本数
        }
        chunks.append(chunk_info)

    # 创建index.json内容
    index_data = {
        "version": "0.1",
        "chunks": chunks,
        "config": {
            "chunk_bytes": "50MB"  # 与TextFiles中使用的配置一致
        },
    }

    # 写入index.json文件
    index_file = output_dir / "index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"已生成 {index_file}，包含 {len(chunks)} 个chunk")


def _verify_index_json(directory: Path) -> bool:
    """
    验证index.json文件是否存在且有效

    Args:
        directory: 要检查的目录

    Returns:
        bool: 如果index.json文件存在且有效则返回True
    """
    index_file = directory / "index.json"

    if not index_file.exists():
        print(f"错误: 在 {directory} 中未找到 index.json 文件")
        return False

    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 检查基本结构
        if "version" not in data or "chunks" not in data:
            print(f"错误: {index_file} 缺少必要的字段")
            return False

        print(f"验证通过: {index_file} 包含 {len(data['chunks'])} 个chunk")
        return True

    except json.JSONDecodeError as e:
        print(f"错误: {index_file} 不是有效的JSON文件: {e}")
        return False
    except Exception as e:
        print(f"错误: 读取 {index_file} 时出现问题: {e}")
        return False


def _cleanup_existing_data(data_module, force_reprocess: bool = False) -> None:
    """
    清理现有数据（如果需要重新处理）

    Args:
        data_module: TextFiles数据模块实例
        force_reprocess: 是否强制重新处理
    """
    if force_reprocess:
        print("强制重新处理数据...")
        if data_module.out_path_train.exists():
            import shutil

            shutil.rmtree(data_module.out_path_train)
            print(f"已删除 {data_module.out_path_train}")

        if data_module.out_path_val.exists():
            import shutil

            shutil.rmtree(data_module.out_path_val)
            print(f"已删除 {data_module.out_path_val}")


# Set environment variables for data processing
os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = "0"
os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = str(min(2, os.cpu_count() or 1))  # 限制工作者数量


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare tokenized dataset for LitGPT TextFiles")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/custom_text"),
        help="Dataset root directory (will contain train/ and val/)",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing even if preprocessed data exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of workers for data processing (default: 2)",
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
        default=os.environ.get("WANDB_DATA_ALIASES", "latest").split(",")
        if os.environ.get("WANDB_DATA_ALIASES")
        else ["latest"],
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

    # 限制工作者数量以避免资源争用
    num_workers = min(args.workers, os.cpu_count() or 1)
    os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = str(num_workers)
    print(f"使用 {num_workers} 个工作进程进行数据处理")

    # Load model config
    with open("configs/moe_200m.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Model config: {config['name']}")
    print(f"Vocab size: {config['padded_vocab_size']}")
    print(f"Context length: {config['block_size']}")
    print(f"MoE experts: {config['n_expert']}, active per token: {config['n_expert_per_token']}")

    # Import LitGPT components
    from litgpt.data import TextFiles
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

    # 清理现有数据（如果需要）
    if args.force_reprocess:
        train_out_dir = data_path / "train" / "train"
        val_out_dir = data_path / "val" / "val"
        if train_out_dir.exists():
            import shutil

            shutil.rmtree(train_out_dir)
            print(f"已删除 {train_out_dir}")
        if val_out_dir.exists():
            import shutil

            shutil.rmtree(val_out_dir)
            print(f"已删除 {val_out_dir}")

    data_module = TextFiles(
        train_data_path=data_path / "train",
        val_data_path=data_path / "val",
        num_workers=num_workers,
    )

    # Connect with tokenizer and batch size
    data_module.connect(tokenizer=tokenizer, batch_size=8, max_seq_length=2047)

    # Prepare data (downloads and tokenizes)
    print("\nPreparing tokenized data...")
    print(f"Data will be saved to: {data_path}")
    print("This may take a while...")
    data_module.prepare_data()

    print("\n" + "=" * 60)
    print("数据处理完成！")

    print("\n验证处理后的数据...")
    train_out_dir = data_module.out_path_train
    val_out_dir = data_module.out_path_val

    print(f"训练数据目录: {train_out_dir}")
    print(f"验证数据目录: {val_out_dir}")

    # 验证训练数据
    if _verify_index_json(train_out_dir):
        print("✓ 训练数据验证成功")
    else:
        print("✗ 训练数据验证失败，index.json 缺失或损坏")

    # 验证验证数据
    if _verify_index_json(val_out_dir):
        print("✓ 验证数据验证成功")
    else:
        print("✗ 验证数据验证失败，index.json 缺失或损坏")

    print("=" * 60)
    print("数据准备结束")
    print("=" * 60)

    if args.log_to_wandb:
        if not HAS_WANDB_DATASET:
            print("Warning: wandb_dataset module not found. Skipping W&B upload.")
        elif not args.wandb_project:
            raise SystemExit("--log-to-wandb requires wandb_dataset module and --wandb-project or WANDB_PROJECT")
        elif log_dataset_to_wandb is not None:
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
