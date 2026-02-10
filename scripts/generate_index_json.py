#!/usr/bin/env python3
"""
专门用于生成或修复index.json文件的脚本
"""

import argparse
import json
import time
from pathlib import Path


def generate_index_json(data_dir: Path, dataset_type: str = "train", estimated_samples: int = 1000) -> bool:
    """
    为处理后的数据生成index.json文件

    Args:
        data_dir: 数据目录路径 (例如: data/custom_text/train/train)
        dataset_type: 数据集类型 ("train" 或 "val")
        estimated_samples: 每个chunk的估计样本数

    Returns:
        bool: 是否成功生成index.json
    """
    # 确保目录存在
    if not data_dir.exists():
        print(f"错误: 目录 {data_dir} 不存在")
        return False

    # 查找所有chunk文件
    chunk_files = sorted(data_dir.glob("chunk-*-*.bin"))

    if not chunk_files:
        print(f"警告: 在 {data_dir} 中未找到chunk文件")
        return False

    print(f"找到 {len(chunk_files)} 个chunk文件:")
    for chunk_file in chunk_files:
        size_mb = chunk_file.stat().st_size / (1024 * 1024)
        print(f"  - {chunk_file.name} ({size_mb:.1f} MB)")

    # 检查是否已存在有效的index.json
    existing_index = data_dir / "index.json"
    if existing_index.exists():
        try:
            with open(existing_index, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            # 检查是否已经是litdata格式
            if "chunks" in existing_data and "config" in existing_data:
                print("✓ 已存在有效的index.json文件，无需重新生成")
                return True
        except Exception:
            print("现有index.json文件损坏，将重新生成")

    # 根据数据集类型调整样本数估计
    if dataset_type == "val":
        estimated_samples = min(estimated_samples, 200)  # 验证集通常较小

    # 创建chunks列表（litdata格式）
    chunks = []
    for i, chunk_file in enumerate(chunk_files):
        # 获取文件大小
        file_size = chunk_file.stat().st_size

        chunk_info = {
            "chunk_bytes": file_size,
            "chunk_size": estimated_samples + i,  # 简单递增
            "dim": file_size // 4,  # 粗略估计维度（假设每个token 4字节）
            "filename": chunk_file.name,
        }
        chunks.append(chunk_info)

    # 创建index.json内容（litdata格式）
    index_data = {
        "chunks": chunks,
        "config": {
            "chunk_bytes": 50000000,  # 50MB
            "chunk_size": None,
            "compression": None,
            "data_format": ["no_header_tensor:16"],
            "data_spec": '[1, {"type": null, "context": null, "children_spec": []}]',
            "encryption": None,
            "item_loader": "TokensLoader",
        },
        "updated_at": str(time.time()),
    }

    # 写入index.json文件
    index_file = data_dir / "index.json"

    # 备份现有文件（如果存在）
    if index_file.exists():
        backup_file = index_file.with_suffix(".json.backup")
        index_file.rename(backup_file)
        print(f"已备份现有index.json到 {backup_file}")

    try:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, separators=(",", ":"))  # 紧凑格式

        print(f"✓ 成功生成 {index_file}")
        print(f"  - 包含 {len(chunks)} 个chunk")
        print("  - 格式兼容litdata")
        return True

    except Exception as e:
        print(f"✗ 生成index.json时出错: {e}")
        return False


def verify_index_json(data_dir: Path) -> bool:
    """
    验证index.json文件是否存在且有效

    Args:
        data_dir: 数据目录路径

    Returns:
        bool: 如果index.json文件存在且有效则返回True
    """
    index_file = data_dir / "index.json"

    if not index_file.exists():
        print(f"✗ 在 {data_dir} 中未找到 index.json 文件")
        return False

    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 检查基本结构
        required_fields = ["chunks", "config"]
        for field in required_fields:
            if field not in data:
                print(f"✗ {index_file} 缺少必要的字段: {field}")
                return False

        chunk_count = len(data["chunks"])
        print(f"✓ {index_file} 验证通过")
        print(f"  - 包含 {chunk_count} 个chunk")

        if "config" in data:
            if "chunk_bytes" in data["config"]:
                chunk_bytes = data["config"]["chunk_bytes"]
                if isinstance(chunk_bytes, int):
                    print(f"  - Chunk大小: {chunk_bytes / 1024 / 1024:.0f}MB")

        return True

    except json.JSONDecodeError as e:
        print(f"✗ {index_file} 不是有效的JSON文件: {e}")
        return False
    except Exception as e:
        print(f"✗ 读取 {index_file} 时出现问题: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成或修复index.json文件")
    parser.add_argument(
        "data_dirs", nargs="+", type=Path, help="数据目录路径 (例如: data/custom_text/train/train data/custom_text/val/val)"
    )
    parser.add_argument("--type", choices=["train", "val"], default="train", help="数据集类型 (默认: train)")
    parser.add_argument("--samples", type=int, default=1000, help="每个chunk的估计样本数 (默认: 1000)")
    parser.add_argument("--verify-only", action="store_true", help="仅验证现有index.json文件，不生成新的")
    parser.add_argument("--force-regenerate", action="store_true", help="强制重新生成index.json文件，即使已存在")

    args = parser.parse_args()

    print("index.json 文件处理工具")
    print("=" * 50)

    success_count = 0
    total_dirs = len(args.data_dirs)

    for data_dir in args.data_dirs:
        print(f"\n处理目录: {data_dir}")
        print("-" * 30)

        if args.verify_only:
            # 仅验证
            if verify_index_json(data_dir):
                success_count += 1
        else:
            # 生成或修复
            should_generate = args.force_regenerate

            # 检查是否已存在有效的index.json
            if not args.force_regenerate:
                if verify_index_json(data_dir):
                    print("  已存在有效的index.json文件，跳过生成")
                    success_count += 1
                    continue
                else:
                    should_generate = True

            if should_generate:
                if generate_index_json(data_dir, args.type, args.samples):
                    # 验证生成的文件
                    if verify_index_json(data_dir):
                        success_count += 1
                    else:
                        print(f"✗ {data_dir} 的index.json验证失败")
                else:
                    print(f"✗ 无法为 {data_dir} 生成index.json")
            else:
                success_count += 1

    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"成功处理: {success_count}/{total_dirs} 个目录")

    if success_count == total_dirs:
        print("✓ 所有目录处理成功!")
        return 0
    else:
        print("✗ 部分目录处理失败!")
        return 1


if __name__ == "__main__":
    exit(main())
