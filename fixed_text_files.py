# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import glob
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer
from torch.utils.data import DataLoader


@dataclass
class FixedTextFiles(DataModule):
    """修复版的TextFiles数据模块，用于预训练。

    从包含.txt文件的数据文件夹中读取文本数据，
    并提供返回token批次的训练和验证数据加载器。
    每个样本都被设置为固定长度。
    """

    train_data_path: Path
    """用于训练的数据目录路径，包含.txt文件"""
    val_data_path: Optional[Path] = None
    """用于验证的数据目录路径，包含.txt文件。
    如果为None，则从训练集中分割出数据用于验证。"""
    seed: int = 42
    """用于打乱数据集的随机种子。"""
    num_workers: int = 2  # 强制使用较少的工作者数量
    """用于数据加载的工作者数量。"""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.out_path_train = self.train_data_path / "train"
        if self.val_data_path is None:
            self.out_path_val = self.train_data_path / "val"
        else:
            self.out_path_val = Path(self.val_data_path) / "val"

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: Optional[int] = 1, max_seq_length: Optional[int] = -1
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size or 1
        self.max_seq_length = (max_seq_length or -1) + 1  # 增加1因为我们还需要下一个token

    def prepare_data(self) -> None:
        from litdata import optimize
        from litdata.streaming import TokensLoader

        # 检查是否已经存在预处理的数据
        if Path(self.out_path_train).is_dir() and Path(self.out_path_val).is_dir():
            print(f"\n跳过数据预处理：在 {self.out_path_train} 和 {self.out_path_val} 中找到了预处理数据。\n")
            return

        train_files = sorted(glob.glob(str(self.train_data_path / "*.txt")))
        assert len(train_files) > 0, f"在训练数据 {train_files} 中未找到.txt文件"

        if self.val_data_path is not None:
            self.val_data_path = Path(self.val_data_path)
            val_files = sorted(glob.glob(str(self.val_data_path / "*.txt")))
            assert len(val_files) > 0, f"在验证数据 {val_files} 中未找到.txt文件"
        # 训练/测试分割。让我们只使用分片0作为测试分割，其余作为训练
        else:
            assert len(train_files) > 1, f"期望在 {train_files} 中至少有两个.txt文件"
            val_files, *train_files = train_files
            val_files = [val_files]

        # 使用配置中指定的工作者数量，而不是几乎所有CPU核心
        use_workers = min(self.num_workers, len(train_files))  # 使用配置的num_workers
        if not Path(self.out_path_train).is_dir():
            print(f"处理训练数据，使用 {use_workers} 个工作者...")
            if self.tokenizer is not None:
                validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=train_files,
                output_dir=str(self.out_path_train),
                num_workers=use_workers,
                chunk_bytes="50MB",
                item_loader=TokensLoader(block_size=self.max_seq_length),
            )
        else:
            print(
                f"\n警告：在 {self.out_path_train} 中找到了预处理的训练数据。"
                "为了效率，跳过重新处理。如果您的文本输入自上次"
                " `litgpt pretrain` 命令以来发生了变化，请删除预处理文件以触发"
                f"重新处理：`rm -rf {self.out_path_train}`\n"
            )
        use_workers = min(self.num_workers, len(val_files))  # 使用配置的num_workers
        if not Path(self.out_path_val).is_dir():
            print(f"处理验证数据，使用 {use_workers} 个工作者...")
            if self.tokenizer is not None:
                validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=val_files,
                output_dir=str(self.out_path_val),
                num_workers=use_workers,
                chunk_bytes="50MB",
                item_loader=TokensLoader(block_size=self.max_seq_length),
            )
        else:
            print(
                f"\n警告：在 {self.out_path_val} 中找到了预处理的验证数据。"
                "为了效率，跳过重新处理。如果您的文本输入自上次"
                " `litgpt pretrain` 命令以来发生了变化，请删除预处理文件以触发"
                f"重新处理：`rm -rf {self.out_path_val}`\n"
            )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=str(self.out_path_train),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
        )

        pad_id = int(self.tokenizer.eos_id) if self.tokenizer is not None and self.tokenizer.eos_id is not None else 0
        collate_fn = partial(
            collate_fixed_length,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            pad_id=pad_id,
        )
        train_dataloader = StreamingDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=str(self.out_path_val),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
        )
        pad_id = int(self.tokenizer.eos_id) if self.tokenizer is not None and self.tokenizer.eos_id is not None else 0
        collate_fn = partial(
            collate_fixed_length,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            pad_id=pad_id,
        )
        val_dataloader = StreamingDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )
        return val_dataloader


def tokenize(filename: str, tokenizer: Optional[Tokenizer]):
    if tokenizer is None:
        raise ValueError("Tokenizer为None。如果您通过`litgpt pretrain`使用此数据模块，请提供有效的`--tokenizer_dir`路径。")
    with open(filename, encoding="utf-8") as file:
        text = file.read()
    text = text.strip()
    yield tokenizer.encode(text, bos=True, eos=False)


def collate_fixed_length(batch: list, *, batch_size: int, max_seq_length: int, pad_id: int) -> torch.Tensor:
    output = torch.full((batch_size, max_seq_length), pad_id, dtype=torch.long)
    if not batch:
        return output
    for row, item in enumerate(batch[:batch_size]):
        tokens = item
        if isinstance(tokens, (tuple, list)) and len(tokens) == 1 and not isinstance(tokens[0], (int, bool)):
            tokens = tokens[0]
        tokens_tensor = torch.as_tensor(tokens, dtype=torch.long).flatten()
        if tokens_tensor.numel() == 0:
            continue
        tokens_tensor = tokens_tensor[:max_seq_length]
        output[row, : tokens_tensor.numel()] = tokens_tensor
    return output


def validate_tokenizer(tokenizer: Tokenizer) -> None:
    if tokenizer is None:
        raise ValueError("Tokenizer为None。如果您通过`litgpt pretrain`使用此数据模块，请提供有效的`--tokenizer_dir`路径。")
