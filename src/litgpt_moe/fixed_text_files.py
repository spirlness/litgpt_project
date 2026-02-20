# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
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
    """
    Fixed version of TextFiles DataModule for pretraining.

    Reads text data from .txt files in the data directory,
    and returns tokenized batches for training and validation data loaders.
    Each sample is set to a fixed length.
    """

    train_data_path: Path
    """Path to the training data directory containing .txt files."""
    val_data_path: Optional[Path] = None
    """Path to the validation data directory containing .txt files.
    If None, verification data will be split from the training set."""
    seed: int = 42
    """Random seed for shuffling the dataset."""
    num_workers: int = 2  # Force usage of fewer workers
    """Number of workers for data loading."""

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
        self.max_seq_length = (max_seq_length or -1) + 1  # Add 1 because we need the next token

    def prepare_data(self) -> None:
        from litdata import optimize
        from litdata.streaming import TokensLoader

        # Check if preprocessed data already exists
        if Path(self.out_path_train).is_dir() and Path(self.out_path_val).is_dir():
            print(
                f"\nSkipping data preprocessing: Found preprocessed data in {self.out_path_train} and {self.out_path_val}.\n"
            )
            return

        train_files = sorted(entry.path for entry in os.scandir(self.train_data_path) if entry.name.endswith(".txt"))
        assert len(train_files) > 0, f"No .txt files found in training data {self.train_data_path}"

        if self.val_data_path is not None:
            self.val_data_path = Path(self.val_data_path)
            val_files = sorted(entry.path for entry in os.scandir(self.val_data_path) if entry.name.endswith(".txt"))
            assert len(val_files) > 0, f"No .txt files found in validation data {self.val_data_path}"
        # Train/Test split. Use chunk 0 as test split, rest as training
        else:
            assert len(train_files) > 1, f"Expected at least two .txt files in {self.train_data_path}"
            val_files, *train_files = train_files
            val_files = [val_files]

        # Use the number of workers specified in the config
        use_workers = self.num_workers
        if not Path(self.out_path_train).is_dir():
            print(f"Processing training data using {use_workers} workers...")
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
                f"\nWarning: Found preprocessed training data in {self.out_path_train}."
                "Skipping reprocessing for efficiency. If your text input has changed since the last"
                " `litgpt pretrain` command, please delete the preprocessed files to trigger"
                f" reprocessing: `rm -rf {self.out_path_train}`\n"
            )
        use_workers = self.num_workers
        if not Path(self.out_path_val).is_dir():
            print(f"Processing validation data using {use_workers} workers...")
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
                f"\nWarning: Found preprocessed validation data in {self.out_path_val}."
                "Skipping reprocessing for efficiency. If your text input has changed since the last"
                " `litgpt pretrain` command, please delete the preprocessed files to trigger"
                f" reprocessing: `rm -rf {self.out_path_val}`\n"
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
        raise ValueError(
            "Tokenizer is None. If using this One `litgpt pretrain`, please provide a valid `--tokenizer_dir` path."
        )
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
        raise ValueError(
            "Tokenizer is None. If using this One `litgpt pretrain`, please provide a valid `--tokenizer_dir` path."
        )
