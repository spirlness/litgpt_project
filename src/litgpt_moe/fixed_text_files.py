# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer
from torch.utils.data import DataLoader

MANIFEST_FILENAME = ".litgpt_preprocess_manifest.json"


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

    @staticmethod
    def _latest_mtime_ns(paths: list[Path]) -> int:
        latest = 0
        for path in paths:
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                continue
            if mtime > latest:
                latest = mtime
        return latest

    @staticmethod
    def _manifest_path(output_dir: Path) -> Path:
        return output_dir / MANIFEST_FILENAME

    def _split_signature(self, files: list[str]) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(self.max_seq_length).encode("utf-8"))
        if self.tokenizer is not None:
            hasher.update(str(self.tokenizer.vocab_size).encode("utf-8"))
            if self.tokenizer.eos_id is not None:
                hasher.update(str(self.tokenizer.eos_id).encode("utf-8"))
        for item in files:
            path = Path(item)
            stat = path.stat()
            hasher.update(str(path.resolve()).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        return hasher.hexdigest()

    def _split_is_up_to_date(self, output_dir: Path, expected_signature: str, source_files: list[str]) -> bool:
        if not output_dir.is_dir():
            return False

        manifest_path = self._manifest_path(output_dir)
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                return manifest.get("signature") == expected_signature
            except Exception:
                return False

        # Back-compat for older preprocessed dirs without manifest:
        # if output files are newer than source .txt files, keep current artifacts and backfill manifest.
        source_latest = self._latest_mtime_ns([Path(p) for p in source_files])
        output_latest = self._latest_mtime_ns([p for p in output_dir.rglob("*") if p.is_file()])
        return output_latest > 0 and source_latest > 0 and output_latest >= source_latest

    def _write_split_manifest(self, output_dir: Path, signature: str, source_files: list[str]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._manifest_path(output_dir)

        vocab_size = None
        eos_id = None
        if self.tokenizer is not None:
            try:
                vocab_size = int(self.tokenizer.vocab_size)
            except Exception:
                vocab_size = None
            try:
                eos_id = int(self.tokenizer.eos_id) if self.tokenizer.eos_id is not None else None
            except Exception:
                eos_id = None

        payload = {
            "signature": signature,
            "source_file_count": len(source_files),
            "max_seq_length": self.max_seq_length,
            "num_workers": self.num_workers,
            "tokenizer_vocab_size": vocab_size,
            "tokenizer_eos_id": eos_id,
        }
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, sort_keys=True)

    def prepare_data(self) -> None:
        from litdata import optimize
        from litdata.streaming import TokensLoader

        train_files = sorted(
            entry.path
            for entry in os.scandir(self.train_data_path)
            if entry.name.endswith(".txt")
        )
        assert len(train_files) > 0, f"No .txt files found in training data {self.train_data_path}"

        if self.val_data_path is not None:
            self.val_data_path = Path(self.val_data_path)
            val_files = sorted(
                entry.path
                for entry in os.scandir(self.val_data_path)
                if entry.name.endswith(".txt")
            )
            assert len(val_files) > 0, f"No .txt files found in validation data {self.val_data_path}"
        # Train/Test split. Use chunk 0 as test split, rest as training
        else:
            assert len(train_files) > 1, f"Expected at least two .txt files in {self.train_data_path}"
            val_files, *train_files = train_files
            val_files = [val_files]

        if self.tokenizer is not None:
            validate_tokenizer(self.tokenizer)

        train_signature = self._split_signature(train_files)
        val_signature = self._split_signature(val_files)
        train_up_to_date = self._split_is_up_to_date(Path(self.out_path_train), train_signature, train_files)
        val_up_to_date = self._split_is_up_to_date(Path(self.out_path_val), val_signature, val_files)

        if train_up_to_date and val_up_to_date:
            print(f"\nSkipping data preprocessing: Up-to-date artifacts found in {self.out_path_train} and {self.out_path_val}.\n")
            self._write_split_manifest(Path(self.out_path_train), train_signature, train_files)
            self._write_split_manifest(Path(self.out_path_val), val_signature, val_files)
            return

        use_workers = self.num_workers
        if not train_up_to_date:
            if Path(self.out_path_train).is_dir():
                print(f"Detected stale training artifacts in {self.out_path_train}, rebuilding...")
                shutil.rmtree(self.out_path_train, ignore_errors=True)
            print(f"Processing training data using {use_workers} workers...")
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=train_files,
                output_dir=str(self.out_path_train),
                num_workers=use_workers,
                chunk_bytes="50MB",
                item_loader=TokensLoader(block_size=self.max_seq_length),
            )
            self._write_split_manifest(Path(self.out_path_train), train_signature, train_files)
        else:
            print(f"Training artifacts are current in {self.out_path_train}; skipping train rebuild.")
            self._write_split_manifest(Path(self.out_path_train), train_signature, train_files)

        if not val_up_to_date:
            if Path(self.out_path_val).is_dir():
                print(f"Detected stale validation artifacts in {self.out_path_val}, rebuilding...")
                shutil.rmtree(self.out_path_val, ignore_errors=True)
            print(f"Processing validation data using {use_workers} workers...")
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=val_files,
                output_dir=str(self.out_path_val),
                num_workers=use_workers,
                chunk_bytes="50MB",
                item_loader=TokensLoader(block_size=self.max_seq_length),
            )
            self._write_split_manifest(Path(self.out_path_val), val_signature, val_files)
        else:
            print(f"Validation artifacts are current in {self.out_path_val}; skipping val rebuild.")
            self._write_split_manifest(Path(self.out_path_val), val_signature, val_files)

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
        raise ValueError("Tokenizer is None. If using this One `litgpt pretrain`, please provide a valid `--tokenizer_dir` path.")
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
        raise ValueError("Tokenizer is None. If using this One `litgpt pretrain`, please provide a valid `--tokenizer_dir` path.")
