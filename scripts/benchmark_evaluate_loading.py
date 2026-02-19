import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Define the datasets
class ManualLoading:
    def __init__(self, file_path, block_size, batch_size):
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_start = len(self.data) - self.block_size - 2

    def get_batch(self):
        ix = torch.randint(0, self.max_start + 1, (self.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        return x, y

class StandardDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64))
        return x, y

class OptimizedDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Zero-copy optimization
        # View as int16 (no copy), create tensor (no copy from numpy), convert to long (copy), mask
        data_slice = self.data[idx : idx + self.block_size + 1]
        data_view = data_slice.view(np.int16)
        t = torch.from_numpy(data_view)
        t = t.to(torch.long)
        t = t & 0xFFFF
        x = t[:-1]
        y = t[1:]
        return x, y

def benchmark(mode, file_path, block_size, batch_size, num_workers, max_batches):
    print(f"Benchmarking mode: {mode}")

    if mode == "manual":
        loader = ManualLoading(file_path, block_size, batch_size)
        start_time = time.time()
        for _ in range(max_batches):
            x, y = loader.get_batch()
            # Simulate device transfer overhead (optional, but keep it minimal to focus on loading)
            # x, y = x.cuda(), y.cuda()
        end_time = time.time()

    else:
        if mode == "standard":
            dataset = StandardDataset(file_path, block_size)
        elif mode == "optimized":
            dataset = OptimizedDataset(file_path, block_size)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=True
        )

        iter_loader = iter(dataloader)

        # Warmup workers
        try:
             next(iter_loader)
        except StopIteration:
             pass

        start_time = time.time()
        for _ in range(max_batches):
            try:
                x, y = next(iter_loader)
            except StopIteration:
                iter_loader = iter(dataloader)
                x, y = next(iter_loader)
        end_time = time.time()

    duration = end_time - start_time
    throughput = max_batches / duration
    print(f"Time: {duration:.4f}s, Throughput: {throughput:.2f} batches/s")
    return throughput

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/custom_text/val/val.bin")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Generating dummy data at {file_path}...")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Create 10MB file
        data = np.random.randint(0, 65535, size=5 * 1024 * 1024, dtype=np.uint16)
        data.tofile(file_path)

    print("Warming up...")
    # Warmup
    benchmark("manual", args.file_path, args.block_size, args.batch_size, 0, 10)

    print("-" * 40)
    res_manual = benchmark("manual", args.file_path, args.block_size, args.batch_size, 0, args.max_batches)
    res_standard = benchmark("standard", args.file_path, args.block_size, args.batch_size, args.num_workers, args.max_batches)
    res_optimized = benchmark("optimized", args.file_path, args.block_size, args.batch_size, args.num_workers, args.max_batches)

    print("-" * 40)
    print(f"Manual vs Standard Speedup: {res_standard / res_manual:.2f}x")
    print(f"Standard vs Optimized Speedup: {res_optimized / res_standard:.2f}x")
    print(f"Manual vs Optimized Speedup: {res_optimized / res_manual:.2f}x")
