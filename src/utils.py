import os
import torch
import threading
import time
import torch.utils.checkpoint
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def apply_patches():
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ["PYTORCH_ALLOC_CONF"])
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    try:
        import torch.fx.experimental._config as fx_config

        fx_config.meta_nonzero_assume_all_nonzero = True
    except (ImportError, AttributeError):
        pass

    try:
        import lightning.fabric.utilities.throughput as throughput_module

        _orig_measure_flops = throughput_module.measure_flops

        def _measure_flops_patch(model, forward_fn, loss_fn, *args, **kwargs):
            try:
                return _orig_measure_flops(model, forward_fn, loss_fn, *args, **kwargs)
            except (NotImplementedError, AttributeError):
                return 0.0

        throughput_module.measure_flops = _measure_flops_patch
    except ImportError:
        pass

    try:
        import lightning.fabric.plugins.io.torch_io as torch_io_module

        def _non_atomic_save(checkpoint, path) -> None:
            torch.save(checkpoint, path)

        setattr(torch_io_module, "_atomic_save", _non_atomic_save)
    except Exception:
        pass


def patch_gradient_checkpointing():
    from litgpt.model import Block

    _orig_block_forward = Block.forward

    def _checkpointed_block_forward(self, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(_orig_block_forward, self, *args, **kwargs, use_reentrant=False)

    setattr(Block, "forward", _checkpointed_block_forward)


def get_latest_metrics_csv(out_dir: Path) -> Optional[Path]:
    logs_dir = (out_dir / "logs").resolve()
    if not logs_dir.exists():
        return None
    candidates = list(logs_dir.rglob("metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_last_total_tokens(metrics_csv: Path) -> Optional[int]:
    try:
        with metrics_csv.open("r", encoding="utf-8") as f:
            header = f.readline().strip("\n\r")
            if not header:
                return None
            columns = header.split(",")
            idx = None
            for name in ("total_tokens", "tokens"):
                if name in columns:
                    idx = columns.index(name)
                    break
            if idx is None:
                return None
            last = None
            for line in f:
                line = line.strip("\n\r")
                if line:
                    last = line
            if not last:
                return None
            parts = last.split(",")
            if idx >= len(parts) or not parts[idx]:
                return None
            return int(float(parts[idx]))
    except Exception:
        return None


def start_progress_bar(*, out_dir: Path, total_tokens: int, stop: threading.Event) -> tuple[threading.Thread, tqdm]:
    bar = tqdm(total=total_tokens, desc="Training", unit="tok", dynamic_ncols=True)

    def _worker() -> None:
        last = 0
        while not stop.is_set():
            metrics = get_latest_metrics_csv(out_dir)
            if metrics is not None:
                current = read_last_total_tokens(metrics)
                if current is not None and current > last:
                    bar.update(min(current, total_tokens) - last)
                    last = min(current, total_tokens)
                    if last >= total_tokens:
                        break
            time.sleep(1.0)

    thread = threading.Thread(target=_worker, name="progress-monitor", daemon=True)
    thread.start()
    return thread, bar
