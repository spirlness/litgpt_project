import importlib.metadata
import importlib.util
import os
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm import tqdm


class FlashAttentionInfo:
    def __init__(
        self,
        available: bool,
        compute_capability: tuple[int, int],
        flash_attn_version: Optional[str] = None,
        sdpa_backends: Optional[dict[str, bool]] = None,
        reason: Optional[str] = None,
    ):
        self.available = available
        self.compute_capability = compute_capability
        self.flash_attn_version = flash_attn_version
        self.sdpa_backends = sdpa_backends or {}
        self.reason = reason

    def __repr__(self) -> str:
        return (
            f"FlashAttentionInfo(available={self.available}, "
            f"compute_capability={self.compute_capability}, "
            f"flash_attn_version={self.flash_attn_version!r})"
        )


def check_flash_attention() -> FlashAttentionInfo:
    if not torch.cuda.is_available():
        return FlashAttentionInfo(
            available=False,
            compute_capability=(0, 0),
            reason="CUDA not available",
        )

    major, minor = torch.cuda.get_device_capability()
    compute_cap = (major, minor)

    if major < 8:
        return FlashAttentionInfo(
            available=False,
            compute_capability=compute_cap,
            reason=f"GPU compute capability {major}.{minor} < 8.0 (Ampere required)",
        )

    flash_attn_version = None
    if importlib.util.find_spec("flash_attn") is not None:
        try:
            flash_attn_version = importlib.metadata.version("flash-attn")
        except importlib.metadata.PackageNotFoundError:
            flash_attn_version = "unknown"

    sdpa_backends: dict[str, bool] = {}
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        test_q = torch.randn(1, 1, 32, 64, device="cuda", dtype=torch.bfloat16)

        for backend_name, backend in [
            ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
            ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
            ("MATH", SDPBackend.MATH),
        ]:
            try:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(test_q, test_q, test_q, is_causal=True)
                sdpa_backends[backend_name] = True
            except Exception:
                sdpa_backends[backend_name] = False
    except ImportError:
        sdpa_backends["FLASH_ATTENTION"] = False

    flash_available = sdpa_backends.get("FLASH_ATTENTION", False)

    return FlashAttentionInfo(
        available=flash_available,
        compute_capability=compute_cap,
        flash_attn_version=flash_attn_version,
        sdpa_backends=sdpa_backends,
        reason=None if flash_available else "FLASH_ATTENTION backend test failed",
    )


def verify_flash_attention(force: bool = False, verbose: bool = True) -> FlashAttentionInfo:
    info = check_flash_attention()

    if verbose:
        print("[Flash Attention Check]")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  Compute Capability: {info.compute_capability[0]}.{info.compute_capability[1]}")
        else:
            print("  CUDA: Not available")

        if info.flash_attn_version:
            print(f"  flash_attn package: v{info.flash_attn_version}")
        else:
            print("  flash_attn package: Not installed (using PyTorch SDPA)")

        if info.sdpa_backends:
            print("  SDPA Backends:")
            for backend, available in info.sdpa_backends.items():
                status = "Available" if available else "Not available"
                print(f"    - {backend}: {status}")

        if info.available:
            print("  Status: Flash Attention 2 ENABLED")
        else:
            print(f"  Status: Flash Attention 2 NOT available ({info.reason})")

    if force and not info.available:
        raise RuntimeError(
            f"Flash Attention 2 is required but not available: {info.reason}. "
            f"Compute capability: {info.compute_capability[0]}.{info.compute_capability[1]}. "
            "Ensure you have an Ampere+ GPU and PyTorch with CUDA support."
        )

    return info


def configure_flash_attention(enable: bool = True, disable_math_fallback: bool = False) -> None:
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.enable_flash_sdp(enable)
    torch.backends.cuda.enable_mem_efficient_sdp(enable)

    if disable_math_fallback:
        torch.backends.cuda.enable_math_sdp(False)
    else:
        torch.backends.cuda.enable_math_sdp(True)


def apply_runtime_config() -> None:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ["PYTORCH_ALLOC_CONF"])
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def patch_flops_measurement() -> None:
    """Patch FLOPs measurement to handle MoE routing ops with meta tensors.

    MoE routing uses torch.where/torch.nonzero which don't have proper meta tensor
    implementations. This patch sets the experimental config flag to allow meta tensor
    operations to assume all elements are non-zero, enabling FLOPs measurement to work.

    If that fails, falls back to returning 0.0 FLOPs rather than crashing.
    """
    # Enable meta tensor support for torch.nonzero (used by MoE routing)
    try:
        import torch.fx.experimental._config as fx_config

        fx_config.meta_nonzero_assume_all_nonzero = True
    except (ImportError, AttributeError):
        pass

    # Wrap measure_flops to catch any remaining errors and return 0.0 instead of crashing
    try:
        import lightning.fabric.utilities.throughput as throughput_module

        _orig_measure_flops = throughput_module.measure_flops

        def _measure_flops_patch(model, forward_fn, loss_fn, *args, **kwargs):
            try:
                return _orig_measure_flops(model, forward_fn, loss_fn, *args, **kwargs)
            except (NotImplementedError, AttributeError, RuntimeError):
                return 0.0

        throughput_module.measure_flops = _measure_flops_patch
    except ImportError:
        pass


def patch_cudagraph_for_compile() -> None:
    if not hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        return

    import litgpt.pretrain
    import litgpt.utils

    _orig_cycle_next = litgpt.utils.CycleIterator.__next__

    def _cycle_next(self):
        torch.compiler.cudagraph_mark_step_begin()
        return _orig_cycle_next(self)

    litgpt.utils.CycleIterator.__next__ = _cycle_next

    _orig_validate = litgpt.pretrain.validate

    def _validate(fabric, model, val_dataloader, max_iters, verbose=True):
        fabric.barrier()
        if verbose:
            fabric.print("Validating ...")
        model.eval()

        losses = []
        for k, batch in enumerate(val_dataloader):
            if k >= max_iters:
                break
            torch.compiler.cudagraph_mark_step_begin()
            input_ids = batch[:, 0 : model.max_seq_length].contiguous().long()
            targets = batch[:, 1 : (model.max_seq_length + 1)].contiguous().long()
            logits = model(input_ids)
            loss = litgpt.pretrain.chunked_cross_entropy(logits, targets)
            losses.append(loss)

        if losses:
            val_loss = torch.stack(losses).mean()
        else:
            device = next(model.parameters()).device
            val_loss = torch.tensor(float("nan"), device=device)
        model.train()
        fabric.barrier()
        return val_loss

    litgpt.pretrain.validate = _validate


def patch_gradient_checkpointing() -> None:
    from litgpt.model import Block

    disable_ckpt_env = os.environ.get("DISABLE_GRADIENT_CHECKPOINTING", "").lower()
    if disable_ckpt_env in ("1", "true", "yes", "on"):
        if hasattr(Block, "_orig_forward"):
            setattr(Block, "forward", getattr(Block, "_orig_forward"))
        setattr(Block, "_ckpt_patched", False)
        print("Gradient checkpointing patch skipped due to DISABLE_GRADIENT_CHECKPOINTING env")
        return

    if getattr(Block, "_ckpt_patched", False):
        return

    _orig_block_forward = Block.forward
    setattr(Block, "_orig_forward", _orig_block_forward)

    def _checkpointed_block_forward(self, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(_orig_block_forward, self, *args, **kwargs, use_reentrant=False)

    setattr(Block, "forward", _checkpointed_block_forward)
    setattr(Block, "_ckpt_patched", True)


def restore_gradient_checkpointing() -> None:
    from litgpt.model import Block

    orig_forward = getattr(Block, "_orig_forward", None)
    if orig_forward is not None:
        setattr(Block, "forward", orig_forward)
    setattr(Block, "_ckpt_patched", False)


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
