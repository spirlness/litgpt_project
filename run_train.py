import os

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import argparse
import importlib
import time
from pathlib import Path
from typing import cast

import lightning as L
import torch
import torch.nn.functional as F
import yaml
from litgpt import Config, GPT
from litgpt.tokenizer import Tokenizer

from fixed_text_files import FixedTextFiles
from src.utils import patch_gradient_checkpointing


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_optimizer(model: torch.nn.Module, optimizer_cfg: dict) -> torch.optim.Optimizer:
    class_path = optimizer_cfg.get("class_path", "torch.optim.AdamW")
    init_args = optimizer_cfg.get("init_args", {})
    optimizer_cls = resolve_class(class_path)
    return optimizer_cls(model.parameters(), **init_args)


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("step-*/lit_model.pth"))
    if not candidates:
        return None
    return candidates[-1]


def save_checkpoint(
    fabric: L.Fabric,
    out_dir: Path,
    step: int,
    total_tokens: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint_dir = out_dir / f"step-{step:08d}"
    if fabric.is_global_zero:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fabric.barrier()
    fabric.save(
        checkpoint_dir / "lit_model.pth",
        {
            "model": model,
            "optimizer": optimizer,
            "step": step,
            "total_tokens": total_tokens,
        },
    )

    if fabric.is_global_zero:
        import shutil

        all_checkpoints = sorted(out_dir.glob("step-*"))
        for old_checkpoint in all_checkpoints[:-1]:
            if old_checkpoint.is_dir():
                shutil.rmtree(old_checkpoint)


def train(model_cfg_path: Path, train_cfg_path: Path) -> None:
    model_cfg = load_yaml(model_cfg_path)
    train_cfg = load_yaml(train_cfg_path)

    if "model_config" in model_cfg:
        model_cfg = model_cfg["model_config"]

    if "norm_eps" in model_cfg:
        model_cfg["norm_eps"] = float(model_cfg["norm_eps"])

    train_section = train_cfg.get("train", {})
    data_section = train_cfg.get("data", {})
    optimizer_section = train_cfg.get("optimizer", {})
    grad_checkpointing = bool(train_section.get("gradient_checkpointing", False))

    fabric = L.Fabric(
        strategy=train_cfg.get("strategy", "ddp"),
        devices=train_cfg.get("devices", 2),
        num_nodes=train_cfg.get("num_nodes", 1),
        precision=train_cfg.get("precision", "16-mixed"),
    )
    fabric.launch()

    if grad_checkpointing:
        patch_gradient_checkpointing()
        fabric.print("Enabled gradient checkpointing via Block.forward patch")

    out_dir = Path(train_cfg.get("out_dir", "checkpoints"))
    tokenizer_dir = Path(train_cfg.get("tokenizer_dir", "data/tokenizer"))

    if fabric.is_global_zero:
        out_dir.mkdir(parents=True, exist_ok=True)
    fabric.barrier()

    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_dir}. Run prepare_data.py first.")

    config = Config(**model_cfg)
    model = GPT(config)
    max_seq_length = int(train_section.get("max_seq_length", config.block_size))
    model.max_seq_length = max_seq_length

    optimizer = build_optimizer(model, optimizer_section)

    model, optimizer = fabric.setup(model, optimizer)

    train_data_path = data_section.get("init_args", {}).get("train_data_path")
    val_data_path = data_section.get("init_args", {}).get("val_data_path")
    num_workers = int(data_section.get("init_args", {}).get("num_workers", 2))

    if train_data_path is None:
        raise ValueError("train_data_path is required in configs/kaggle_t4_ddp.yaml")

    data = FixedTextFiles(
        train_data_path=Path(train_data_path),
        val_data_path=Path(val_data_path) if val_data_path else None,
        num_workers=num_workers,
    )

    tokenizer = Tokenizer(tokenizer_dir)

    micro_batch_size = int(train_section.get("micro_batch_size", 1))
    global_batch_size = int(train_section.get("global_batch_size", micro_batch_size * fabric.world_size))

    data.connect(tokenizer=tokenizer, batch_size=micro_batch_size, max_seq_length=max_seq_length)
    if fabric.is_global_zero:
        data.prepare_data()
    fabric.barrier()

    train_dataloader = data.train_dataloader()
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    world_size = fabric.world_size
    denom = micro_batch_size * world_size
    if global_batch_size % denom != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by micro_batch_size * world_size ({denom})."
        )
    grad_accum_steps = global_batch_size // denom

    max_tokens = int(train_section.get("max_tokens", 0))
    log_interval = int(train_section.get("log_interval", 1))
    save_interval = int(train_section.get("save_interval", 0))
    max_norm = float(train_section.get("max_norm", 0.0))
    lr_warmup_steps = int(train_section.get("lr_warmup_steps", 0))

    base_lrs = [group["lr"] for group in optimizer.param_groups]

    start_step = 0
    total_tokens = 0
    resume = train_cfg.get("resume", None)
    if resume:
        ckpt_path: Path | None = None
        if isinstance(resume, str) and resume.lower() == "auto":
            ckpt_path = find_latest_checkpoint(out_dir)
        else:
            ckpt_path = Path(resume)

        if ckpt_path and ckpt_path.exists():
            state = {"model": model, "optimizer": optimizer, "step": 0, "total_tokens": 0}
            fabric.load(ckpt_path, state)
            start_step = int(state.get("step", 0))
            total_tokens = int(state.get("total_tokens", 0))
            fabric.print(f"Resumed from {ckpt_path} (step={start_step}, tokens={total_tokens})")

    fabric.print(
        " | ".join(
            [
                f"Model={config.name}",
                f"Devices={world_size}",
                f"MicroBS={micro_batch_size}",
                f"GlobalBS={global_batch_size}",
                f"Accum={grad_accum_steps}",
                f"MaxSeq={max_seq_length}",
                f"MaxTokens={max_tokens}",
            ]
        )
    )

    model.train()
    data_iter = iter(train_dataloader)
    last_log_time = time.perf_counter()
    last_log_tokens = total_tokens
    global_step = start_step

    while max_tokens <= 0 or total_tokens < max_tokens:
        loss_sum = torch.zeros((), device=fabric.device)
        for micro_step in range(grad_accum_steps):
            try:
                batch = cast(torch.Tensor, next(data_iter))
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = cast(torch.Tensor, next(data_iter))

            input_ids = batch[:, :max_seq_length].contiguous()
            targets = batch[:, 1 : max_seq_length + 1].contiguous()

            sync_gradients = micro_step == grad_accum_steps - 1
            with fabric.no_backward_sync(model, enabled=not sync_gradients):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss_sum += loss.detach()
                fabric.backward(loss / grad_accum_steps)

            total_tokens += input_ids.numel() * world_size

            if max_tokens > 0 and total_tokens >= max_tokens:
                break

        if max_norm > 0:
            fabric.clip_gradients(model, optimizer, max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if lr_warmup_steps > 0 and global_step <= lr_warmup_steps:
            scale = min(global_step / lr_warmup_steps, 1.0)
            for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=False):
                group["lr"] = base_lr * scale

        if log_interval > 0 and global_step % log_interval == 0:
            avg_loss = loss_sum / grad_accum_steps
            avg_loss = cast(torch.Tensor, fabric.all_reduce(avg_loss, reduce_op="mean"))
            now = time.perf_counter()
            elapsed = now - last_log_time
            tokens_delta = total_tokens - last_log_tokens
            tokens_per_sec = tokens_delta / elapsed if elapsed > 0 else 0.0
            lr = optimizer.param_groups[0]["lr"]
            fabric.print(
                f"step={global_step} loss={avg_loss.item():.4f} tokens={total_tokens} tok/s={tokens_per_sec:.0f} lr={lr:.2e}"
            )
            last_log_time = now
            last_log_tokens = total_tokens

        if save_interval > 0 and global_step % save_interval == 0:
            save_checkpoint(fabric, out_dir, global_step, total_tokens, model, optimizer)

        if max_tokens > 0 and total_tokens >= max_tokens:
            break

    save_checkpoint(fabric, out_dir, global_step, total_tokens, model, optimizer)
    if fabric.device.type == "cuda":
        allocated_gib = torch.cuda.max_memory_allocated() / (1024**3)
        reserved_gib = torch.cuda.max_memory_reserved() / (1024**3)
        fabric.print(f"max_memory_allocated_gib={allocated_gib:.3f}")
        fabric.print(f"max_memory_reserved_gib={reserved_gib:.3f}")
    fabric.print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LitGPT Fabric training")
    parser.add_argument("--model-config", type=Path, default=Path("model_config.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/kaggle_t4_ddp.yaml"))
    args = parser.parse_args()

    train(args.model_config, args.train_config)
