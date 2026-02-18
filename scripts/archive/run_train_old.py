import argparse
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Literal, Union
from unittest.mock import patch

import requests
import torch
import yaml

# from litgpt.data import TextFiles
from fixed_text_files import FixedTextFiles as TextFiles
from litgpt.args import LogArgs, TrainArgs
from litgpt.config import Config
from src.utils import (
    apply_runtime_config,
    configure_flash_attention,
    patch_cudagraph_for_compile,
    patch_flops_measurement,
    patch_gradient_checkpointing,
    restore_gradient_checkpointing,
    start_progress_bar,
    verify_flash_attention,
)


def load_configs(model_config_path: Path, train_config_path: Path):
    with open(model_config_path, "r") as f:
        model_cfg = yaml.safe_load(f)
    with open(train_config_path, "r") as f:
        train_cfg = yaml.safe_load(f)
    return model_cfg, train_cfg


def create_compile_context(
    use_compile: bool,
    mode: Literal["default", "reduce-overhead", "max-autotune"],
    dynamic: bool,
    fullgraph: bool,
):
    if not use_compile:
        # When compile is disabled, patch torch.compile to be a no-op
        # This ensures litgpt.pretrain's internal torch.compile call doesn't execute
        return patch("torch.compile", side_effect=lambda m, *a, **kw: m)

    _orig_compile = torch.compile

    def _custom_compile(model, *args, **kwargs):
        kwargs.setdefault("mode", mode)
        kwargs.setdefault("dynamic", dynamic)
        kwargs.setdefault("fullgraph", fullgraph)
        # We must return the compiled model directly (OptimizedModule) so that
        # fabric.setup() can recognize it as a module. Wrapping it in a function
        # breaks fabric.setup().
        # cudagraph_mark_step_begin is handled via patch_cudagraph_for_compile()
        # which patches the data iterator and validation loop.
        return _orig_compile(model, *args, **kwargs)

    # Patch torch.compile globally
    # This affects litgpt.pretrain since it imports torch at module level
    return patch("torch.compile", side_effect=_custom_compile)


def get_optimizer_config(use_8bit: bool = False):
    if use_8bit:
        import importlib.util

        if importlib.util.find_spec("bitsandbytes") is not None:
            return {"class_path": "bitsandbytes.optim.AdamW8bit", "init_args": {"lr": 0.0003, "weight_decay": 0.01}}

    return {"class_path": "torch.optim.AdamW", "init_args": {"lr": 0.0003, "weight_decay": 0.01}}


def ensure_smoke_text_data(data_dir: Path) -> None:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    tokenized_train_dir = train_dir / "train"
    tokenized_val_dir = val_dir / "val"
    shutil.rmtree(tokenized_train_dir, ignore_errors=True)
    shutil.rmtree(tokenized_val_dir, ignore_errors=True)

    sample_sentence = "Once upon a time, a small model learned to run on a tiny dataset. "
    sample_text = (sample_sentence * 200).strip() + "\n"
    (train_dir / "sample.txt").write_text(sample_text, encoding="utf-8")
    (val_dir / "sample.txt").write_text(sample_text, encoding="utf-8")


def ensure_smoke_tokenizer(tokenizer_dir: Path) -> None:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    config_path = tokenizer_dir / "tokenizer_config.json"
    if tokenizer_path.exists() and config_path.exists():
        return

    tokenizer_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json"
    tokenizer_config_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer_config.json"

    if not tokenizer_path.exists():
        response = requests.get(tokenizer_url, stream=True, timeout=30)
        response.raise_for_status()
        with tokenizer_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    if not config_path.exists():
        response = requests.get(tokenizer_config_url, stream=True, timeout=30)
        response.raise_for_status()
        with config_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LitGPT MoE model")
    parser.add_argument("--model-config", type=Path, default=Path("model_config.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("train_config.yaml"))
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--precision", type=str)
    parser.add_argument("--resume", type=str, default=os.environ.get("RESUME", "auto"))
    parser.add_argument("--optimizer-8bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-mode", type=str, choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-fullgraph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--flash-attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--flash-attention-force", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--logger", type=str, choices=["csv", "wandb", "tensorboard"])
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a minimal CPU-friendly training sanity check with tiny data/tokenizer.",
    )

    args, _ = parser.parse_known_args()

    model_cfg_raw, train_cfg_raw = load_configs(args.model_config, args.train_config)

    if "norm_eps" in model_cfg_raw:
        model_cfg_raw["norm_eps"] = float(model_cfg_raw["norm_eps"])

    if args.micro_batch_size:
        train_cfg_raw["train"]["micro_batch_size"] = args.micro_batch_size
    if args.global_batch_size:
        train_cfg_raw["train"]["global_batch_size"] = args.global_batch_size
    if args.max_tokens:
        train_cfg_raw["train"]["max_tokens"] = args.max_tokens
    if args.max_seq_length:
        train_cfg_raw["train"]["max_seq_length"] = args.max_seq_length
    if args.logger:
        train_cfg_raw["logger_name"] = args.logger
    if args.out_dir:
        train_cfg_raw["out_dir"] = str(args.out_dir)
    if args.tokenizer_dir:
        train_cfg_raw["tokenizer_dir"] = str(args.tokenizer_dir)
    if args.precision:
        train_cfg_raw["precision"] = args.precision

    if args.smoke_test:
        train_cfg_raw["train"]["micro_batch_size"] = 1
        train_cfg_raw["train"]["global_batch_size"] = 1
        train_cfg_raw["train"]["max_tokens"] = 64
        train_cfg_raw["train"]["max_seq_length"] = 64
        train_cfg_raw["precision"] = "32-true"
        train_cfg_raw["logger_name"] = "csv"
        smoke_out_dir = Path("checkpoints_smoke")
        shutil.rmtree(smoke_out_dir, ignore_errors=True)
        train_cfg_raw["out_dir"] = str(smoke_out_dir)
        train_cfg_raw["tokenizer_dir"] = str(Path("data/tokenizer"))
        args.optimizer_8bit = False
        ensure_smoke_text_data(Path("data/custom_text"))
        ensure_smoke_tokenizer(Path(train_cfg_raw["tokenizer_dir"]))

    grad_checkpointing = args.gradient_checkpointing
    if grad_checkpointing is None:
        grad_checkpointing = train_cfg_raw["train"].get("gradient_checkpointing", False)

    disable_ckpt_env = os.environ.get("DISABLE_GRADIENT_CHECKPOINTING", "").lower()
    if disable_ckpt_env in ("1", "true", "yes", "on") and grad_checkpointing:
        print("DISABLE_GRADIENT_CHECKPOINTING set; forcing gradient checkpointing OFF")
        grad_checkpointing = False

    train_cfg_raw["train"].pop("gradient_checkpointing", None)

    opt_cfg = train_cfg_raw.get("optimization", {})

    # Check environment variable for compile override
    env_compile = os.environ.get("TORCH_COMPILE")
    if env_compile is not None:
        env_compile = env_compile.lower() in ("1", "true", "yes", "on")

    if args.smoke_test:
        use_compile = False
    elif args.compile is not None:
        use_compile = args.compile
    elif env_compile is not None:
        use_compile = env_compile
    else:
        use_compile = opt_cfg.get("compile", False)

    print(f"Compilation {'ENABLED' if use_compile else 'DISABLED'}", flush=True)

    compile_mode = args.compile_mode or opt_cfg.get("compile_mode", "default")
    compile_dynamic = args.compile_dynamic if args.compile_dynamic is not None else opt_cfg.get("compile_dynamic", False)
    compile_fullgraph = (
        args.compile_fullgraph if args.compile_fullgraph is not None else opt_cfg.get("compile_fullgraph", False)
    )
    if args.smoke_test:
        use_flash_attention = False
        flash_attention_force = False
    else:
        use_flash_attention = (
            args.flash_attention if args.flash_attention is not None else opt_cfg.get("flash_attention", False)
        )
        flash_attention_force = (
            args.flash_attention_force
            if args.flash_attention_force is not None
            else opt_cfg.get("flash_attention_force", False)
        )
    disable_math_fallback = opt_cfg.get("disable_math_fallback", False)

    configure_flash_attention(enable=True, disable_math_fallback=disable_math_fallback)

    if use_flash_attention or flash_attention_force:
        verify_flash_attention(force=flash_attention_force, verbose=True)

    apply_runtime_config()
    patch_flops_measurement()

    if grad_checkpointing:
        patch_gradient_checkpointing()
        print("Enabled gradient checkpointing via Block.forward patch")
    else:
        restore_gradient_checkpointing()

    model_config = Config(**model_cfg_raw)

    data_module = TextFiles(
        train_data_path=Path(train_cfg_raw["data"]["init_args"]["train_data_path"]),
        val_data_path=Path(train_cfg_raw["data"]["init_args"]["val_data_path"]),
        num_workers=train_cfg_raw["data"]["init_args"].get("num_workers", 2),
    )

    train_args = TrainArgs(**train_cfg_raw["train"])

    out_dir = Path(train_cfg_raw.get("out_dir", "./checkpoints"))
    sys.argv = [
        sys.argv[0],
        model_cfg_raw["name"],
        "--out_dir",
        str(out_dir),
        "--precision",
        train_cfg_raw.get("precision", "bf16-mixed"),
        "--tokenizer_dir",
        train_cfg_raw.get("tokenizer_dir", "./data/tokenizer"),
        "--train.global_batch_size",
        str(train_args.global_batch_size),
        "--train.micro_batch_size",
        str(train_args.micro_batch_size),
        "--train.max_tokens",
        str(train_args.max_tokens),
        "--train.max_seq_length",
        str(train_args.max_seq_length),
    ]

    resume_val: Union[bool, str, Path] = False
    if args.resume:
        if args.resume.lower() == "auto":
            resume_val = "auto"
        elif args.resume.lower() in ["true", "1", "yes"]:
            resume_val = True
        elif args.resume.lower() in ["false", "0", "no"]:
            resume_val = False
        else:
            p = Path(args.resume)
            if p.is_dir():
                if (p / "lit_model.pth").exists():
                    resume_val = p / "lit_model.pth"
                else:
                    files = list(p.glob("**/*.pth"))
                    if files:
                        resume_val = sorted(files)[-1]
                    else:
                        resume_val = p
            else:
                resume_val = p
        sys.argv.extend(["--resume", str(args.resume)])

    log_args = LogArgs()
    if train_cfg_raw["logger_name"] == "wandb":
        log_args = LogArgs(project=os.environ.get("WANDB_PROJECT", "moe-training"))

    stop_event = threading.Event()
    monitor_thread, bar = None, None
    if args.progress:
        monitor_thread, bar = start_progress_bar(
            out_dir=out_dir,
            total_tokens=train_args.max_tokens or 0,
            stop=stop_event,
        )

    try:
        if use_compile:
            patch_cudagraph_for_compile()

        # CRITICAL FIX: Apply compile context BEFORE importing/using litgpt.pretrain
        # This ensures torch.compile is patched when litgpt.pretrain.setup() runs
        compile_ctx = create_compile_context(
            use_compile=use_compile,
            mode=compile_mode,
            dynamic=compile_dynamic,
            fullgraph=compile_fullgraph,
        )
        with compile_ctx:
            from litgpt.pretrain import setup

            setup(
                model_name=model_cfg_raw["name"],
                model_config=model_config,
                out_dir=out_dir,
                precision=train_cfg_raw.get("precision", "bf16-mixed"),
                tokenizer_dir=Path(train_cfg_raw.get("tokenizer_dir", "./data/tokenizer")),
                data=data_module,
                train=train_args,
                logger_name=train_cfg_raw["logger_name"],
                log=log_args,
                optimizer=get_optimizer_config(args.optimizer_8bit),
                resume=resume_val,
                devices=train_cfg_raw.get("devices", "auto"),
                num_nodes=train_cfg_raw.get("num_nodes", 1),
            )
    finally:
        stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=3.0)
        if bar:
            bar.close()
