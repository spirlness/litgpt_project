import sys
import os
import argparse
import threading
import time
from typing import Optional

# Helps reduce CUDA allocator fragmentation (must be set before importing torch)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# Back-compat alias used by some docs/tools
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ["PYTORCH_ALLOC_CONF"])

import torch
from pathlib import Path

from tqdm import tqdm

# Fix for MoE meta-device FLOP counting
try:
    import torch.fx.experimental._config as fx_config
    fx_config.meta_nonzero_assume_all_nonzero = True
    print("Enabled meta_nonzero_assume_all_nonzero for MoE FLOP counting.")
except (ImportError, AttributeError) as e:
    print(f"Could not set meta_nonzero_assume_all_nonzero: {e}")

# Patch measure_flops to skip for MoE models
import lightning.fabric.utilities.throughput as throughput_module
_orig_measure_flops = throughput_module.measure_flops

def _measure_flops_patch(model, forward_fn, loss_fn, *args, **kwargs):
    try:
        return _orig_measure_flops(model, forward_fn, loss_fn, *args, **kwargs)
    except (NotImplementedError, AttributeError) as e:
        print(f"FLOPs measurement not supported with MoE: {e}")
        return 0.0  # Return 0 flops

throughput_module.measure_flops = _measure_flops_patch

from litgpt.config import Config
from litgpt.pretrain import setup
from litgpt.args import TrainArgs, EvalArgs
from litgpt.data import TextFiles


def _resolve_wandb_artifact_ref(ref: str) -> str:
    # Accept:
    # - entity/project/artifact:alias
    # - project/artifact:alias (uses WANDB_ENTITY if set)
    parts = ref.split("/")
    if len(parts) >= 3:
        return ref
    if len(parts) == 2:
        entity = os.environ.get("WANDB_ENTITY")
        if not entity:
            raise ValueError(
                "W&B artifact reference must include entity (entity/project/name:alias) "
                "or set WANDB_ENTITY."
            )
        return f"{entity}/{ref}"
    raise ValueError(
        "Invalid W&B artifact reference. Expected entity/project/name:alias or project/name:alias."
    )


def _download_dataset_from_wandb(*, artifact_ref: str, root_dir: Path) -> Path:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("wandb is required to download dataset artifacts") from exc

    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    download_dir = Path(artifact.download(root=str(root_dir))).resolve()
    data_dir = download_dir / "custom_text"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Downloaded artifact does not contain expected 'custom_text' directory: {data_dir}"
        )
    return data_dir


def _latest_metrics_csv(out_dir: Path) -> Optional[Path]:
    logs_dir = (out_dir / "logs").resolve()
    if not logs_dir.exists():
        return None
    candidates = list(logs_dir.rglob("metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_last_total_tokens(metrics_csv: Path) -> Optional[int]:
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


def _start_progress_bar(*, out_dir: Path, total_tokens: int, stop: threading.Event) -> tuple[threading.Thread, tqdm]:
    bar = tqdm(total=total_tokens, desc="Training", unit="tok", dynamic_ncols=True)

    def _worker() -> None:
        last = 0
        while not stop.is_set():
            metrics = _latest_metrics_csv(out_dir)
            if metrics is not None:
                current = _read_last_total_tokens(metrics)
                if current is not None and current > last:
                    bar.update(min(current, total_tokens) - last)
                    last = min(current, total_tokens)
                    if last >= total_tokens:
                        break
            time.sleep(1.0)

    thread = threading.Thread(target=_worker, name="progress-monitor", daemon=True)
    thread.start()
    return thread, bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LitGPT MoE model")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(os.environ.get("DATASET_DIR", "./data/custom_text")),
        help="Local dataset root directory containing train/ and val/",
    )
    parser.add_argument(
        "--wandb-dataset",
        type=str,
        default=os.environ.get("WANDB_DATASET_ARTIFACT"),
        help=(
            "Optional W&B dataset Artifact reference to download and use for training. "
            "Examples: entity/project/dataset-custom_text:latest or project/dataset-custom_text:latest (needs WANDB_ENTITY)."
        ),
    )
    parser.add_argument(
        "--wandb-artifacts-dir",
        type=Path,
        default=Path(os.environ.get("WANDB_ARTIFACTS_DIR", "./data/wandb_artifacts")),
        help="Local cache directory for downloaded W&B artifacts",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=os.environ.get("RESUME"),
        help="Resume from checkpoint: 'auto', 'true', or a checkpoint directory path",
    )

    # Memory-related knobs (safe defaults for 6GB GPUs)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=int(os.environ.get("MICRO_BATCH_SIZE", "1")),
        help="Per-device micro batch size (lower to reduce CUDA memory)",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=int(os.environ.get("GLOBAL_BATCH_SIZE", "8")),
        help="Global batch size (effective batch via gradient accumulation)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("MAX_TOKENS", "320000")),
        help="Training budget in tokens (lower for quick sanity runs)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=int(os.environ.get("MAX_SEQ_LENGTH", "1024")),
        help="Max sequence length for training (lower to reduce CUDA memory)",
    )

    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a progress bar (reads tokens from CSV logger)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile (default: False). FixedLLaMAMoE is used to ensure compatibility.",
    )

    # Parse our custom flags.
    # Note: LitGPT internally saves hyperparameters by parsing sys.argv via jsonargparse.
    # We'll set a LitGPT-compatible argv (including the required positional `model_name`) right before calling setup().
    args, _remaining_argv = parser.parse_known_args()
    resume: bool | str | Path = False
    if args.resume:
        value = str(args.resume).strip()
        if value.lower() in {"auto"}:
            resume = "auto"
        elif value.lower() in {"true", "1", "yes", "y"}:
            resume = True
        else:
            resume = Path(value)

    # Use FixedLLaMAMoE to improve compatibility with torch.compile
    try:
        from custom_moe import FixedLLaMAMoE
        import litgpt.model
        litgpt.model.LLaMAMoE = FixedLLaMAMoE
        print("Patched litgpt.model.LLaMAMoE with FixedLLaMAMoE")
    except ImportError:
        print("Warning: Could not import FixedLLaMAMoE, using default LLaMAMoE")

    class TorchCompileMocker:
        def __init__(self, enable_mock):
            self.enable_mock = enable_mock
            self.original_compile = torch.compile

        def __enter__(self):
            if self.enable_mock:
                def _mock_compile(model, *args, **kwargs):
                    return model
                torch.compile = _mock_compile
                print("Disabled torch.compile (mocked). Pass --compile to enable.")
            else:
                print("Enabled torch.compile.")

        def __exit__(self, exc_type, exc_val, exc_tb):
            torch.compile = self.original_compile

    # Create MoE Config
    model_config = Config(
        name='MoE-200M',
        block_size=2048,
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_query_groups=4,
        mlp_class_name='LLaMAMoE',
        moe_intermediate_size=2048,
        n_expert=8,
        n_expert_per_token=2,
        padded_vocab_size=50257,
        vocab_size=50257,
        bias=False,
        parallel_residual=False,
        rope_base=10000,
        norm_class_name='RMSNorm',
        norm_eps=1e-5,
    )

    dataset_root = args.dataset_dir
    if args.wandb_dataset:
        artifact_ref = _resolve_wandb_artifact_ref(args.wandb_dataset)
        print(f"Downloading dataset Artifact from W&B: {artifact_ref}")
        dataset_root = _download_dataset_from_wandb(
            artifact_ref=artifact_ref,
            root_dir=args.wandb_artifacts_dir,
        )
        print(f"Using dataset from: {dataset_root}")

    # Setup data module
    data_module = TextFiles(
        train_data_path=Path(dataset_root) / "train",
        val_data_path=Path(dataset_root) / "val",
        num_workers=2,
    )

    # Setup training args
    train = TrainArgs(
        global_batch_size=args.global_batch_size,
        log_interval=1,
        max_tokens=args.max_tokens,
        lr_warmup_steps=5,
        micro_batch_size=args.micro_batch_size,
        max_seq_length=args.max_seq_length,
        save_interval=10,
        max_norm=1.0,
    )

    # Ensure LitGPT's internal hyperparameter capture (jsonargparse) sees a valid CLI.
    # This prevents post-run errors like "Option 'model_name' is required".
    sys.argv = [
        sys.argv[0],
        "MoE-200M",
        "--logger_name",
        "csv",
        "--precision",
        "bf16-mixed",
        "--out_dir",
        str(Path("./checkpoints")),
        "--tokenizer_dir",
        str(Path("./data/tokenizer")),
        "--train.global_batch_size",
        str(args.global_batch_size),
        "--train.micro_batch_size",
        str(args.micro_batch_size),
        "--train.max_tokens",
        str(args.max_tokens),
        "--train.max_seq_length",
        str(args.max_seq_length),
    ]

    if args.resume:
        sys.argv.extend(["--resume", str(args.resume)])

    stop = threading.Event()
    monitor_thread = None
    bar = None
    if args.progress:
        monitor_thread, bar = _start_progress_bar(
            out_dir=Path("./checkpoints"),
            total_tokens=args.max_tokens,
            stop=stop,
        )

    try:
        # Run pretrain
        with TorchCompileMocker(enable_mock=not args.compile):
            setup(
                model_name='MoE-200M',
                model_config=model_config,
                out_dir=Path('./checkpoints'),
                precision='bf16-mixed',
                tokenizer_dir=Path('./data/tokenizer'),
                data=data_module,
                train=train,
                logger_name='csv',
                optimizer={'class_path': 'torch.optim.AdamW', 'init_args': {'lr': 0.0003, 'weight_decay': 0.01}},
                resume=resume,
            )
    finally:
        stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=3.0)
        if bar is not None:
            bar.close()
