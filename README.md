# MoE-200M Training Project with LitGPT

A Mixture of Experts (MoE) language model training project using Lightning LitGPT framework, based on TinyStories dataset.

## Performance Optimizations (New)

This project has been optimized for high-performance training with the following features:

### 1. Flash Attention 2 Support
- **Automatic Verification**: The system checks for GPU compute capability (Ampere 8.0+ required) and ensures the `FLASH_ATTENTION` backend is available in PyTorch.
- **Enforcement**: Added `--flash-attention-force` to ensure training only runs with optimized kernels, preventing silent fallbacks to slower implementations.
- **Hardware Optimized**: Specifically tuned for NVIDIA RTX 30/40 series GPUs.

### 2. Configurable `torch.compile`
- **Injected Patching**: Intercepts internal LitGPT calls to inject optimization parameters.
- **Modes**: Support for `default`, `reduce-overhead`, and `max-autotune`.
- **Dynamic Shapes**: Optimized for variable sequence lengths using `--compile-dynamic`.
- **Proven Speedup**: Benchmark shows a throughput increase to ~16.25 TFLOPs on mobile hardware (RTX 3060).

### 3. MoE Specific Fixes
- **FLOPs Measurement**: Custom patch to handle MoE routing incompatibility with meta tensors.
- **Gradient Checkpointing**: Non-reentrant checkpointing enabled by default to save VRAM while maintaining performance.

## Project Overview

This project implements a custom 200M parameter Mixture of Experts (MoE) language model using:
- **LitGPT 0.5.11** - Lightning AI's GPT training framework
- **PyTorch 2.10.0+cu128** - Deep learning framework
- **Lightning 2.6.0** - Training orchestration
- **MoE Architecture** - 8 experts with 2 active experts per token

## Model Architecture

### Model Configuration
- **Name**: MoE-200M
- **Total Parameters**: ~549 million
- **Block Size (Context Length)**: 2048 tokens
- **Vocabulary Size**: 50257 tokens

### Architecture Details
| Parameter | Value | Description |
|-----------|---------|-------------|
| `n_layer` | 12 | Number of transformer blocks |
| `n_embd` | 768 | Embedding dimension |
| `n_head` | 12 | Number of attention heads |
| `n_query_groups` | 4 | Number of query groups (GQA) |
| `mlp_class_name` | `LLaMAMoE` | Mixture of Experts MLP |
| `moe_intermediate_size` | 2048 | MoE intermediate dimension |
| `n_expert` | 8 | Total number of experts |
| `n_expert_per_token` | 2 | Active experts per token (Top-2 routing) |
| `norm_class_name` | `RMSNorm` | Normalization type |
| `parallel_residual` | False | Residual connection style |
| `rope_base` | 10000 | Rotary position embedding base |

### MoE Design Decisions
- **Top-2 Routing**: Each token is routed to top 2 experts based on gate scores
- **Expert Capacity**: No explicit capacity limit (uses standard routing)
- **Load Balancing**: Implicit through gradient-based routing
- **Grouped Query Attention (GQA)**: 4 query groups for efficiency

## File Structure

```
litgpt_project/
├── run_train.py              # Main training script (with optimization patches)
├── src/
│   ├── utils.py                  # Performance & verification utilities
│   └── custom_moe.py             # Custom MoE implementation
├── docs/
│   └── AGENTS.md                 # Agent documentation
├── prepare_data.py            # Data preprocessing
├── generate.py               # Text generation script
├── evaluate.py               # Model evaluation script
├── model_config.yaml          # Model architecture configuration
├── train_config.yaml          # Training hyperparameters (with optimization section)
├── pyproject.toml             # Project dependencies (uv)
├── uv.lock                   # Locked dependencies
├── data/
│   ├── tokenizer/             # Tokenizer files
│   └── custom_text/
│       ├── train/             # Training text data
│       └── val/              # Validation text data
└── checkpoints/              # Model checkpoints
```

## Installation

### Prerequisites
- Python 3.12–3.13
- CUDA 12.8 compatible GPU
- NVIDIA RTX 3060 Laptop GPU (or similar)

### Setup with uv
```bash
# Install dependencies
uv sync

# Verify installation
uv run python -c "import litgpt; print('litgpt installed')"
uv run python -c "import torch; print(f'torch: {torch.__version__}')"
```

## Usage

### Training

#### Optimized Training (Recommended)
```bash
# Basic training with default optimizations
uv run python run_train.py --compile --flash-attention

# High-performance training (Longer warmup, faster execution)
uv run python run_train.py --compile --compile-mode default --flash-attention-force
```

*Note: `max-autotune` mode is available but may cause CUDAGraphs memory conflicts on some hardware. `default` mode is recommended for stability.*

#### Optimization Flags
| Flag | Description |
|------|-------------|
| `--compile` | Enable `torch.compile` optimization |
| `--compile-mode` | Mode: `default`, `reduce-overhead`, `max-autotune` |
| `--compile-dynamic` | Enable dynamic shape support |
| `--flash-attention` | Verify Flash Attention 2 activation |
| `--flash-attention-force` | Fail if Flash Attention 2 is unavailable |

#### Resume training (checkpoint)

```bash
# Auto-resume from latest checkpoint in ./checkpoints
uv run python run_train.py --resume auto

# Resume from a specific checkpoint directory
uv run python run_train.py --resume checkpoints/step-00000010
```

#### Progress bar

The training script can show a progress bar based on `total_tokens` written by the CSV logger.

```bash
uv run python run_train.py --progress
```

#### Training Configuration

Training and optimization parameters are defined in `train_config.yaml`:
```yaml
# Performance optimization options
optimization:
  compile: true
  compile_mode: default
  compile_dynamic: false
  flash_attention: true
  flash_attention_force: false
```

### Generating Text

```bash
uv run python generate.py
```

### Evaluating Model

```bash
uv run python evaluate.py
```

## Training Script Details

### `run_train.py` Implementation

The training script handles several compatibility and performance fixes:

#### 1. torch.compile Injection
We use a custom context manager to inject optimization settings into LitGPT's internal calls:
```python
def create_compile_context(use_compile, mode, dynamic, fullgraph):
    if not use_compile:
        return patch("torch.compile", side_effect=lambda m, *a, **kw: m)
    # ... wraps torch.compile with user-provided args
```

#### 2. Meta Tensor FLOPs Patch
Handled via `src/utils.py` to prevent crashes when measuring MoE throughput.

## Troubleshooting

### Common Issues

#### 1. CUDAGraphs Error with `max-autotune`
**Error**: `accessing tensor output of CUDAGraphs that has been overwritten`
**Fix**: Switch to `--compile-mode default` or `reduce-overhead`.

#### 2. Flash Attention 2 Unavailable
**Error**: `RuntimeError: GPU compute capability < 8.0`
**Fix**: Flash Attention 2 requires Ampere GPUs. Disable with `--no-flash-attention`.

#### 3. CUDA Out of Memory
**Fixes**:
- Reduce `micro_batch_size` to 1.
- Reduce `max_seq_length` in `train_config.yaml`.
- Ensure `expandable_segments:True` is set in `PYTORCH_ALLOC_CONF`.

## Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|---------|------------|
| `micro_batch_size` | 1 | Fits in ~6GB VRAM |
| `global_batch_size` | 4-8 | Balanced throughput |
| `learning_rate` | 0.0003 | Standard for MoE |
| `precision` | `bf16-mixed` | Required for Flash Attention |

## Monitoring Training

### CSV Logger
Training metrics are saved to `checkpoints/metrics.csv`.

### Checking Progress
```bash
tail checkpoints/metrics.csv
```

## Version History

- **Current Version**: Performance Optimization Update
  - Added Flash Attention 2 verification
  - Added configurable `torch.compile`
  - Integrated performance flags into CLI and YAML config

## License

This project uses LitGPT (Apache 2.0 license) and PyTorch (BSD-style license).
