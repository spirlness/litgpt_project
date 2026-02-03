# LitGPT MoE Training Project

Mixture of Experts language model training with LitGPT framework on TinyStories dataset.

## Quick Start

```bash
# Install dependencies
uv sync

# Download dataset
uv run python download_tinystories.py

# Prepare tokenized data
uv run python prepare_data.py

# Train model
uv run python run_train.py

# Generate text
uv run python generate.py --prompt "Once upon a time"

# Evaluate model
uv run python evaluate.py
```

## Project Structure

```
litgpt_project/
├── run_train.py          # Training entrypoint
├── generate.py           # Text generation
├── evaluate.py           # Model evaluation
├── prepare_data.py       # Data preprocessing
├── download_tinystories.py # Dataset downloader
├── src/
│   └── utils.py          # Performance utilities
├── tests/
│   ├── test_smoke.py     # CPU smoke tests
│   └── test_gpu_training.py # GPU performance tests
├── configs/
│   ├── moe_30m_debug.yaml
│   ├── moe_200m.yaml
│   └── moe_400m.yaml
├── model_config.yaml     # Active model config
└── train_config.yaml     # Training hyperparameters
```

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | ~549M |
| Layers | 12 |
| Embedding Dim | 768 |
| Attention Heads | 12 |
| Query Groups (GQA) | 4 |
| Experts | 8 |
| Active Experts | 2 |
| Context Length | 2048 |
| Vocab Size | 50257 |

## Training

### Basic Training

```bash
uv run python run_train.py
```

### With Optimizations

```bash
# Enable torch.compile and Flash Attention
uv run python run_train.py --compile --flash-attention

# High-performance mode
uv run python run_train.py --compile --compile-mode reduce-overhead --flash-attention-force
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--compile` | Enable torch.compile |
| `--compile-mode` | `default`, `reduce-overhead`, `max-autotune` |
| `--flash-attention` | Verify Flash Attention 2 |
| `--flash-attention-force` | Fail if Flash Attention unavailable |
| `--gradient-checkpointing` | Enable gradient checkpointing |
| `--resume auto` | Auto-resume from latest checkpoint |
| `--smoke-test` | Minimal CPU sanity check |
| `--progress` | Show training progress bar |

### Resume Training

```bash
# Auto-resume from latest checkpoint
uv run python run_train.py --resume auto

# Resume from specific checkpoint
uv run python run_train.py --resume checkpoints/step-00001000
```

## Generation

```bash
uv run python generate.py --prompt "The little robot" --max_tokens 200
```

## Testing

```bash
# Run all tests
uv run pytest

# CPU smoke tests only
uv run pytest tests/test_smoke.py

# GPU performance tests
uv run pytest tests/test_gpu_training.py

# Skip slow tests
uv run pytest -m "not slow"
```

## Requirements

- Python 3.12+
- CUDA 12.8+ (for GPU training)
- Ampere+ GPU (SM 8.0+) for Flash Attention 2

## Configuration

### train_config.yaml

```yaml
train:
  global_batch_size: 128
  micro_batch_size: 4
  max_tokens: 320000
  max_seq_length: 512

optimization:
  compile: true
  compile_mode: reduce-overhead
  flash_attention: true
```

### Model Configs

| Config | Use Case |
|--------|----------|
| `configs/moe_30m_debug.yaml` | Fast debugging |
| `configs/moe_200m.yaml` | Default training |
| `configs/moe_400m.yaml` | Larger variant |
