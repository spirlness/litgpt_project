# MoE-200M Training Project with LitGPT

A Mixture of Experts (MoE) language model training project using Lightning LitGPT framework, based on TinyStories dataset.

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
├── run_train.py              # Main training script
├── src/
│   └── custom_moe.py             # Custom MoE implementation (if used)
├── docs/
│   └── AGENTS.md                 # Agent documentation
├── prepare_data.py            # Data preprocessing
├── generate.py               # Text generation script
├── evaluate.py               # Model evaluation script
├── model_config.yaml          # Model architecture configuration
├── train_config.yaml          # Training hyperparameters
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

### Log dataset to W&B (Artifact)

After running data preparation, you can upload the dataset directory (raw text + tokenized chunks) to Weights & Biases as a **dataset Artifact**.

```bash
# Required: set your target W&B project
$env:WANDB_PROJECT="your-project"

# Optional: team/entity
$env:WANDB_ENTITY="your-entity"

# Enable dataset upload
$env:WANDB_LOG_DATASET="1"

# Prepare data and upload as an Artifact
uv run python prepare_data.py --log-to-wandb --wandb-artifact dataset-custom_text
```

If the dataset is already prepared and you only want to upload it (without re-tokenizing), run:

```bash
$env:WANDB_PROJECT="your-project"
uv run python wandb_dataset.py --data-dir data/custom_text --wandb-artifact dataset-custom_text
```

If you want to test without network access, set offline mode:

```bash
$env:WANDB_MODE="offline"
uv run python prepare_data.py --log-to-wandb --wandb-artifact dataset-custom_text
```

## Usage

### Training

#### Basic Training
```bash
uv run python run_train.py
```

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

#### Train using a W&B dataset Artifact

If you uploaded the dataset with [wandb_dataset.py](wandb_dataset.py), you can point training to the Artifact and it will be downloaded automatically.

```bash
$env:WANDB_ENTITY="your-entity"  # required if you use project/name:alias form
uv run python run_train.py --wandb-dataset your-entity/your-project/dataset-custom_text:latest
```

Artifacts will be cached under `./data/wandb_artifacts` by default (override with `--wandb-artifacts-dir`).

#### Training Configuration

Training parameters are defined in `train_config.yaml`:
```yaml
out_dir: ./checkpoints
precision: bf16-mixed
tokenizer_dir: ./data/tokenizer

data:
  class_path: litgpt.data.TextFiles
  init_args:
    train_data_path: ./data/custom_text/train
    val_data_path: ./data/custom_text/val
    num_workers: 2

train:
  global_batch_size: 8
  log_interval: 1
  max_tokens: 320000
  lr_warmup_steps: 5
  micro_batch_size: 1
  save_interval: 10

logger_name: csv

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.0003
    weight_decay: 0.01
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

The training script handles several compatibility fixes for MoE on Windows:

#### 1. Meta Tensor FLOPs Patch
```python
# Patch measure_flops to skip MoE incompatibility
import lightning.fabric.utilities.throughput as throughput_module
_orig_measure_flops = throughput_module.measure_flops

def _measure_flops_patch(model, forward_fn, loss_fn, *args, **kwargs):
    try:
        return _orig_measure_flops(model, forward_fn, loss_fn, *args, **kwargs)
    except (NotImplementedError, AttributeError) as e:
        print(f"FLOPs measurement not supported with MoE: {e}")
        return 0.0  # Return 0 flops for MoE models

throughput_module.measure_flops = _measure_flops_patch
```

**Why needed**: MoE models use `torch.where(mask)` for expert routing, which doesn't support meta tensor FLOPs measurement in PyTorch 2.6. The patch gracefully falls back to 0 FLOPs.

#### 2. Torch Compile Mock
```python
# Mock torch.compile to avoid issues on Windows
_orig_compile = torch.compile

def _mock_compile(model, *args, **kwargs):
    return model  # Return model unmodified

torch.compile = _mock_compile
```

**Why needed**: On Windows, `torch.compile` with MoE can cause issues. The mock returns model as-is.

#### 3. Direct API Usage
```python
from litgpt.config import Config
from litgpt.pretrain import setup
from litgpt.data import TextFiles

# Create Config directly (newer litgpt API)
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

# Setup data module
data_module = TextFiles(
    train_data_path=Path('./data/custom_text/train'),
    val_data_path=Path('./data/custom_text/val'),
    num_workers=2,
)

# Run training
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
)
```

**Why needed**: Newer litgpt versions use direct Python API instead of CLI arguments. This provides better control and easier debugging.

## Data Preparation

### Expected Data Format
Text files in `data/custom_text/train/` and `data/custom_text/val/` directories.

### Automatic Preprocessing
LitGPT automatically preprocesses text data on first run:
- Tokenizes text files
- Creates optimized dataset files
- Caches for subsequent runs

### Clearing Cache
If training data changes, remove cached files:
```bash
rm -rf data/custom_text/train/train
rm -rf data/custom_text/val/val
```

## Troubleshooting

### Common Issues

#### 1. `meta_nonzero_assume_all_nonzero does not exist`
**Error**: Torch 2.6 removed this config option.
**Fix**: The script already handles this with a try-except block (no error will be raised).

#### 2. FLOPs Measurement Error
**Error**: `NotImplementedError: aten::nonzero: attempted to run with Meta tensors`
**Fix**: Applied in `run_train.py` via FLOPs patch (lines 13-24).

#### 3. Tokenizer Not Found
**Error**: `Tokenizer directory not found`
**Fix**: Ensure `data/tokenizer/` contains `tokenizer.json` or `tokenizer.model`:
```bash
# Create tokenizer from Hugging Face
uv run python -c "
from transformers import LlamaTokenizerFast
tokenizer = LlamaTokenizerFast.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
tokenizer.save_pretrained('./data/tokenizer')
"
```

#### 4. CUDA Out of Memory
**Error**: `CUDA out of memory`
**Fixes**:
- Set allocator config to reduce fragmentation (PowerShell):
```bash
$env:PYTORCH_ALLOC_CONF="expandable_segments:True"
```
- Reduce `micro_batch_size` / `global_batch_size` (6GB GPU suggested):
```bash
uv run python run_train.py --micro-batch-size 1 --global-batch-size 8
```
- Reduce sequence length (often the biggest VRAM lever):
```bash
uv run python run_train.py --max-seq-length 1024
```
- Reduce `n_layer` or `n_embd` in `model_config.yaml`

#### 5. Build System Errors
**Error**: `uv sync` fails with build errors
**Fix**: The project uses virtual environment mode (no `[build-system]` in `pyproject.toml`).

## Training Hyperparameters

### Current Settings
| Parameter | Value | Rationale |
|-----------|---------|------------|
| `micro_batch_size` | 1 | Fits in ~6GB VRAM |
| `global_batch_size` | 8 | Lower effective batch for 6GB GPUs |
| `learning_rate` | 0.0003 | Standard for MoE |
| `weight_decay` | 0.01 | Regularization |
| `lr_warmup_steps` | 5 | Minimal warmup |
| `max_tokens` | 320,000 | Quick test run (~20k tokens/batch) |
| `save_interval` | 10 | Save every 10 steps |

### Adjusting for Full Training
For longer training, modify `train_config.yaml`:
```yaml
train:
  max_tokens: 1000000000  # 1B tokens
  lr_warmup_steps: 2000
  save_interval: 1000
  log_interval: 100
```

## Monitoring Training

### CSV Logger
Training metrics are saved to `checkpoints/metrics.csv`:
```
step,train_loss,val_loss,learning_rate,epoch,time,iter_tokens
1,2.345,2.456,0.000300,0,1.23,2048
...
```

### Checking Progress
```bash
# View latest metrics
tail checkpoints/metrics.csv

# Check checkpoints
ls checkpoints/
```

## Dependencies

### Core Dependencies (from `pyproject.toml`)
- `torch >= 2.10.0` (CUDA 12.8 / cu128)
- `litgpt >= 0.5.11`
- `lightning >= 2.6.0`
- `transformers >= 4.57.6`
- `datasets >= 2.19.1`
- `accelerate >= 1.12.0`
- `bitsandbytes >= 0.49.1`

### Additional Dependencies
- `litdata >= 0.2.59` - Data preprocessing and optimization

## Version History

### Recent Changes
- **AF88420** - Update run_train.py: fixed for new litgpt API and added FLOPs patch for MoE
- **B0346BD** - Fix uv sync: remove build-system for virtual environment project
- Initial setup - Created MoE-200M training project

## License

This project uses LitGPT (Apache 2.0 license) and PyTorch (BSD-style license).

## References

- [LitGPT Documentation](https://github.com/Lightning-AI/litgpt)
- [MoE Papers](https://arxiv.org/abs/2101.03961)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
