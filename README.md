# MoE TinyStories with LitGPT

This project implements a Mixture of Experts (MoE) Language Model trained on the TinyStories dataset using [LitGPT](https://github.com/Lightning-AI/litgpt).

## Features

- **Mixture of Experts**: Uses `LLaMAMoE` architecture with configurable experts.
- **TinyStories Dataset**: Trains on a small, synthetic dataset for fast experimentation.
- **Full Pipeline**: Includes data prep, training, evaluation, and inference scripts.
- **W&B Logging**: Integrated Weights & Biases logging.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Pipeline

### 1. Prepare Data
Download the GPT-2 tokenizer and TinyStories dataset:
```bash
python prepare_data.py
```
This will create `data/tokenizer` and `data/tinystories`.

### 2. Generate Configuration
Create the training configuration file (`litgpt_config.yaml`). This script loads the base model config from `configs/moe_200m.yaml` and adds training parameters.
```bash
python create_litgpt_config.py
```

### 3. Train
Run the training using LitGPT's `pretrain` command:
```bash
litgpt pretrain --config litgpt_config.yaml
```
Checkpoints will be saved to `checkpoints/`.

### 4. Evaluate
Calculate the perplexity on the validation set to measure model performance:
```bash
python evaluate.py
```

### 5. Inference
Generate text using your trained model:
```bash
python generate.py --prompt "Once upon a time there was a little" --max_tokens 100
```

## Model Configurations

The project comes with a default 200M MoE configuration (`configs/moe_200m.yaml`).

### Available Configs (Planned)
- `configs/moe_30m_debug.yaml`: Small model for debugging on consumer GPUs/CPU.
- `configs/moe_200m.yaml`: Default balanced configuration.
- `configs/moe_400m.yaml`: Larger model for scaling experiments.

## Directory Structure

- `configs/`: Model architecture YAMLs.
- `data/`: Tokenizers and datasets.
- `checkpoints/`: Model checkpoints.
- `create_litgpt_config.py`: Training config generator.
- `prepare_data.py`: Data download script.
- `generate.py`: Inference script.
- `evaluate.py`: Perplexity evaluation script.
