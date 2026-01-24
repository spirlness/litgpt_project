#!/usr/bin/env python
"""
Update LitGPT config to support MoE
"""

import yaml
from pathlib import Path
import sys
import os

# Set environment for data processing
os.environ["DATA_OPTIMIZER_GLOBAL_RANK"] = "0"
os.environ["DATA_OPTIMIZER_NUM_WORKERS"] = "1"

# Load model config
with open("configs/moe_200m.yaml", "r", encoding="utf-8") as f:
    moe_config = yaml.safe_load(f)

# LitGPT pretrain config for MoE
pretrain_config = {
    "model_config": moe_config,
    "tokenizer_dir": "./data/tokenizer",
    "num_workers": 2,
    "precision": "bf16-mixed",
    "out_dir": "./checkpoints",
    "train": {
        "max_steps": 10000,
        "global_batch_size": 16,
        "micro_batch_size": 4,
        "learning_rate": 3e-4,
        "save_interval": 500,
        "log_interval": 10,
    },
    "logger_name": "wandb",
    "wandb": {"project": "moe-tinystories", "log_model": False},
}

# Save LitGPT config
config_path = Path("litgpt_config.yaml")
with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(pretrain_config, f)

print(f"Created LitGPT config at: {config_path}")
print("You can now run training with:")
print("  litgpt pretrain --config litgpt_config.yaml")
