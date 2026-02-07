#!/bin/bash
# TPU setup script using uv
# Usage: source setup_tpu_env.sh

echo "Setting up TPU environment with uv..."

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 2. Create virtual environment
uv venv .venv_tpu
source .venv_tpu/bin/activate

# 3. Install core dependencies
# Note: TPU requires torch-xla. We install the version compatible with the current TPU VM runtime.
# Typically TPU VMs come with pre-installed pytorch/xla in the system python, 
# but for a venv we need to be careful.
# This assumes a standard Google Cloud TPU VM environment (usually Python 3.10)

echo "Installing generic dependencies..."
uv pip install litgpt lightning jsonargparse bitsandbytes-cuda110- # Remove cuda bitsandbytes if present

# 4. Install PyTorch XLA
# IMPORTANT: TPU VMs usually need specific torch/xla versions matching the runtime.
# This command installs the latest stable release for TPU.
echo "Installing PyTorch XLA for TPU..."
uv pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 \
    -f https://storage.googleapis.com/libtpu-releases/index.html

# 5. Install Hugging Face Hub for upload
uv pip install huggingface_hub[cli]

# 6. Set environment variables optimization for TPU
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1  # Default to BF16

echo "TPU environment setup complete!"
echo "To run training: python run_train.py --train-config configs/train_tpu_v6e.yaml"
