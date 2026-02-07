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
uv venv .venv_tpu --python 3.11 --clear
source .venv_tpu/bin/activate

# 3. Install dependencies using uv
# We find the project root (where pyproject.toml is)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Installing dependencies from $PROJECT_ROOT (including TPU support)..."
cd "$PROJECT_ROOT"
uv pip install -e ".[tpu]"

# 4. Set environment variables optimization for TPU
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1

echo "TPU environment setup complete!"
echo "To run training: python run_train.py --train-config configs/train_tpu_v6e.yaml"
