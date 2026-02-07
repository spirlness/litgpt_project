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
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Installing dependencies from $PROJECT_ROOT into .venv_tpu (Python 3.11)..."
# We must use --python pointing to the venv's interpreter to avoid uv defaulting to system Python 3.12
# We also allow pre-releases to avoid the 'unsatisfiable' error for specific XLA builds
uv pip install --python "$PROJECT_ROOT/.venv_tpu/bin/python" -e "$PROJECT_ROOT[tpu]" --prerelease=allow

# 4. Set environment variables optimization for TPU
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1

echo "TPU environment setup complete!"
echo "To run training: python run_train.py --train-config configs/train_tpu_v6e.yaml"
