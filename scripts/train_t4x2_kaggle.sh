#!/bin/bash
# Kaggle T4 x2 optimized launcher with NCCL configuration
# This script sets environment variables to avoid NCCL timeout issues

# NCCL timeout (30 minutes - increased from default 10 minutes)
export NCCL_TIMEOUT=1800000

# NCCL configuration for Kaggle virtualization environment
export NCCL_IB_HCA=0
export NCCL_IB_TIMEOUT=22
export NCCL_NET_GDR_LEVEL=2
export NCCL_BLOCKING_WAIT=0
export NCCL_DYNAMIC_TREE=0
export NCCL_DEBUG=INFO

# PyTorch alloc config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
uv run python run_train.py --train-config configs/train_t4x2.yaml
