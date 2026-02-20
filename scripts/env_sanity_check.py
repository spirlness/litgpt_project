"""
Simple environment sanity check script for CI.
Verifies that key dependencies can be imported successfully.
"""

import sys

try:
    import datasets
    import lightning
    import litgpt
    import torch
    import transformers

    print(f"torch: {torch.__version__}")
    print(f"litgpt: imported from {litgpt.__file__}")
    print(f"lightning: {lightning.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"datasets: {datasets.__version__}")
    print("Environment sanity check passed!")
except ImportError as e:
    print(f"Environment sanity check failed: {e}")
    sys.exit(1)
