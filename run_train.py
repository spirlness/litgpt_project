import sys
import torch
import argparse
from litgpt.__main__ import main
import os

# Fix for MoE meta-device FLOP counting
try:
    import torch.fx.experimental._config as fx_config

    fx_config.meta_nonzero_assume_all_nonzero = True
    print("Enabled meta_nonzero_assume_all_nonzero for MoE FLOP counting.")
except ImportError:
    print("Could not import torch.fx.experimental._config")

def parse_args():
    parser = argparse.ArgumentParser(description="Run LitGPT training with MoE support")
    parser.add_argument("--config", type=str, default="litgpt_config_200m.yaml", help="Path to config file")
    parser.add_argument("--compile", type=str, default="false", choices=["true", "false", "True", "False"], help="Enable or disable torch.compile (default: false)")

    args, unknown = parser.parse_known_args()
    return args, unknown

if __name__ == "__main__":
    args, rest = parse_args()

    use_compile = args.compile.lower() == "true"

    if not use_compile:
        print("Disabling torch.compile...")
        _orig_compile = torch.compile

        def _mock_compile(model, *args, **kwargs):
            # Return the original model unmodified
            return model

        torch.compile = _mock_compile
    else:
        print("Enabling torch.compile...")

    # Construct sys.argv for litgpt
    # litgpt pretrain --config <config> [rest]
    sys.argv = ["litgpt", "pretrain", "--config", args.config] + rest

    main()
