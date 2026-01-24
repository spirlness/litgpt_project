import sys
import torch

# Fix for MoE meta-device FLOP counting
try:
    import torch.fx.experimental._config as fx_config

    fx_config.meta_nonzero_assume_all_nonzero = True
    print("Enabled meta_nonzero_assume_all_nonzero for MoE FLOP counting.")
except ImportError:
    print("Could not import torch.fx.experimental._config")

from litgpt.__main__ import main

if __name__ == "__main__":
    # Fake argv to call pretrain
    # The main() in litgpt uses sys.argv logic or CLI.
    # litgpt main entry point expects "litgpt" as prog name maybe?
    # We will simulate the command line arguments
    # Disable torch.compile which seems to cause issues on this env with MoE
    # by using --compile=False if supported or env var?
    # LitGPT uses lightning fabric.
    import os
    # Disable Triton cache to avoid weird file path issues on Windows?
    # Or just completely disable compile.
    # LitGPT might be forcing compile=True internally for some models?
    # Actually, try setting arguments directly to disable compile if possible.

    # Force disable compile at a lower level
    import torch
    # torch.compile = lambda *args, **kwargs: (lambda x: x) # This replaces torch.compile with a dummy function.
    # But wait, LitGPT might expect an object that behaves like a model if it calls compile(model).
    # The error "AttributeError: 'function' object has no attribute 'named_parameters'"
    # suggests LitGPT called torch.compile(model), and got back our dummy lambda x:x.
    # Then it tried to use that as a model.
    # The correct mock should be:

    sys.argv = ["litgpt", "pretrain", "--config", "litgpt_config_200m.yaml"]

    main()
