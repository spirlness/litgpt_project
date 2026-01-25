import sys
import torch
from pathlib import Path

# Fix for MoE meta-device FLOP counting
try:
    import torch.fx.experimental._config as fx_config
    fx_config.meta_nonzero_assume_all_nonzero = True
    print("Enabled meta_nonzero_assume_all_nonzero for MoE FLOP counting.")
except (ImportError, AttributeError) as e:
    print(f"Could not set meta_nonzero_assume_all_nonzero: {e}")

# Patch measure_flops to skip for MoE models
import lightning.fabric.utilities.throughput as throughput_module
_orig_measure_flops = throughput_module.measure_flops

def _measure_flops_patch(model, forward_fn, loss_fn, *args, **kwargs):
    try:
        return _orig_measure_flops(model, forward_fn, loss_fn, *args, **kwargs)
    except (NotImplementedError, AttributeError) as e:
        print(f"FLOPs measurement not supported with MoE: {e}")
        return 0.0  # Return 0 flops

throughput_module.measure_flops = _measure_flops_patch

from litgpt.config import Config
from litgpt.pretrain import setup
from litgpt.args import TrainArgs, EvalArgs
from litgpt.data import TextFiles

if __name__ == "__main__":
    # Mock torch.compile to avoid issues on Windows
    _orig_compile = torch.compile

    def _mock_compile(model, *args, **kwargs):
        return model

    torch.compile = _mock_compile

    # Create MoE Config
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

    # Setup training args
    train = TrainArgs(
        global_batch_size=16,
        log_interval=1,
        max_tokens=320000,
        lr_warmup_steps=5,
        micro_batch_size=2,
        save_interval=10,
        max_norm=1.0,
    )

    # Run pretrain
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
