# PROJECT KNOWLEDGE BASE

Generated: 2026-01-30
Commit: 7a1b64c
Branch: master

## OVERVIEW
MoE language model training on TinyStories-style text using LitGPT + Lightning + PyTorch (Windows/CUDA-oriented).

## STRUCTURE
```
./
|-- run_train.py              # main training entry point
|-- prepare_data.py           # tokenizer download + TextFiles prepare_data()
|-- generate.py               # sampling from checkpoints
|-- evaluate.py               # simple loss/perplexity eval on .bin shards
|-- wandb_dataset.py          # upload prepared dataset dir as W&B Artifact
|-- custom_moe.py             # alternative MoE layers; contains known bug
|-- configs/                  # model-size variants (YAML)
|-- tools/                    # inspection + env sanity checks
|-- model_config.yaml         # canonical MoE-200M model params (YAML)
|-- train_config.yaml         # canonical training params (YAML)
|-- litgpt_config_200m.yaml   # legacy combined config (deprecated)
`-- .github/workflows/        # CI workflow(s)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Train | run_train.py | Patches FLOPs counting; mocks torch.compile on Windows.
| Prepare dataset | prepare_data.py | Downloads TinyLlama tokenizer; writes tokenized data.
| Generate text | generate.py | Reads config + checkpoint; streams tokens.
| Evaluate | evaluate.py | Requires `*.bin` token data; uses random weights if no checkpoint.
| Upload dataset to W&B | wandb_dataset.py | Uploads local `data/custom_text` as Artifact.
| Model variants | configs/moe_*.yaml | 30M debug / 200M / 400M.
| Canonical configs | model_config.yaml, train_config.yaml | Prefer these over legacy.
| Env validation | tools/env_sanity_check.py | Import + CUDA smoke test.

## CODE MAP
| File | Role | Key details |
|------|------|------------|
| run_train.py | training | Uses `litgpt.pretrain.setup()`; forces `sys.argv` for LitGPT hparam capture.
| prepare_data.py | data | Uses `litgpt.data.TextFiles` + `litgpt.tokenizer.Tokenizer`.
| generate.py | inference | Loads `litgpt_config.yaml` from checkpoints or root.
| evaluate.py | eval | Memmaps token `.bin` and computes cross-entropy.
| wandb_dataset.py | infra | Computes dir fingerprint; uploads Artifact; writes manifest.json.
| custom_moe.py | experiments | Multiple MoE implementations; some are not production-ready.

## CONVENTIONS
- Python: `>=3.12,<3.14` in `pyproject.toml`.
- Dependency manager: `uv` (lockfile `uv.lock`); `requirements.txt` is legacy.
- Package indexes: Aliyun mirror default; PyTorch wheels from cu128 index via `tool.uv`.
- Data layout:
  - tokenizer: `data/tokenizer/`
  - raw text: `data/custom_text/{train,val}/`
  - tokenized shards: `data/custom_text/**.bin` (generated)
- Outputs:
  - checkpoints: `./checkpoints/`
  - CSV logs: `./checkpoints/logs/**/metrics.csv`

## ANTI-PATTERNS (THIS PROJECT)
- Do not rely on `litgpt_config_200m.yaml`; it is marked legacy/deprecated.
- CI uses `.github/workflows/ci.yml` (uv-based). GitHub runners are GPU-free, so CI sets `SKIP_BNB_RUNTIME=1` for `tools/env_sanity_check.py`.
- Do not move `PYTORCH_ALLOC_CONF` setup below the first `import torch` in `run_train.py` (allocator config must be set before torch import).
- Do not use `custom_moe.SimplifiedLLaMAMoE` without fixing it (ruff reports an undefined name at `custom_moe.py:179`).
- Do not assume `generate.py` is clean: it defines `AsyncTokenStreamer` twice (second definition shadows the first).
- Do not treat `evaluate.py` results as meaningful if it prints "Warning: Evaluating random weights!".

## COMMANDS
```bash
# Install deps
uv sync

# Environment sanity check (imports + CUDA smoke test)
uv run python tools/env_sanity_check.py

# Prepare dataset (downloads tokenizer)
uv run python prepare_data.py --data-dir data/custom_text

# Upload prepared dataset as W&B Artifact
uv run python wandb_dataset.py --data-dir data/custom_text --wandb-project <project> --wandb-artifact dataset-custom_text

# Train
uv run python run_train.py
uv run python run_train.py --resume auto

# Generate
uv run python generate.py --ckpt_dir checkpoints --prompt "Once upon a time"

# Evaluate
uv run python evaluate.py
```

## NOTES
- `WANDB_ENTITY` is required if you pass `--wandb-dataset project/name:alias` (otherwise use full `entity/project/name:alias`).
- LSP: document symbols are not available in this environment; rely on grep/AST-grep for navigation.
