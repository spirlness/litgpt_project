# configs/

Generated: 2026-01-30
Commit: 7a1b64c
Branch: master

## OVERVIEW
Model-size variants for MoE configs used by scripts in repo.

## STRUCTURE
```
configs/
|-- moe_30m_debug.yaml
|-- moe_200m.yaml
`-- moe_400m.yaml
```

## WHERE TO LOOK
| Need | File | Notes |
|------|------|-------|
| Debug-scale run | configs/moe_30m_debug.yaml | Smaller block_size=1024; fewer layers/experts.
| Default config | configs/moe_200m.yaml | Matches `MoE-200M` settings.
| Larger variant | configs/moe_400m.yaml | More layers/experts; higher memory requirements.

## CONVENTIONS
- These files only cover model-side fields (name/size/MoE params).
- Canonical "current" config for the project is `model_config.yaml` (root); `litgpt_config_200m.yaml` is legacy.
- Keep `norm_eps` numeric (some YAML loaders can coerce `1e-5` to string; `generate.py` explicitly casts).

## ANTI-PATTERNS
- Do not add training/runtime paths here (they belong in `train_config.yaml` or script args).
- Do not diverge field names from LitGPT `Config` (scripts assume LitGPT naming).
