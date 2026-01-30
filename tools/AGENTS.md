# tools/

Generated: 2026-01-30
Commit: 7a1b64c
Branch: master

## OVERVIEW
One-off utilities for inspecting LitGPT internals and validating local environment.

## STRUCTURE
```
tools/
|-- env_sanity_check.py
|-- inspect_litgpt_setup.py
`-- inspect_litgpt_capture_hparams.py
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Verify deps + CUDA | tools/env_sanity_check.py | Imports core libs; does tiny CUDA matmul if available.
| Inspect setup() | tools/inspect_litgpt_setup.py | Prints `litgpt.pretrain.setup` source head.
| Inspect hparam capture | tools/inspect_litgpt_capture_hparams.py | Prints `litgpt.utils.capture_hparams` source head.

## CONVENTIONS
- Treat these as diagnostics, not library code.
- Prefer `uv run python ...` so imports resolve from the project env.

## ANTI-PATTERNS
- Do not treat `tools/env_sanity_check.py` as a full test suite: it is an env/CUDA smoke check; unit/smoke tests live in `tests/`.
