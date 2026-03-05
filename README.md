# LitGPT MoE 训练项目

基于 [LitGPT](https://github.com/Lightning-AI/litgpt) 的 MoE 训练工程，包含数据准备、训练、生成、评估和测试脚本。

## 快速开始

### 1. 安装依赖

```bash
pip install uv
uv sync
```

建议所有命令都通过 `uv run` 执行，确保使用项目虚拟环境。

### 2. 准备数据

```bash
uv run python scripts/download_tinystories.py
uv run python prepare_data.py --data-dir data/custom_text
```

### 3. 训练

默认参数（当前代码）：
- `run_train.py --model-config configs/moe_200m.yaml`
- `run_train.py --train-config configs/kaggle_t4_ddp.yaml`

默认入口运行：

```bash
uv run python run_train.py
```

0.3B MoE（RTX 3060 6GB，推荐 fit6gb）：

```bash
# 0) 先做环境检查（必须在项目目录运行）
uv run python scripts/env_sanity_check.py

# 1) 下载 TinyStories（默认 10k train / 2k val）
uv run python scripts/download_tinystories.py

# 2) 预处理到 LitGPT TextFiles 目录
uv run python prepare_data.py --data-dir data/custom_text

# 3) 先跑 fit6gb smoke（20k tokens, seq=256）
HF_TOKEN=your_hf_token \
HF_REPO_ID=your-username/your-model-repo \
uv run python run_train.py \
  --model-config configs/moe_300m_fit6gb.yaml \
  --train-config configs/local_rtx3060_moe300_fit6gb_smoke.yaml

# 4) smoke 稳定后跑 fit6gb baseline（10M tokens, seq=256）
HF_TOKEN=your_hf_token \
HF_REPO_ID=your-username/your-model-repo \
uv run python run_train.py \
  --model-config configs/moe_300m_fit6gb.yaml \
  --train-config configs/local_rtx3060_moe300_fit6gb_baseline.yaml
```

说明：
- `HF_REPO_ID` 通过环境变量传入，训练配置中不会写死仓库名。
- 建议始终用 `uv run`（或 `.venv/bin/python`）执行，避免系统 Python 导入路径污染。
- 如果 baseline 仍遇到 OOM，先把 `global_batch_size` 从 `8` 降到 `4`。

### 4. 生成与评估

```bash
# generate.py 参数名是 --ckpt_dir
uv run python generate.py --prompt "Once upon a time" --ckpt_dir checkpoints

# evaluate.py 参数名是 --checkpoint-dir
uv run python evaluate.py --checkpoint-dir checkpoints
```

## Flash Attention 与 torch.compile

- Flash Attention 开关在训练配置的 `optimization` 段：
  - `flash_attention`
  - `flash_attention_force`
  - `disable_math_fallback`
- `torch.compile` 开关：
  - `compile`
  - `compile_mode`
  - `compile_dynamic`
  - `compile_fullgraph`

注意：
- 在当前 `run_train.py` 中，**MoE 模型默认禁用 `torch.compile`**（已知兼容性保护逻辑）。

## Hugging Face 自动上传检查点

默认关闭。配置位于训练配置的 `checkpointing` 段：

```yaml
checkpointing:
  upload_to_hf: false
  # hf_repo_id: your-username/your-model-repo
```

也支持环境变量 `HF_REPO_ID`。

## 常用命令

```bash
# 运行全量测试
uv run pytest -q

# 仅运行 smoke 测试
uv run pytest -q tests/test_smoke.py

# 验证 Flash Attention 支持
uv run python scripts/verify_flash.py
```

## 项目结构

```text
litgpt_project/
├── configs/                  # 训练/模型配置
├── data/                     # 数据与 tokenizer
├── docs/                     # 文档（含历史报告）
├── scripts/                  # 辅助脚本
├── src/litgpt_moe/           # 核心代码
├── run_train.py              # 训练入口
├── prepare_data.py           # 数据准备入口
├── generate.py               # 生成入口
├── evaluate.py               # 评估入口
└── tests/                    # 测试
```
