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

单卡本地训练示例：

```bash
uv run python run_train.py \
  --model-config configs/moe_200m.yaml \
  --train-config configs/optimized_rtx3060.yaml
```

快速小模型调试：

```bash
uv run python run_train.py \
  --model-config configs/moe_30m_debug.yaml \
  --train-config configs/optimized_rtx3060.yaml
```

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
- 如需验证 Flash Attention + compile 同时生效，可使用 dense smoke 配置：
  - `configs/dense_compile_flash_smoke_model.yaml`
  - `configs/dense_compile_flash_smoke_train.yaml`

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
