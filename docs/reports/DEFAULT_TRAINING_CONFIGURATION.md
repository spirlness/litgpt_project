# 默认训练配置（更新）

## 当前默认入口

`run_train.py` 当前默认参数：
- `--model-config configs/moe_200m.yaml`
- `--train-config configs/kaggle_t4_ddp.yaml`

即直接运行：

```bash
uv run python run_train.py
```

会使用 `configs/moe_200m.yaml + configs/kaggle_t4_ddp.yaml`。

## 主要配置文件

### 模型配置

- `configs/moe_200m.yaml`
- `configs/moe_30m_debug.yaml`
- `configs/moe_400m.yaml`

### 训练配置

- `configs/kaggle_t4_ddp.yaml`（默认）
- `configs/optimized_rtx3060.yaml`（本地单卡）
- `configs/local_rtx3060*.yaml`（本地变体）

## 关键配置段说明

### `train`

常用字段：
- `micro_batch_size`
- `global_batch_size`
- `max_tokens`
- `max_seq_length`
- `save_interval`
- `gradient_checkpointing`

### `optimization`

常用字段：
- `compile`
- `compile_mode`
- `compile_dynamic`
- `compile_fullgraph`
- `flash_attention`
- `flash_attention_force`
- `disable_math_fallback`

### `checkpointing`

```yaml
checkpointing:
  upload_to_hf: false
  # hf_repo_id: your-username/your-model-repo
```

## 命令行可覆盖项（当前代码支持）

`run_train.py` 目前仅支持以下 CLI 覆盖：
- `--compile / --no-compile`
- `--compile-mode`
- `--compile-dynamic / --no-compile-dynamic`
- `--compile-fullgraph / --no-compile-fullgraph`
- `--flash-attention / --no-flash-attention`
- `--flash-attention-force / --no-flash-attention-force`

不支持 `--smoke-test`、`--micro-batch-size`、`--global-batch-size` 这类参数；这些应通过 YAML 配置文件修改。

## 典型命令

```bash
# 默认配置训练（kaggle_t4_ddp + moe_200m）
uv run python run_train.py

# 本地单卡训练
uv run python run_train.py \
  --model-config configs/moe_200m.yaml \
  --train-config configs/optimized_rtx3060.yaml

# 临时关闭 compile
uv run python run_train.py --no-compile

# 强制 Flash Attention 可用性检查
uv run python run_train.py --flash-attention --flash-attention-force
```

## 注意事项

- `run_train.py` 对 MoE 模型默认禁用 `torch.compile`（兼容性保护逻辑）。
- 若需验证 Flash Attention + compile 联合路径，可使用 dense smoke 配置：
  - `configs/dense_compile_flash_smoke_model.yaml`
  - `configs/dense_compile_flash_smoke_train.yaml`
