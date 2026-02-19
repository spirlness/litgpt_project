# LitGPT Flash Attention 2 使用说明（更新）

## 适用范围

本项目通过 PyTorch SDPA backend 使用 Flash Attention 2 能力。是否可用取决于：
- GPU 计算能力（通常要求 SM >= 8.0）
- PyTorch CUDA 构建与驱动环境

`flash_attn` 第三方包不是必需项；未安装时可使用 PyTorch 内置 backend。

## 训练配置

在训练配置文件（如 `configs/optimized_rtx3060.yaml`）中设置：

```yaml
optimization:
  flash_attention: true
  flash_attention_force: false
  disable_math_fallback: false
```

参数说明：
- `flash_attention`: 启用 Flash/Efficient SDPA backend（禁用时回到 math）。
- `flash_attention_force`: 若不可用则直接报错。
- `disable_math_fallback`: 禁用 math backend 回退。

## CLI 覆盖

`run_train.py` 支持以下命令行覆盖：

```bash
uv run python run_train.py \
  --flash-attention \
  --flash-attention-force
```

注意：不支持 `--optimization.xxx` 这种嵌套参数写法。

## 验证方式

### 1. 脚本验证

```bash
uv run python scripts/verify_flash.py
```

### 2. 训练日志验证

当训练配置启用并检查成功时，日志会出现：
- `Status: Flash Attention 2 ENABLED`

## 已知限制

- 本项目的 `run_train.py` 对 MoE 模型默认禁用 `torch.compile`（兼容性保护）。
- Flash Attention 与 `torch.compile` 联合验证可使用 dense smoke 配置：
  - `configs/dense_compile_flash_smoke_model.yaml`
  - `configs/dense_compile_flash_smoke_train.yaml`

## 常见问题

1. 报错 `Flash Attention 2 is required but not available`
- 检查 GPU 计算能力是否 >= 8.0
- 检查 CUDA 驱动和 PyTorch CUDA 版本

2. 已启用但速度提升不明显
- 短序列/小 batch 下收益有限
- 先确认日志确实显示 `Flash Attention 2 ENABLED`
