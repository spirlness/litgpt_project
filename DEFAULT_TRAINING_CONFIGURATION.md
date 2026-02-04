# LitGPT默认训练配置

## 概述

LitGPT使用YAML配置文件来定义训练参数。默认配置针对MoE-200M模型优化，平衡了性能和资源使用。

## 主要配置文件

### 1. `train_config.yaml` - 训练配置
```yaml
out_dir: ./checkpoints
precision: bf16-mixed
tokenizer_dir: ./data/tokenizer

data:
  class_path: litgpt.data.TextFiles
  init_args:
    train_data_path: ./data/custom_text/train
    val_data_path: ./data/custom_text/val
    num_workers: 2

train:
  global_batch_size: 128
  log_interval: 1
  max_tokens: 320000
  lr_warmup_steps: 5
  micro_batch_size: 4
  save_interval: 500
  gradient_checkpointing: false
  max_norm: 1.0
  max_seq_length: 512

eval:
  interval: 0
  final_validation: false

resume: auto

optimization:
  compile: true
  compile_mode: reduce-overhead
  compile_dynamic: true
  compile_fullgraph: false
  flash_attention: true
  flash_attention_force: false
  disable_math_fallback: false

logger_name: csv

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 0.0003
    weight_decay: 0.01
```

### 2. `model_config.yaml` - 模型配置
```yaml
name: MoE-200M
bias: false
block_size: 2048
mlp_class_name: LLaMAMoE
moe_intermediate_size: 2048
n_embd: 768
n_expert: 8
n_expert_per_token: 2
n_head: 12
n_layer: 12
n_query_groups: 4
norm_class_name: RMSNorm
norm_eps: 0.00001
padded_vocab_size: 50257
parallel_residual: false
rope_base: 10000
vocab_size: 50257
```

## 详细配置说明

### 训练参数 (train)
- **global_batch_size**: 128 - 全局批量大小
- **micro_batch_size**: 4 - 微批量大小
- **max_tokens**: 320000 - 训练的最大token数
- **max_seq_length**: 512 - 最大序列长度
- **lr_warmup_steps**: 5 - 学习率预热步数
- **save_interval**: 500 - 检查点保存间隔(步数)
- **log_interval**: 1 - 日志记录间隔
- **max_norm**: 1.0 - 梯度裁剪范数
- **gradient_checkpointing**: false - 梯度检查点(节省内存)

### 优化器参数 (optimizer)
- **类型**: AdamW
- **学习率**: 0.0003
- **权重衰减**: 0.01

### 性能优化 (optimization)
- **torch.compile**: 启用，模式为reduce-overhead
- **Flash Attention**: 启用
- **动态形状编译**: 启用

### 数据配置 (data)
- **数据类**: TextFiles
- **训练数据路径**: ./data/custom_text/train
- **验证数据路径**: ./data/custom_text/val
- **工作进程数**: 2

### 精度设置
- **计算精度**: bf16-mixed (BF16混合精度)
- **优化器精度**: 32位

## 烟雾测试配置

当使用`--smoke-test`参数时，配置会自动调整为：

```yaml
train:
  micro_batch_size: 1
  global_batch_size: 1
  max_tokens: 64
  max_seq_length: 64
precision: 32-true  # 全精度
logger_name: csv
```

## 命令行覆盖

可以通过命令行参数覆盖配置文件中的设置：

```bash
# 基本训练
uv run python run_train.py

# 烟雾测试
uv run python run_train.py --smoke-test

# 自定义参数
uv run python run_train.py --micro-batch-size 2 --global-batch-size 64

# 禁用编译
uv run python run_train.py --compile false

# 启用梯度检查点
uv run python run_train.py --gradient-checkpointing
```

## 推荐的生产配置

对于生产环境训练，建议以下配置：

```yaml
train:
  global_batch_size: 256  # 增大批量大小
  gradient_checkpointing: true  # 节省内存
  max_tokens: 300000000  # 更多训练数据

optimization:
  flash_attention: true
  flash_attention_force: true
  disable_math_fallback: true
  compile: true
  compile_mode: max-autotune  # 最大优化
```

## 资源需求估算

### 默认配置资源使用
- **GPU内存**: ~8-10 GB
- **训练时间**: 取决于max_tokens设置
- **存储空间**: 取决于检查点保存频率

### 烟雾测试资源使用
- **GPU内存**: ~4-6 GB
- **训练时间**: < 2分钟
- **存储空间**: < 100 MB

## 配置最佳实践

1. **批量大小调整**: 根据GPU内存调整global_batch_size
2. **精度选择**: BF16混合精度提供良好性能和数值稳定性
3. **编译优化**: 使用reduce-overhead模式平衡编译时间和运行时性能
4. **Flash Attention**: 在支持的硬件上始终启用
5. **梯度检查点**: 内存受限时启用以支持更大模型