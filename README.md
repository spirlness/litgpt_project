# LitGPT MoE 训练项目

本项目基于 [LitGPT](https://github.com/Lightning-AI/litgpt) 框架，专注于在 TinyStories 数据集上训练混合专家 (MoE) 语言模型。项目经过优化，支持从本地消费级显卡 (如 RTX 3060) 到云端多卡环境的训练。

## ✨ 特性

- **混合专家 (MoE)**: 支持配置专家数量、激活专家数等 MoE 关键参数。
- **环境适配**: 提供针对单卡 (RTX 3060 6GB) 和多卡 (Kaggle T4 x2) 的专用配置。
- **数据管线**: 包含从下载、清洗到 Tokenize 的完整数据处理脚本。
- **监控集成**: 支持 Weights & Biases (W&B) 进行实验监控和数据集版本管理。
- **本地优化**: 针对 Windows 和有限显存环境进行了专门适配。

## 🚀 快速开始

### 1. 环境安装

本项目使用 `uv` 进行依赖管理（推荐），也支持标准 pip。

```bash
# 安装 uv (如果尚未安装)
pip install uv

# 同步依赖 (会自动创建 .venv 虚拟环境)
uv sync

# 激活虚拟环境 (Windows Git Bash)
source .venv/Scripts/activate
# 或者 Windows CMD
# .venv\Scripts\activate.bat
```

### 2. 数据准备

```bash
# 下载 TinyStories 数据集
python scripts/download_tinystories.py

# 预处理与 Tokenize (生成 index.json 索引)
python prepare_data.py --data-dir data/custom_text
```

### 3. 开始训练

#### 💻 本地单卡训练 (推荐 RTX 3060/4060 等)

使用我们专门优化的配置文件 `configs/optimized_rtx3060.yaml`，该配置针对 6GB+ 显存进行了优化（单卡、低 Batch Size、梯度累积）。

```bash
# 训练完整模型 (200M 参数)
python run_train.py --model-config configs/moe_200m.yaml --train-config configs/optimized_rtx3060.yaml

# 快速调试 (30M 参数，启动更快)
python run_train.py --model-config configs/moe_30m_debug.yaml --train-config configs/optimized_rtx3060.yaml
```

#### ☁️ 云端/多卡训练

```bash
# 使用 Kaggle T4 x2 配置
python run_train.py --train-config configs/kaggle_t4_ddp.yaml
```

### 4. 模型生成与评估

```bash
# 文本生成测试
python generate.py --prompt "Once upon a time" --checkpoint_dir checkpoints/final

# 评估模型
python evaluate.py --checkpoint_dir checkpoints/final
```

## 📂 项目结构

```text
litgpt_project/
├── configs/                 # 配置文件目录
│   ├── optimized_rtx3060.yaml # [新增] 本地单卡优化配置
│   ├── kaggle_t4_ddp.yaml   # Kaggle 双卡 DDP 配置
│   ├── moe_30m_debug.yaml   # 调试用小模型配置
│   └── moe_200m.yaml        # 默认 200M 模型配置
├── data/                    # 数据目录 (自动生成)
├── docs/                    # 文档与报告
│   └── reports/             # 历史修复报告与技术文档
├── scripts/                 # 辅助脚本
│   ├── download_tinystories.py # 数据集下载
│   ├── generate_index_json.py  # 索引生成工具
│   ├── test_compile.py         # 编译测试
│   ├── env_sanity_check.py     # 环境检查
│   └── verify_flash.py         # Flash Attention 验证
├── src/                     # 源代码模块
│   └── litgpt_moe/          # [新增] 核心包
│       ├── fixed_text_files.py     # 修复版数据加载器
│       ├── wandb_dataset.py        # W&B 数据集集成
│       ├── config.py               # MoE 配置类
│       └── utils.py                # 通用工具
├── prepare_data.py          # 数据预处理入口
├── run_train.py             # 训练主程序
├── generate.py              # 生成脚本
├── evaluate.py              # 评估脚本
└── pyproject.toml           # 项目依赖定义
```

## ⚙️ 配置说明

### 训练配置 (`configs/*.yaml`)

| 参数 | 说明 | 推荐值 (本地) |
|------|------|---------------|
| `devices` | 使用 GPU 数量 | `1` |
| `micro_batch_size` | 单次前向传播的样本数 (显存敏感) | `2` 或 `4` |
| `global_batch_size` | 梯度累积后的总 Batch Size | `64` 或 `128` |
| `gradient_checkpointing` | 梯度检查点 (节省显存) | `true` |
| `num_workers` | 数据加载进程数 | Windows设为 `0` |

### 模型架构

默认模型配置 (`configs/moe_200m.yaml`)：
- **总参数量**: ~200M
- **专家数**: 8 (Top-2 激活)
- **层数**: 12
- **隐藏层维度**: 768

## 🛠️ 常见问题与修复

1. **Windows 下报错 `BrokenPipeError` 或 DataLoader 卡住**
   - **解决**: 确保在配置中设置 `num_workers: 0`。`local_rtx3060.yaml` 已默认包含此设置。

2. **OOM (Out of Memory)**
   - **解决**: 减小 `micro_batch_size` (如设为 1)，或者使用更小的模型配置 (`moe_30m_debug.yaml`)。

3. **`AttributeError: 'Config' object has no attribute 'moe_...'`**
   - **解决**: 请确保使用最新的 `run_train.py`，我们已修复了 MoE 参数注入的逻辑。

4. **恢复训练失败 (Size Mismatch)**
   - **解决**: 确保 `resume` 参数设置为 `null` (在配置文件中) 以重新开始训练，或者指定正确的 checkpoint 路径。不同模型配置产生的 checkpoint 不兼容。

## 📊 监控

支持 Weights & Biases 监控。设置环境变量或在 `prepare_data.py` 中启用：

```bash
export WANDB_PROJECT="litgpt-moe"
python prepare_data.py --log-to-wandb
```
