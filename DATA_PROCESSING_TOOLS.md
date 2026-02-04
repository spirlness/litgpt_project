# 数据处理工具说明

## 概述

本项目包含几个用于处理和验证训练数据的工具脚本：

1. `prepare_data.py` - 主要的数据处理脚本
2. `generate_index_json.py` - 专门用于生成和验证index.json文件的工具
3. `enhanced_prepare_data.py` - 增强版数据处理脚本（实验性）

## prepare_data.py

这是主要的数据处理脚本，负责：

- 下载和准备tokenizer
- 处理训练和验证文本数据
- 生成tokenized数据chunk
- 自动生成index.json文件

### 使用方法

```bash
# 基本使用
uv run python prepare_data.py

# 指定数据目录
uv run python prepare_data.py --data-dir ./my_data

# 使用更多工作进程（默认2个）
uv run python prepare_data.py --workers 4

# 强制重新处理
uv run python prepare_data.py --force-reprocess
```

## generate_index_json.py

这是一个专门用于处理index.json文件的工具，可以验证现有文件或重新生成缺失/损坏的文件。

### 功能特性

- 验证index.json文件的有效性
- 重新生成缺失或损坏的index.json文件
- 支持多种数据目录同时处理
- 兼容litdata生成的文件格式

### 使用方法

```bash
# 验证现有index.json文件
uv run python generate_index_json.py --verify-only data/custom_text/train/train data/custom_text/val/val

# 生成缺失的index.json文件
uv run python generate_index_json.py data/custom_text/train/train data/custom_text/val/val

# 强制重新生成index.json文件
uv run python generate_index_json.py --force-regenerate data/custom_text/train/train

# 指定样本数估计
uv run python generate_index_json.py --samples 500 data/custom_text/train/train
```

## 常见问题解决

### 1. "Missing index.json file" 错误

如果遇到以下错误：
```
ValueError: The provided dataset `/path/to/data` doesn't contain any index.json file.
```

解决方法：
```bash
# 使用我们的工具重新生成index.json
uv run python generate_index_json.py /path/to/data/train/train /path/to/data/val/val
```

### 2. 数据处理卡住或过慢

问题原因：使用了过多的CPU工作进程

解决方法：
```bash
# 限制工作进程数量
uv run python prepare_data.py --workers 2
```

### 3. 需要重新处理数据

```bash
# 强制重新处理所有数据
uv run python prepare_data.py --force-reprocess
```

## 文件结构

处理后的数据目录结构应该是这样的：

```
data/custom_text/
├── train/
│   └── train/
│       ├── chunk-0-0.bin
│       ├── chunk-1-0.bin
│       └── index.json
└── val/
    └── val/
        ├── chunk-0-0.bin
        ├── chunk-1-0.bin
        └── index.json
```

## 最佳实践

1. **限制工作进程数量**：避免使用过多CPU核心，建议使用2-4个工作进程
2. **定期验证数据完整性**：使用`generate_index_json.py --verify-only`检查文件
3. **备份重要数据**：在重新处理前备份现有的处理数据
4. **监控磁盘空间**：数据处理可能需要大量临时存储空间

## 故障排除

### 检查数据目录状态

```bash
# 查看数据目录内容
ls -la data/custom_text/train/train/
ls -la data/custom_text/val/val/

# 验证index.json文件
uv run python generate_index_json.py --verify-only data/custom_text/train/train
```

### 清理和重新开始

```bash
# 删除处理后的数据（谨慎操作！）
rm -rf data/custom_text/train/train/*
rm -rf data/custom_text/val/val/*

# 重新处理数据
uv run python prepare_data.py
```