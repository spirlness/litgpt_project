# 模型架构评估（更新版）

## 范围

本文档描述当前仓库中可直接追踪到的模型与训练实现，避免使用机器相关绝对路径。

主要代码位置：
- 训练入口：`run_train.py`
- 生成入口：`generate.py`
- 评估入口：`evaluate.py`
- MoE 配置扩展：`src/litgpt_moe/config.py`
- 数据模块：`src/litgpt_moe/fixed_text_files.py`
- 运行时与 Flash 配置：`src/litgpt_moe/utils.py`
- 模型配置：`configs/moe_200m.yaml`

## 当前默认模型（MoE-200M）

来自 `configs/moe_200m.yaml`：
- `n_layer=12`
- `n_embd=768`
- `n_head=12`
- `n_query_groups=4`
- `mlp_class_name=LLaMAMoE`
- `n_expert=8`
- `n_expert_per_token=2`
- `moe_intermediate_size=2048`
- `block_size=2048`

## 架构要点

1. 基础骨架是 LitGPT 解码器结构，项目侧通过配置驱动模型大小与 MoE 参数。
2. MoE 扩展在项目中通过 `MoEConfig` 注入额外字段：
   - `moe_aux_loss_weight`
   - `moe_router_stats`
3. 训练中会读取 router 统计并可将 `aux_loss` 加到总 loss（见 `run_train.py` 的训练循环）。

## 训练行为说明

1. `run_train.py` 支持从 YAML 读取优化开关，包括 Flash Attention 和 `torch.compile`。
2. 当前实现中，MoE 模型默认禁用 `torch.compile`（兼容性保护逻辑）。
3. Flash Attention backend 由 `configure_flash_attention()` 根据配置控制。

## 主要风险与关注点

1. MoE 训练稳定性依赖路由负载分布，建议持续观察 `router_stats` 指标。
2. 不同 GPU/驱动下 Flash backend 可用性可能不同，建议开启 `flash_attention_force` 做硬失败检查。
3. 当恢复训练时，需保证模型配置与 checkpoint 一致，避免权重 shape mismatch。

## 结论

当前代码结构清晰，入口脚本与配置文件分工明确。文档建议以 `configs/*.yaml + run_train.py` 为事实来源，不再依赖历史的绝对路径报告。
