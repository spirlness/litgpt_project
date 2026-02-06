# 模型架构评估报告（含 MoE 路由与负载均衡）

## 1. 评估范围与依据
**范围**：本仓库代码与其本地依赖（`.venv` 中的 `litgpt` 实现）。

**主要证据文件**（含行号）：
- 核心模型实现：`C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:22-535`
- MoE 路由实现：`C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:763-850`
- 配置结构：`C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\config.py:25-248`
- 项目模型配置：`C:\Users\Administrator\litgpt_project\model_config.yaml:1-18`
- 训练入口：`C:\Users\Administrator\litgpt_project\run_train.py:82-121`
- 推理入口：`C:\Users\Administrator\litgpt_project\generate.py:70-163`
- 评估入口：`C:\Users\Administrator\litgpt_project\evaluate.py:40-67`
- MoE FLOPs 相关补丁：`C:\Users\Administrator\litgpt_project\src\utils.py:155-185`

---

## 2. 当前模型配置（MoE‑200M）
来自 `model_config.yaml`：
- 模型名：`MoE-200M`（`model_config.yaml:2`）
- 结构：12 层、隐藏维 768、注意力头 12（`model_config.yaml:7-11`）
- GQA：`n_query_groups=4`（`model_config.yaml:12`）
- MoE：`n_expert=8`, `n_expert_per_token=2`, `moe_intermediate_size=2048`（`model_config.yaml:6-9`）
- Norm：`RMSNorm`（`model_config.yaml:13-14`）
- 其它：`block_size=2048`（`model_config.yaml:4`），`parallel_residual=false`（`model_config.yaml:16`）

---

## 3. 架构概览

### 3.1 GPT 主体结构
- `GPT` 组装：token embedding、Transformer Blocks、最终归一化、LM Head
  `C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:22-181`

### 3.2 Block 结构（残差与归一化路径）
- `Block` 中使用 `parallel_residual` 控制残差路径
  `C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:263-341`

### 3.3 注意力实现（GQA + RoPE）
- GQA 通过 `n_query_groups` 实现（Q 与 KV 头数量解耦）
  `C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:408-479`
- RoPE 缓存与应用
  `C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:187-223`, `853-925`

### 3.4 MLP / MoE
- MLP 类型通过 `Config.mlp_class_name` 选择
  `C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\config.py:88-101`, `223-227`
- MoE 实现为 `LLaMAMoE`
  `C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:763-806`

---

## 4. MoE 路由与负载均衡评估（核心）

### 4.1 路由实现（Top‑k）
**非分组路由**（默认）：
- 线性 gate 产生专家打分
- `topk` 选择 `n_expert_per_token`
- **对 top‑k 结果做 softmax**
- 掩码 + `torch.where` 聚合专家输出
对应代码：
`C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:763-802`

**分组路由**（GroupedTopkRouter）：
- `sigmoid` 得分
- 先在 expert group 维度 top‑k，再在组内 top‑k
- `norm_topk_prob` 可选归一化
对应代码：
`C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:809-850`

### 4.2 配置支持但未体现负载均衡机制
配置项包括：
- `n_expert`, `n_expert_per_token`, `n_expert_groups`, `n_topk_groups`, `n_topk_scores_per_group`
- `routed_scaling_factor`, `norm_topk_prob`
来源：
`C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\config.py:88-101`, `176-182`

### 4.3 已确认缺失的负载均衡机制
在仓库内未发现以下机制实现或指标：
- Aux / load‑balancing loss（如 router_loss、aux_loss、z_loss、entropy）
- capacity factor / token dropping
- 专家并行 all‑to‑all dispatch/combiner
- 专家利用率日志/统计

**结论**：当前 MoE 路由为**基础 top‑k 路由**，**缺失显式负载均衡约束**。

---

## 5. 性能 / 吞吐 / 延迟（与 MoE 关联）

### 5.1 正向优势
- GQA 减少 KV head 数，降低注意力显存与带宽压力
  `C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:408-479`
- SDPA/Flash Attention 兼容（当无 mask/softcapping）
  `C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:516-535`

### 5.2 MoE 路由带来的性能风险
- 聚合实现为 **按专家循环** + `torch.where`（非融合 kernel），在大 batch 上可能出现调度与内存开销
  `C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\model.py:796-802`
- **无容量控制**，极端负载下性能退化风险上升

---

## 6. 可扩展性 / 可维护性

**优势**：
- 配置集中化，字段与校验完善
  `C:\Users\Administrator\litgpt_project\.venv\Lib\site-packages\litgpt\config.py:25-248`
- 训练/推理/评估入口结构清晰
  `run_train.py:82-121`, `generate.py:70-110`, `evaluate.py:40-47`

**风险**：
- 核心 MoE 实现位于 `.venv` 依赖路径，深度改造需管理依赖版本或进行本地 vendoring
  `...\.venv\Lib\site-packages\litgpt\model.py`
- 训练中存在 monkey‑patch（如梯度检查点补丁），对升级更敏感
  `C:\Users\Administrator\litgpt_project\src\utils.py:235-257`

---

## 7. 质量与鲁棒性（数值稳定）

**稳定性基础**：
- RMSNorm + RoPE 是当前主流稳定配置
  `model_config.yaml:13-18`, `...litgpt\model.py:853-925`

**MoE 特有风险**：
- 缺失负载均衡损失 → **专家塌缩**与**负载偏置**风险高
- 分组路由默认 `sigmoid` 权重可能不归一（除非 `norm_topk_prob`）
  `...litgpt\model.py:842-849`

---

## 8. 风险清单（MoE 路由/负载均衡）

| 风险 | 影响 | 证据 |
|---|---|---|
| 专家负载不均衡 | 训练不稳定、吞吐下降 | 未发现 aux loss / capacity / token drop |
| 专家塌缩 | 少量专家被过度使用 | 路由仅 top‑k，无均衡约束 |
| 性能瓶颈 | 大 batch 时路由开销偏高 | `torch.where` + per‑expert loop (`model.py:796-802`) |
| 权重归一不确定 | 输出尺度波动 | `norm_topk_prob` 可选 (`model.py:847-849`) |

---

## 9. 建议的评估指标（无需改代码即可定义）

**负载均衡指标**（评估时重点关注）：
1. **专家负载分布**：每 expert 处理的 token 数
2. **负载不均衡系数**：CoV 或 Gini
3. **路由熵**：top‑k 权重熵的均值
4. **最大负载比例**：max / mean
5. **负载漂移**：随 step 的分布漂移

---

## 10. 结论摘要

- 当前模型架构是 **标准解码器 GPT + MoE MLP**，配置驱动，结构清晰。
- MoE 路由是**基础 top‑k 路由**，具备分组路由支持，但**缺少显式负载均衡机制**。
- 在实际训练中，负载不均衡与专家塌缩是主要风险；性能方面，路由实现可能成为瓶颈。
