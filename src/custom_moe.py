import torch
import torch.nn as nn
import torch.nn.functional as F
from litgpt.model import LLaMAMLP


class FixedLLaMAMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_expert_groups = int(getattr(config, "n_expert_groups", 0) or 0)
        self.n_shared_expert = int(getattr(config, "n_shared_expert", 0) or 0)
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0) or 1.0)

        if not self.n_expert_groups:
            self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        else:
            import importlib

            litgpt_model = importlib.import_module("litgpt.model")
            router_cls = getattr(litgpt_model, "GroupedTopkRouter")
            self.gate = router_cls(config)

        self.experts = nn.ModuleList(
            LLaMAMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_expert)
        )

        if self.n_shared_expert:
            self.shared_experts = LLaMAMLP(
                config,
                intermediate_size=config.moe_intermediate_size * self.n_shared_expert,
            )

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        residual_x = x.clone()
        x = x.view(-1, C)

        if not self.n_expert_groups:
            router = self.gate(x)
            probs, indices = torch.topk(router, self.config.n_expert_per_token)
            probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        else:
            probs, indices = self.gate(x)

        if self.routed_scaling_factor != 1.0:
            probs = probs * self.routed_scaling_factor

        y = torch.zeros_like(x)
        for expert_idx in range(len(self.experts)):
            expert_mask = indices == expert_idx
            token_indices, k_indices = torch.where(expert_mask)
            expert_input = x[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            token_probs = probs[token_indices, k_indices]
            y.index_add_(0, token_indices, token_probs.unsqueeze(-1) * expert_output)

        y = y.view(B, T, C)

        if self.n_shared_expert:
            y = y + self.shared_experts(residual_x)

        return y


class SimplifiedLLaMAMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(
            LLaMAMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_expert)
        )

        if config.n_shared_expert:
            self.shared_experts = LLaMAMLP(
                config,
                intermediate_size=config.moe_intermediate_size * config.n_shared_expert,
            )

        self.config = config
        self.n_expert = config.n_expert
        self.n_expert_per_token = config.n_expert_per_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        residual_x = x.clone()
        x_flat = x.view(-1, C)

        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        topk_probs, topk_indices = torch.topk(router_probs, k=self.n_expert_per_token, dim=-1)
        topk_probs = topk_probs.to(dtype=x.dtype)

        y = torch.zeros_like(x_flat)

        for k in range(self.n_expert_per_token):
            expert_idx_for_each_token = topk_indices[:, k]
            prob_for_each_token = topk_probs[:, k].unsqueeze(-1)

            for expert_id in range(self.n_expert):
                mask = expert_idx_for_each_token == expert_id
                token_indices = mask.nonzero(as_tuple=True)[0]

                if token_indices.numel() > 0:
                    expert_input = x_flat[token_indices]
                    expert_output = self.experts[expert_id](expert_input)
                    token_probs = prob_for_each_token[token_indices]
                    y[token_indices] += token_probs * expert_output

        y = y.view(B, T, C)

        if self.config.n_shared_expert:
            y = y + self.shared_experts(residual_x)

        return y


class BatchedMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(
            LLaMAMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_expert)
        )

        if config.n_shared_expert:
            self.shared_experts = LLaMAMLP(
                config,
                intermediate_size=config.moe_intermediate_size * config.n_shared_expert,
            )

        self.config = config
        self.n_expert = config.n_expert
        self.n_expert_per_token = config.n_expert_per_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        residual_x = x.clone()
        x_flat = x.view(-1, C)

        router = self.gate(x_flat)
        probs, indices = torch.topk(router, self.n_expert_per_token)
        probs = probs.softmax(dim=-1, dtype=torch.float).to(dtype=x.dtype)

        expert_one_hot = F.one_hot(indices, num_classes=self.n_expert).to(dtype=x.dtype)
        expert_weights = probs.unsqueeze(-1) * expert_one_hot
        token_expert_weights = expert_weights.sum(dim=1)

        y = torch.zeros_like(x_flat)

        for expert_idx in range(self.n_expert):
            weights = token_expert_weights[:, expert_idx]
            mask = weights > 0
            if mask.any():
                expert_input = x_flat[mask]
                expert_weights = weights[mask]
                expert_output = self.experts[expert_idx](expert_input)
                y[mask] += expert_weights.unsqueeze(-1) * expert_output

        y = y.view(B, T, C)

        if self.config.n_shared_expert:
            y = y + self.shared_experts(residual_x)

        return y


class ManualMoELayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.n_embd, config.moe_intermediate_size, bias=config.bias),
                    nn.SiLU(),
                    nn.Linear(config.moe_intermediate_size, config.n_embd, bias=config.bias),
                )
                for _ in range(config.n_expert)
            ]
        )
        self.config = config
        self.n_expert = config.n_expert
        self.n_expert_per_token = config.n_expert_per_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        scores = self.gate(x_flat)
        top_k = min(self.n_expert_per_token, self.n_expert)
        topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=-1)
        topk_probs = F.softmax(topk_scores, dim=-1)

        y = torch.zeros_like(x_flat)

        for k in range(top_k):
            expert_indices = topk_indices[:, k]
            weights = topk_probs[:, k]

            for expert_id in range(self.n_expert):
                mask = expert_indices == expert_id
                token_idx = mask.nonzero(as_tuple=True)[0]

                if token_idx.numel() > 0:
                    expert_input = x_flat[token_idx]
                    expert_output = self.experts[expert_id](expert_input)
                    expert_weights = weights[token_idx]
                    y[token_idx] += expert_weights.unsqueeze(-1) * expert_output

        y = y.view(B, T, C)
        return y


__all__ = ["FixedLLaMAMoE", "SimplifiedLLaMAMoE", "BatchedMoE", "ManualMoELayer"]
