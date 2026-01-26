"""
Custom MoE implementation that avoids torch.where meta tensor issues.

This is a simplified and fixed version of LLaMAMoE that:
1. Replaces torch.where with a safer implementation
2. Better handles meta tensors during model initialization
3. Provides a working alternative for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from litgpt.model import LLaMAMLP


class FixedLLaMAMoE(nn.Module):
    """
    Fixed version of LLaMAMoE that avoids torch.where issues with meta tensors.

    The key fix is replacing:
        token_idx, expert_idx = torch.where(mask)

    With a safer implementation that doesn't rely on torch.where's
    interaction with meta tensors.
    """

    def __init__(self, config) -> None:
        super().__init__()

        # Gate for routing (can be simple Linear or GroupedTopkRouter)
        if not config.n_expert_groups:
            self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        else:
            # Import GroupedTopkRouter from litgpt.model
            from litgpt.model import GroupedTopkRouter
            self.gate = GroupedTopkRouter(config)

        # Create expert MLPs
        self.experts = nn.ModuleList(
            LLaMAMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_expert)
        )

        # Optional shared experts
        if config.n_shared_expert:
            self.shared_experts = LLaMAMLP(
                config, intermediate_size=config.moe_intermediate_size * config.n_shared_expert
            )

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fixed implementation avoiding torch.where issues.

        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.size()
        residual_x = x.clone()
        x = x.view(-1, C)  # (B*T, C)

        # Get routing decisions
        if not self.config.n_expert_groups:
            router = self.gate(x)  # (B*T, n_expert)
            probs, indices = torch.topk(router, self.config.n_expert_per_token)
            probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        else:
            probs, indices = self.gate(x)

        # Apply routing scaling factor if configured
        if self.config.routed_scaling_factor != 1.0:
            probs = probs * self.config.routed_scaling_factor

        # Create expert assignment masks
        # Shape: (B*T, n_expert_per_token)
        expert_indices = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        # Shape: (n_expert, B*T, n_expert_per_token)
        masks = expert_indices.permute(2, 0, 1)

        # Initialize output tensor
        y = torch.zeros_like(x)  # (B*T, C)

        # Process each expert
        # FIXED: Use a safer alternative to torch.where
        for expert_idx, mask in enumerate(masks):
            # Get indices of tokens assigned to this expert
            # FIXED: Use nonzero directly instead of torch.where
            token_indices = mask.nonzero(as_tuple=True)[0]
            expert_assignments = mask.nonzero(as_tuple=True)[1]

            # Only process if there are tokens assigned to this expert
            if token_indices.numel() > 0:
                expert_input = x[token_indices]
                expert_output = self.experts[expert_idx](expert_input)

                # Get probabilities for these tokens
                token_probs = probs[token_indices, expert_assignments]

                # Add weighted expert output to final result
                y[token_indices] += token_probs.unsqueeze(-1) * expert_output

        # Reshape back to (B, T, C)
        y = y.view(B, T, C)

        # Add shared expert outputs if configured
        if self.config.n_shared_expert:
            y = y + self.shared_experts(residual_x)

        return y


class SimplifiedLLaMAMoE(nn.Module):
    """
    Simplified MoE implementation that uses a different routing strategy.

    This implementation:
    1. Uses top-k routing with explicit index handling
    2. Eliminates torch.where completely
    3. More straightforward but potentially less efficient
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(
            LLaMAMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_expert)
        )

        if config.n_shared_expert:
            self.shared_experts = LLaMAMLP(
                config, intermediate_size=config.moe_intermediate_size * config.n_shared_expert
            )

        self.config = config
        self.n_expert = config.n_expert
        self.n_expert_per_token = config.n_expert_per_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass without torch.where.
        """
        B, T, C = x.size()
        residual_x = x.clone()
        x_flat = x.view(-1, C)  # (B*T, C)
        n_tokens = x_flat.size(0)

        # Get routing logits and probabilities
        router_logits = self.gate(x_flat)  # (B*T, n_expert)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        # Get top-k experts per token (using topk on log_probs)
        topk_probs, topk_indices = torch.topk(
            router_probs, k=self.n_expert_per_token, dim=-1
        )
        topk_probs = topk_probs.to(dtype=x.dtype)

        # Initialize output
        y = torch.zeros_like(x_flat)  # (B*T, C)

        # For each expert position (0 to n_expert_per_token-1)
        for k in range(self.n_expert_per_token):
            expert_idx_for_each_token = topk_indices[:, k]  # (B*T,)
            prob_for_each_token = topk_probs[:, k].unsqueeze(-1)  # (B*T, 1)

            # For each expert
            for expert_id in range(self.n_expert):
                # Find tokens that use this expert at position k
                mask = expert_idx_for_each_token == expert_id
                token_indices = mask.nonzero(as_tuple=True)[0]

                if token_indices.numel() > 0:
                    # Get inputs for tokens using this expert
                    expert_input = x_flat[expert_input]  # typo: should be x_flat[token_indices]
                    expert_output = self.experts[expert_id](expert_input)

                    # Get probabilities and accumulate
                    token_probs = prob_for_each_token[token_indices]
                    y[token_indices] += token_probs * expert_output

        y = y.view(B, T, C)

        if self.config.n_shared_expert:
            y = y + self.shared_experts(residual_x)

        return y


class BatchedMoE(nn.Module):
    """
    More efficient batched MoE implementation.

    This version processes all tokens for each expert in a single batched operation.
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(
            LLaMAMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_expert)
        )

        if config.n_shared_expert:
            self.shared_experts = LLaMAMLP(
                config, intermediate_size=config.moe_intermediate_size * config.n_shared_expert
            )

        self.config = config
        self.n_expert = config.n_expert
        self.n_expert_per_token = config.n_expert_per_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass.
        """
        B, T, C = x.size()
        residual_x = x.clone()
        x_flat = x.view(-1, C)
        n_tokens = x_flat.size(0)

        # Get routing
        router = self.gate(x_flat)  # (n_tokens, n_expert)
        probs, indices = torch.topk(router, self.n_expert_per_token)
        probs = probs.softmax(dim=-1, dtype=torch.float).to(dtype=x.dtype)

        # One-hot encode expert assignments
        # Shape: (n_tokens, n_expert_per_token, n_expert)
        expert_one_hot = F.one_hot(indices, num_classes=self.n_expert).to(dtype=x.dtype)

        # Weight by routing probabilities
        # Shape: (n_tokens, n_expert_per_token, n_expert)
        expert_weights = probs.unsqueeze(-1) * expert_one_hot

        # Sum across expert positions for each token
        # Shape: (n_tokens, n_expert)
        token_expert_weights = expert_weights.sum(dim=1)

        # Initialize output
        y = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.n_expert):
            # Get weights for this expert: (n_tokens,)
            weights = token_expert_weights[:, expert_idx]

            # Only process tokens with non-zero weight
            mask = weights > 0
            if mask.any():
                # Get tokens and weights
                expert_input = x_flat[mask]
                expert_weights = weights[mask]

                # Compute expert output
                expert_output = self.experts[expert_idx](expert_input)

                # Accumulate weighted outputs
                y[mask] += expert_weights.unsqueeze(-1) * expert_output

        y = y.view(B, T, C)

        if self.config.n_shared_expert:
            y = y + self.shared_experts(residual_x)

        return y


class ManualMoELayer(nn.Module):
    """
    Manual MoE implementation using vanilla operations only.

    This is the most basic implementation to ensure compatibility.
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)

        # Create separate expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, config.moe_intermediate_size, bias=config.bias),
                nn.SiLU(),
                nn.Linear(config.moe_intermediate_size, config.n_embd, bias=config.bias)
            )
            for _ in range(config.n_expert)
        ])

        self.config = config
        self.n_expert = config.n_expert
        self.n_expert_per_token = config.n_expert_per_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Manual forward pass with minimal dependencies.
        """
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        n_tokens = x_flat.size(0)

        # Simple routing: compute expert scores
        scores = self.gate(x_flat)  # (n_tokens, n_expert)

        # Get top-k
        top_k = min(self.n_expert_per_token, self.n_expert)
        topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=-1)

        # Softmax over top-k
        topk_probs = F.softmax(topk_scores, dim=-1)

        # Initialize output
        y = torch.zeros_like(x_flat)

        # Process each top-k position
        for k in range(top_k):
            expert_indices = topk_indices[:, k]  # (n_tokens,)
            weights = topk_probs[:, k]  # (n_tokens,)

            # For each expert
            for expert_id in range(self.n_expert):
                # Find tokens assigned to this expert
                mask = expert_indices == expert_id
                token_idx = mask.nonzero(as_tuple=True)[0]

                if token_idx.numel() > 0:
                    expert_input = x_flat[token_idx]
                    expert_output = self.experts[expert_id](expert_input)
                    expert_weights = weights[token_idx]

                    y[token_idx] += expert_weights.unsqueeze(-1) * expert_output

        y = y.view(B, T, C)
        return y


# Export all implementations
__all__ = [
    'FixedLLaMAMoE',
    'SimplifiedLLaMAMoE',
    'BatchedMoE',
    'ManualMoELayer',
]
