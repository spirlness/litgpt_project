from dataclasses import dataclass

from litgpt.config import Config as LitGPTConfig


@dataclass
class MoEConfig(LitGPTConfig):
    """
    Configuration class for Mixture of Experts (MoE) models.
    Inherits from litgpt.config.Config and adds MoE-specific fields.
    """
    moe_aux_loss_weight: float = 0.01
    moe_router_stats: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Ensure MoE specific fields are properly typed if loaded from dict
        if self.moe_aux_loss_weight is not None:
             self.moe_aux_loss_weight = float(self.moe_aux_loss_weight)
