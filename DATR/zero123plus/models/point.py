from typing import Optional

import torch
import torch.nn as nn
from diffusers.utils import logging
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from zero123plus.models.transformer_1d import Transformer1D


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TransformerPointTokenizer(ModelMixin, ConfigMixin):
    """
    A Transformer-based point tokenizer, compatible with the Diffusers pipeline.
    """
    
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 32,
        in_channels: Optional[int] = 3,
        out_channels: Optional[int] = 1024,
        num_layers: int = 12,
        norm_num_groups: int = 32,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()
        
        transformer_cfg = {
            "num_attention_heads": num_attention_heads,
            "attention_head_dim": attention_head_dim,
            "in_channels": num_attention_heads * attention_head_dim,
            "num_layers": num_layers,
            "norm_num_groups": norm_num_groups,
            "attention_bias": attention_bias,
            "activation_fn": activation_fn,
            "norm_elementwise_affine": norm_elementwise_affine,
        }

        self.model = Transformer1D(**transformer_cfg)
        self.linear_in = nn.Linear(in_channels, transformer_cfg["in_channels"])
        self.linear_out = nn.Linear(transformer_cfg["in_channels"], out_channels)

    def forward(self, points: torch.Tensor, **kwargs) -> torch.Tensor:
        """Tokenizes input points using the transformer model."""
        assert points.ndim == 3
        inputs = self.linear_in(points).permute(0, 2, 1)  # B N Ci -> B Ci N
        out = self.model(inputs).permute(0, 2, 1)  # B Ci N -> B N Ci
        return self.linear_out(out)  # B N Ci -> B N Co

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError