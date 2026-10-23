import os
import types

import torch
import torch.nn as nn
from transformers import ViTConfig

# Local checkpoints carry no config.json: ViT-Base, resized when the file name says so
_VIT_SIZE_OVERRIDES = {
    "tiny": {"hidden_size": 192, "num_attention_heads": 3, "intermediate_size": 768},
    "small": {"hidden_size": 384, "num_attention_heads": 6, "intermediate_size": 1536},
    "base": {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 3072},
}


def resolve_vit_config(backbone_name: str) -> ViTConfig:
    """ViTConfig of a HF hub model, or a ViT-Base/Small/Tiny config for local .pth files."""
    try:
        return ViTConfig.from_pretrained(backbone_name)
    except Exception:
        config = ViTConfig.from_pretrained("facebook/vit-mae-base")
        ckpt_name = os.path.basename(backbone_name).lower()
        if "tiny" in ckpt_name:
            size = "tiny"
        elif "small" in ckpt_name:
            size = "small"
        else:
            size = "base"
        for key, value in _VIT_SIZE_OVERRIDES[size].items():
            setattr(config, key, value)
        return config


class DropPath(nn.Module):
    """Stochastic depth: drops the residual branch per sample (as timm's DropPath)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


def inject_drop_path(backbone, drop_path_rate: float = 0.0):
    """Add DropPath to both residual branches of every HF ViT layer, with a rate growing
    linearly from 0 (first layer) to `drop_path_rate` (last layer)."""
    if drop_path_rate <= 0.0:
        return

    encoder = backbone.encoder
    num_layers = len(encoder.layer)

    for layer_idx, layer in enumerate(encoder.layer):
        layer.drop_path = DropPath(drop_path_rate * layer_idx / max(num_layers - 1, 1))

        def _make_forward(ly):
            def _forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                attn_out = ly.attention(ly.layernorm_before(hidden_states))
                hidden_states = hidden_states + ly.drop_path(attn_out)

                mlp_out = ly.intermediate(ly.layernorm_after(hidden_states))
                mlp_out = ly.output.dense(mlp_out)
                mlp_out = ly.output.dropout(mlp_out)
                return hidden_states + ly.drop_path(mlp_out)

            return _forward

        layer.forward = types.MethodType(_make_forward(layer), layer)
