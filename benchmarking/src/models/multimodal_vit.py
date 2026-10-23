import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

from src.models.utils import inject_drop_path, resolve_vit_config
from src.weight_adapters import load_multimae_weights, load_sensormae_weights, load_standard_weights


def _strip_cls(pos_embed):
    """Drop a leading CLS position if the grid is not square without it."""
    n = pos_embed.shape[1]
    if int(math.sqrt(n)) ** 2 != n and int(math.sqrt(n - 1)) ** 2 == n - 1:
        return pos_embed[:, 1:, :]
    return pos_embed


class MultimodalViT(nn.Module):
    """Shared ViT encoder with an independent patch adapter, positional embedding and
    modality embedding per input modality. Tokens of all modalities are encoded jointly,
    then concatenated along the channel dimension and fused by a linear layer.

    Subclasses add their head and then call `_load_weights`.
    """

    def __init__(self, backbone_name, modalities, modality_channels, drop_path_rate=0.0):
        super().__init__()
        self.modalities = modalities
        self.modality_channels = modality_channels

        self.config = resolve_vit_config(backbone_name)
        self.config.mask_ratio = 0.0
        self.backbone = ViTModel(self.config)

        self.hidden_dim = self.config.hidden_size
        self.patch_size = self.config.patch_size

        if drop_path_rate > 0.0:
            inject_drop_path(self.backbone, drop_path_rate)

        for modality in modalities:
            adapter = nn.Conv2d(
                modality_channels[modality],
                self.hidden_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            self.register_module(f"{modality}_adapter", adapter)

        initial_pos = _strip_cls(self.backbone.embeddings.position_embeddings.data.clone())
        for modality in modalities:
            self.register_parameter(f"pos_embed_{modality}", nn.Parameter(initial_pos))
        for modality in modalities:
            self.register_parameter(
                f"modality_embed_{modality}", nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            )

        self.fusion_proj = nn.Linear(len(modalities) * self.hidden_dim, self.hidden_dim)

    def _load_weights(self, backbone_name):
        ckpt_file = os.path.basename(backbone_name).lower()
        if "sensormae" in ckpt_file:
            load_sensormae_weights(self, backbone_name, self.modalities)
        elif "multimae" in ckpt_file:
            load_multimae_weights(self, backbone_name, self.modalities, self.modality_channels)
        else:
            load_standard_weights(self, backbone_name, self.modalities, self.modality_channels)

    @staticmethod
    def _interpolate_pos_embed(pos_embed, grid_h, grid_w):
        """Bicubic resize of a square spatial positional embedding to (grid_h, grid_w)."""
        pos_embed = _strip_cls(pos_embed)
        n_src, dim = pos_embed.shape[1], pos_embed.shape[-1]
        grid_src = int(math.sqrt(n_src))

        if grid_src == grid_h and grid_src == grid_w:
            return pos_embed

        pos_embed = pos_embed.transpose(1, 2).reshape(1, dim, grid_src, grid_src)
        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        return pos_embed.flatten(2).transpose(1, 2)

    def encode(self, images):
        """images: one [B, C_m, H, W] tensor per modality (in `self.modalities` order).
        Returns the fused tokens [B, grid_h * grid_w, D] and the token grid size."""
        b = images[0].shape[0]

        emb = [getattr(self, f"{m}_adapter")(x) for m, x in zip(self.modalities, images)]
        grid_h, grid_w = emb[0].shape[2], emb[0].shape[3]
        emb = [values.flatten(2).transpose(1, 2) for values in emb]

        emb = [
            e
            + self._interpolate_pos_embed(getattr(self, f"pos_embed_{m}"), grid_h, grid_w)
            + getattr(self, f"modality_embed_{m}")
            for m, e in zip(self.modalities, emb)
        ]

        cls_token = self.backbone.embeddings.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_token, *emb), dim=1)

        outputs = self.backbone.encoder(tokens)
        tokens = self.backbone.layernorm(outputs.last_hidden_state)[:, 1:, :]

        fused = torch.cat(tokens.chunk(len(self.modalities), dim=1), dim=2)
        return self.fusion_proj(fused), grid_h, grid_w
