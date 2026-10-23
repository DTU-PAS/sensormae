import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.multimodal_vit import MultimodalViT


class SegmenterDecoder(nn.Module):
    """Segmenter mask transformer: learnable class embeddings decoded jointly with the
    patch tokens, masks are patch-class dot products."""

    def __init__(self, embed_dim, num_classes, num_layers=2, n_heads=8):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.class_embeddings = nn.Parameter(torch.randn(1, num_classes, embed_dim))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_embeddings):
        b, s, _ = patch_embeddings.shape
        cls_emb = self.class_embeddings.expand(b, -1, -1)

        x = self.decoder(torch.cat((patch_embeddings, cls_emb), dim=1))
        x = self.norm(x)

        patches, classes = x[:, :s, :], x[:, s:, :]
        return patches @ classes.transpose(1, 2)


class Segmenter(MultimodalViT):
    def __init__(
        self,
        backbone_name,
        num_classes,
        modalities: list[str],
        modality_channels: dict[str, int],
        drop_path_rate: float = 0.0,
    ):
        super().__init__(backbone_name, modalities, modality_channels, drop_path_rate)
        self.decoder = SegmenterDecoder(embed_dim=self.hidden_dim, num_classes=num_classes)
        self._load_weights(backbone_name)

    def forward(self, inputs):
        images = [inputs[modality] for modality in self.modalities]
        h, w = images[0].shape[2:]

        fused, grid_h, grid_w = self.encode(images)
        masks = self.decoder(fused)

        b, _, c_out = masks.shape
        masks = masks.permute(0, 2, 1).reshape(b, c_out, grid_h, grid_w)
        return F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
