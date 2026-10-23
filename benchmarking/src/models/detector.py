import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from src.models.cascade_roi_heads import CascadeRoIHeads
from src.models.multimodal_vit import MultimodalViT


class SimpleFPN(nn.Module):
    """ViTDet simple feature pyramid from the stride-16 token grid (strides 4 to 32)."""

    def __init__(self, embedding_dim, out_channels=256):
        super().__init__()
        self.lateral_conv = nn.Conv2d(embedding_dim, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(32, out_channels)

        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        p4 = self.norm(self.lateral_conv(x))
        p5 = F.max_pool2d(p4, kernel_size=2, stride=2)
        p3 = F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = F.interpolate(p3, scale_factor=2, mode="nearest")

        return {
            "0": self.smooth_p2(p2),
            "1": self.smooth_p3(p3),
            "2": self.smooth_p4(p4),
            "3": self.smooth_p5(p5),
        }


class MultimodalFPNBackbone(MultimodalViT):
    """Multimodal ViT + FPN with torchvision's backbone interface: the modalities arrive
    stacked along the channel dimension in a single tensor."""

    def __init__(self, config, modalities: list[str], modality_channels: dict[str, int]):
        super().__init__(
            config["backbone"], modalities, modality_channels, config.get("drop_path_rate", 0.0)
        )
        self.backbone.encoder.gradient_checkpointing = True

        self.fpn = SimpleFPN(self.hidden_dim, out_channels=256)
        self.out_channels = 256
        self._load_weights(config["backbone"])

    def forward(self, x):
        sizes = [self.modality_channels[modality] for modality in self.modalities]
        fused, grid_h, grid_w = self.encode(torch.split(x, sizes, dim=1))
        fused = fused.permute(0, 2, 1).reshape(x.shape[0], self.hidden_dim, grid_h, grid_w)
        return self.fpn(fused)


class MultimodalCascadeRCNN(nn.Module):
    """torchvision Faster R-CNN (anchors, RPN, RoI pooling) on the multimodal ViT + simple
    feature pyramid, with the RoI heads replaced by Cascade R-CNN heads when
    head_type: cascade_rcnn (all experiments)."""

    def __init__(self, config):
        super().__init__()
        self.modalities = config["modalities"]
        self.modality_channels = config["modality_channels"]
        self.head_type = config.get("head_type", "faster_rcnn")

        self.backbone = MultimodalFPNBackbone(
            config, modalities=self.modalities, modality_channels=self.modality_channels
        )

        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 4
        )
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )

        num_classes = config["num_classes"] + 1  # + background
        total_channels = sum(self.modality_channels.values())

        self.model = FasterRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=min(config["image_size"]),
            max_size=max(config["image_size"]),
            # inputs are normalized by the data pipeline
            image_mean=[0.0] * total_channels,
            image_std=[1.0] * total_channels,
        )
        if self.head_type == "cascade_rcnn":
            self.model.roi_heads = CascadeRoIHeads(
                box_roi_pool=roi_pooler,
                pool_channels=self.backbone.out_channels,
                num_classes=num_classes,
            )

    def forward(self, inputs, targets=None):
        """inputs: {modality: [B, C, H, W]}; targets: list of {"boxes", "labels"}."""
        stacked = torch.cat([inputs[modality] for modality in self.modalities], dim=1)
        return self.model(list(stacked.unbind(0)), targets)
