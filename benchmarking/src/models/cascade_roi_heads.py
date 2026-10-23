"""Cascade R-CNN box head on top of torchvision's RoI interfaces.

The design (3 stages with IoU thresholds 0.5 / 0.6 / 0.7, gradient scaling into the
backbone, conv-fc box head) follows Detectron2's Cascade R-CNN
(https://github.com/facebookresearch/detectron2) and Cai & Vasconcelos (2018).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision.ops import batched_nms, box_iou, clip_boxes_to_image


class _ScaleGradient(Function):
    """Identity forward, gradients scaled by `scale` backward."""

    @staticmethod
    def forward(ctx, input_tensor, scale):
        ctx.scale = scale
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class FastRCNNConvFCHead(nn.Module):
    """4x (3x3 conv, GroupNorm, GELU) followed by one fully connected layer."""

    def __init__(self, pool_channels, output_size, conv_dims=256, num_conv=4, fc_dim=1024):
        super().__init__()
        in_channels = pool_channels
        convs = []
        for _ in range(num_conv):
            convs.append(nn.Conv2d(in_channels, conv_dims, kernel_size=3, padding=1, bias=False))
            convs.append(nn.GroupNorm(32, conv_dims))
            convs.append(nn.GELU())
            in_channels = conv_dims
        self.convs = nn.Sequential(*convs)
        self.fc = nn.Linear(conv_dims * output_size * output_size, fc_dim)
        self.out_dim = fc_dim

    def forward(self, x):
        x = self.convs(x)
        x = x.flatten(start_dim=1)
        return F.gelu(self.fc(x))


def _encode_box_targets(gt_boxes, proposals, weights):
    """xyxy ground truth -> (dx, dy, dw, dh) regression targets relative to the proposals."""
    wx, wy, ww, wh = weights

    p_w = (proposals[:, 2] - proposals[:, 0]).clamp(min=1.0)
    p_h = (proposals[:, 3] - proposals[:, 1]).clamp(min=1.0)
    p_cx = proposals[:, 0] + 0.5 * p_w
    p_cy = proposals[:, 1] + 0.5 * p_h

    g_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
    g_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)
    g_cx = gt_boxes[:, 0] + 0.5 * g_w
    g_cy = gt_boxes[:, 1] + 0.5 * g_h

    return torch.stack([
        wx * (g_cx - p_cx) / p_w,
        wy * (g_cy - p_cy) / p_h,
        ww * torch.log(g_w / p_w),
        wh * torch.log(g_h / p_h),
    ], dim=-1)


def _decode_box_deltas(deltas, proposals, weights):
    """Inverse of `_encode_box_targets`: deltas -> absolute xyxy boxes."""
    wx, wy, ww, wh = weights

    p_w = (proposals[:, 2] - proposals[:, 0]).clamp(min=1.0)
    p_h = (proposals[:, 3] - proposals[:, 1]).clamp(min=1.0)
    p_cx = proposals[:, 0] + 0.5 * p_w
    p_cy = proposals[:, 1] + 0.5 * p_h

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(max=math.log(1000.0 / 16))
    dh = (deltas[:, 3] / wh).clamp(max=math.log(1000.0 / 16))

    pred_cx = dx * p_w + p_cx
    pred_cy = dy * p_h + p_cy
    pred_w = torch.exp(dw) * p_w
    pred_h = torch.exp(dh) * p_h

    return torch.stack([
        pred_cx - 0.5 * pred_w,
        pred_cy - 0.5 * pred_h,
        pred_cx + 0.5 * pred_w,
        pred_cy + 0.5 * pred_h,
    ], dim=-1)


class CascadeRoIHeads(nn.Module):
    """Three-stage Cascade R-CNN with class-agnostic box regression.

    Drop-in replacement for torchvision's RoIHeads inside GeneralizedRCNN.
    """

    def __init__(
        self,
        box_roi_pool,
        pool_channels,
        num_classes,
        representation_size=1024,
        batch_size_per_image=512,
        positive_fraction=0.25,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    ):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.num_classes = num_classes
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.num_stages = 3
        self.stage_iou_thresholds = [0.5, 0.6, 0.7]
        # (wx, wy, ww, wh) normalization of the regression targets per stage
        self.stage_bbox_reg_weights = [
            (10.0, 10.0, 5.0, 5.0),
            (20.0, 20.0, 10.0, 10.0),
            (30.0, 30.0, 15.0, 15.0),
        ]

        out_h, out_w = box_roi_pool.output_size

        self.box_heads = nn.ModuleList()
        self.cls_scores = nn.ModuleList()
        self.bbox_preds = nn.ModuleList()
        for _ in range(self.num_stages):
            head = FastRCNNConvFCHead(
                pool_channels, out_h, conv_dims=256, num_conv=4, fc_dim=representation_size
            )
            self.box_heads.append(head)
            self.cls_scores.append(nn.Linear(representation_size, num_classes))
            self.bbox_preds.append(nn.Linear(representation_size, 4))

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            if targets is None:
                raise ValueError("targets required during training")
            return self._forward_train(features, proposals, image_shapes, targets)
        return self._forward_eval(features, proposals, image_shapes)

    def _sample_initial_proposals(self, proposals, targets):
        """Add the GT boxes to the RPN proposals and sample a balanced fg/bg set (stage 0)."""
        sampled_proposals = []
        sampled_labels = []
        sampled_reg_targets = []
        for proposals_i, targets_i in zip(proposals, targets):
            gt_boxes = targets_i["boxes"]
            gt_labels = targets_i["labels"]

            all_proposals = torch.cat([proposals_i, gt_boxes], dim=0)

            if gt_boxes.shape[0] == 0:
                labels = torch.zeros(
                    all_proposals.shape[0], dtype=torch.long, device=all_proposals.device
                )
                matched_gt_boxes = torch.zeros_like(all_proposals)
            else:
                ious = box_iou(all_proposals, gt_boxes)
                matched_vals, matched_idxs = ious.max(dim=1)
                labels = torch.zeros(
                    all_proposals.shape[0], dtype=torch.long, device=all_proposals.device
                )
                pos_mask = matched_vals >= self.stage_iou_thresholds[0]
                labels[pos_mask] = gt_labels[matched_idxs[pos_mask]]
                matched_gt_boxes = gt_boxes[matched_idxs]

            pos = torch.where(labels > 0)[0]
            neg = torch.where(labels == 0)[0]
            num_pos = min(int(self.batch_size_per_image * self.positive_fraction), pos.numel())
            num_neg = min(self.batch_size_per_image - num_pos, neg.numel())
            perm_p = torch.randperm(pos.numel(), device=pos.device)[:num_pos]
            perm_n = torch.randperm(neg.numel(), device=neg.device)[:num_neg]
            inds = torch.cat([pos[perm_p], neg[perm_n]])

            sampled_proposals.append(all_proposals[inds])
            sampled_labels.append(labels[inds])
            sampled_reg_targets.append(
                _encode_box_targets(
                    matched_gt_boxes[inds],
                    all_proposals[inds],
                    self.stage_bbox_reg_weights[0],
                )
            )
        return sampled_proposals, sampled_labels, sampled_reg_targets

    def _match_stage_proposals(self, proposals, targets, stage):
        """Relabel the refined proposals with the stricter IoU threshold of `stage`."""
        matched_labels = []
        matched_reg_targets = []
        iou_threshold = self.stage_iou_thresholds[stage]
        weights = self.stage_bbox_reg_weights[stage]

        for proposals_i, targets_i in zip(proposals, targets):
            gt_boxes = targets_i["boxes"]
            gt_labels = targets_i["labels"]

            if gt_boxes.shape[0] == 0:
                labels = torch.zeros(
                    proposals_i.shape[0], dtype=torch.long, device=proposals_i.device
                )
                matched_gt_boxes = torch.zeros_like(proposals_i)
            else:
                ious = box_iou(proposals_i, gt_boxes)
                matched_vals, matched_idxs = ious.max(dim=1)
                labels = torch.zeros(
                    proposals_i.shape[0], dtype=torch.long, device=proposals_i.device
                )
                pos_mask = matched_vals >= iou_threshold
                labels[pos_mask] = gt_labels[matched_idxs[pos_mask]]
                matched_gt_boxes = gt_boxes[matched_idxs]

            matched_labels.append(labels)
            matched_reg_targets.append(
                _encode_box_targets(matched_gt_boxes, proposals_i, weights)
            )

        return matched_labels, matched_reg_targets

    def _refine_proposals(self, proposals, box_deltas, image_shapes, stage, filter_empty):
        """Apply the predicted deltas; optionally drop degenerate boxes (training only)."""
        weights = self.stage_bbox_reg_weights[stage]
        new_proposals = []
        offset = 0

        for img_idx, proposals_i in enumerate(proposals):
            num_boxes = proposals_i.shape[0]
            decoded = _decode_box_deltas(
                box_deltas[offset : offset + num_boxes], proposals_i, weights
            )
            decoded = clip_boxes_to_image(decoded, image_shapes[img_idx])
            if filter_empty:
                keep = (decoded[:, 2] > decoded[:, 0]) & (decoded[:, 3] > decoded[:, 1])
                decoded = decoded[keep]
            new_proposals.append(decoded)
            offset += num_boxes

        return new_proposals

    def _forward_train(self, features, proposals, image_shapes, targets):
        losses = {}

        current_proposals, current_labels, current_reg_targets = self._sample_initial_proposals(
            [p.detach() for p in proposals], targets
        )

        for stage in range(self.num_stages):
            box_features = self.box_roi_pool(features, current_proposals, image_shapes)
            box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_stages)
            box_features = self.box_heads[stage](box_features)
            class_logits = self.cls_scores[stage](box_features)
            box_deltas = self.bbox_preds[stage](box_features)

            all_labels = torch.cat(current_labels)
            all_reg_targets = torch.cat(current_reg_targets)
            if all_labels.numel() == 0:
                losses[f"loss_classifier_{stage}"] = class_logits.sum() * 0
                losses[f"loss_box_reg_{stage}"] = box_deltas.sum() * 0
                continue

            pos_mask = all_labels > 0

            losses[f"loss_classifier_{stage}"] = F.cross_entropy(class_logits, all_labels)

            # smooth L1 on the positives, normalized by the number of sampled proposals
            if pos_mask.any():
                losses[f"loss_box_reg_{stage}"] = (
                    F.smooth_l1_loss(
                        box_deltas[pos_mask],
                        all_reg_targets[pos_mask],
                        beta=0.0,
                        reduction="sum",
                    )
                    / max(all_labels.numel(), 1)
                )
            else:
                losses[f"loss_box_reg_{stage}"] = box_deltas.sum() * 0

            if stage < self.num_stages - 1:
                with torch.no_grad():
                    current_proposals = self._refine_proposals(
                        current_proposals,
                        box_deltas,
                        image_shapes,
                        stage,
                        filter_empty=True,
                    )
                    current_labels, current_reg_targets = self._match_stage_proposals(
                        current_proposals, targets, stage + 1
                    )

        return [], losses

    def _forward_eval(self, features, proposals, image_shapes):
        """Run all stages; average the per-stage softmax scores, boxes from the last stage."""
        current_proposals = proposals
        stage_scores = []

        for stage in range(self.num_stages):
            box_features = self.box_roi_pool(features, current_proposals, image_shapes)
            box_features = self.box_heads[stage](box_features)
            class_logits = self.cls_scores[stage](box_features)
            box_deltas = self.bbox_preds[stage](box_features)

            stage_scores.append(F.softmax(class_logits, dim=-1))

            current_proposals = self._refine_proposals(
                current_proposals,
                box_deltas,
                image_shapes,
                stage,
                filter_empty=False,
            )

        avg_scores = sum(stage_scores) / self.num_stages

        results = []
        offset = 0
        for img_idx in range(len(current_proposals)):
            n = current_proposals[img_idx].shape[0]
            boxes = current_proposals[img_idx]

            scores = avg_scores[offset : offset + n, 1:]  # drop background
            offset += n
            num_fg_classes = scores.shape[1]

            all_boxes, all_scores, all_labels = [], [], []
            for cls in range(num_fg_classes):
                cls_scores = scores[:, cls]
                keep = cls_scores > self.score_thresh
                if not keep.any():
                    continue
                all_boxes.append(boxes[keep])
                all_scores.append(cls_scores[keep])
                all_labels.append(
                    torch.full_like(cls_scores[keep], cls + 1, dtype=torch.long)
                )

            if all_boxes:
                all_boxes = torch.cat(all_boxes)
                all_scores = torch.cat(all_scores)
                all_labels = torch.cat(all_labels)
                keep = batched_nms(all_boxes, all_scores, all_labels, self.nms_thresh)
                keep = keep[: self.detections_per_img]
                results.append(
                    {"boxes": all_boxes[keep], "scores": all_scores[keep], "labels": all_labels[keep]}
                )
            else:
                results.append({
                    "boxes": torch.zeros(0, 4, device=boxes.device),
                    "scores": torch.zeros(0, device=boxes.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=boxes.device),
                })

        return results, {}
