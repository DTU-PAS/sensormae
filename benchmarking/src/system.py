import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes, make_grid

from src.models.segmenter import Segmenter
from src.models.detector import MultimodalCascadeRCNN

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


class BaseSystem(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config


def _warmup_then(optimizer, total_steps, warmup_steps, decay_type, min_lr):
    """Linear warmup, then cosine decay to `min_lr` or a constant lr."""
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(1, warmup_steps)
    )
    if decay_type == "cosine":
        decay_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr
        )
    else:
        decay_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, decay_sched], milestones=[warmup_steps]
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }


class SegmentationSystem(BaseSystem):
    """Segmenter on {modality: [B, C, H, W]} inputs, cross-entropy, mIoU."""

    def __init__(self, config):
        super().__init__(config)
        self.modalities = config["modalities"]
        self.model = Segmenter(
            backbone_name=config["backbone"],
            num_classes=config["num_classes"],
            modalities=self.modalities,
            modality_channels=config["modality_channels"],
            drop_path_rate=config.get("drop_path_rate", 0.0),
        )
        self.ignore_index = 255

        metric_params = {
            "num_classes": config["num_classes"],
            "ignore_index": self.ignore_index,
            "average": "macro",
        }
        self.train_iou = MulticlassJaccardIndex(**metric_params)
        self.val_iou = MulticlassJaccardIndex(**metric_params)
        self.test_iou = MulticlassJaccardIndex(**metric_params)

        self.experiment_name = self.config.get("experiment_name", "").lower()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        if targets.ndim == 4:
            targets = targets.squeeze(1)
        loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        if targets.ndim == 4:
            targets = targets.squeeze(1)
        loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)

        self.val_iou(torch.argmax(logits, dim=1), targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mIoU", self.val_iou, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        if "cityscapes" in self.experiment_name:
            logits = self.sliding_window_inference(inputs)
        else:
            logits = self(inputs)
        if targets.ndim == 4:
            targets = targets.squeeze(1)
        self.test_iou.update(torch.argmax(logits, dim=1), targets)

    def on_test_epoch_end(self):
        miou = self.test_iou.compute()
        self.log("test_mIoU", miou)
        print(f"\nTest mIoU: {miou:.4f}\n")
        self.test_iou.reset()

    def sliding_window_inference(self, images):
        """512x1024 windows with stride 256x512 over full-resolution Cityscapes frames."""
        b, _, h, w = images["rgb"].shape
        window_size = (512, 1024)
        stride = (256, 512)
        final_logits = torch.zeros((b, self.config["num_classes"], h, w), device=self.device)
        count_map = torch.zeros((b, 1, h, w), device=self.device)

        for y in range(0, h - window_size[0] + 1, stride[0]):
            for x in range(0, w - window_size[1] + 1, stride[1]):
                crop = images["rgb"][:, :, y : y + window_size[0], x : x + window_size[1]]
                final_logits[:, :, y : y + window_size[0], x : x + window_size[1]] += self.model({"rgb": crop})
                count_map[:, :, y : y + window_size[0], x : x + window_size[1]] += 1.0

        count_map[count_map == 0] = 1.0
        return final_logits / count_map

    def configure_optimizers(self):
        """AdamW with layer-wise lr decay over the encoder layers (embeddings one step
        below layer 0), linear warmup, then cosine decay or a constant lr."""
        head_lr = float(self.config["lr"])
        backbone_lr = float(self.config.get("backbone_lr", head_lr))
        weight_decay = float(self.config.get("weight_decay", 0.05))
        layer_decay = float(self.config.get("layer_decay", 0.75))

        backbone = self.model.backbone
        layers = backbone.encoder.layer
        num_layers = len(layers)

        param_groups = []
        assigned_ids = set()
        for layer_id, layer in enumerate(layers):
            layer_params = list(layer.parameters())
            assigned_ids.update(id(p) for p in layer_params)
            param_groups.append({
                "params": layer_params,
                "lr": backbone_lr * layer_decay ** (num_layers - layer_id - 1),
                "weight_decay": weight_decay,
            })

        embed_params = list(backbone.embeddings.parameters())
        assigned_ids.update(id(p) for p in embed_params)
        param_groups.append({
            "params": embed_params,
            "lr": backbone_lr * layer_decay**num_layers,
            "weight_decay": weight_decay,
        })

        remaining_backbone = [
            p for p in backbone.parameters() if id(p) not in assigned_ids and p.requires_grad
        ]
        if remaining_backbone:
            param_groups.append(
                {"params": remaining_backbone, "lr": backbone_lr, "weight_decay": weight_decay}
            )

        backbone_ids = {id(p) for p in backbone.parameters()}
        decoder_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        param_groups.append({"params": decoder_params, "lr": head_lr, "weight_decay": weight_decay})

        optimizer = torch.optim.AdamW(
            param_groups, lr=head_lr, betas=tuple(self.config.get("adam_betas", [0.9, 0.999]))
        )

        total_steps = self.trainer.estimated_stepping_batches
        epochs = self.config.get("epochs", 100)
        num_warmup_epochs = float(self.config.get("num_warmup_epochs", 1.0))
        warmup_steps = max(1, int(total_steps * (num_warmup_epochs / epochs)))
        return _warmup_then(
            optimizer, total_steps, warmup_steps,
            self.config.get("decay_type", "cosine"), float(self.config.get("min_lr", 1e-6)),
        )


class DetectionSystem(BaseSystem):
    """Cascade R-CNN on {modality: [B, C, H, W]} inputs, COCO mAP."""

    def __init__(self, config):
        super().__init__(config)
        self.model = MultimodalCascadeRCNN(config)
        self.modalities = config["modalities"]

        self.map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        self.test_map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        self.class_map = config.get("class_map", {})

        self.min_box_side = config.get("box_min_side")
        self.min_box_diagonal = config.get("box_min_diagonal")

    def configure_optimizers(self):
        """AdamW with layer-wise lr decay (Detectron2 / 4M style).

        Depth 0: ViT embeddings + modality adapters + positional / modality embeddings,
        depth 1..N: encoder layers, no decay: everything else (norm, fusion, FPN, heads).
        lr = backbone_lr * layer_decay ** (N - depth)
        """
        cfg = self.config
        head_lr = float(cfg["lr"])
        backbone_lr = float(cfg.get("backbone_lr", head_lr * 0.1))
        weight_decay = float(cfg.get("weight_decay", 0.05))
        layer_decay = float(cfg.get("layer_decay", 0.75))
        num_warmup_epochs = int(cfg.get("num_warmup_epochs", 0))

        wrapper = self.model.backbone
        transformer = wrapper.backbone
        layers = transformer.encoder.layer
        num_layers = len(layers)

        param_groups = []

        embed_params = list(transformer.embeddings.parameters())
        for modality in self.modalities:
            embed_params.extend(getattr(wrapper, f"{modality}_adapter").parameters())
        for modality in self.modalities:
            for prefix in ("pos_embed_", "modality_embed_"):
                embed_params.append(getattr(wrapper, f"{prefix}{modality}"))
        assigned_ids = {id(p) for p in embed_params}
        param_groups.append({
            "params": embed_params,
            "lr": backbone_lr * layer_decay**num_layers,
            "weight_decay": weight_decay,
            "name": "backbone_embeddings",
        })

        for layer_id, layer in enumerate(layers):
            layer_params = list(layer.parameters())
            assigned_ids.update(id(p) for p in layer_params)
            param_groups.append({
                "params": layer_params,
                "lr": backbone_lr * layer_decay ** (num_layers - (layer_id + 1)),
                "weight_decay": weight_decay,
                "name": f"backbone_layer_{layer_id}",
            })

        remaining = [p for p in self.parameters() if id(p) not in assigned_ids and p.requires_grad]
        if remaining:
            param_groups.append(
                {"params": remaining, "lr": head_lr, "weight_decay": weight_decay, "name": "head"}
            )

        optimizer = torch.optim.AdamW(
            param_groups, lr=head_lr, betas=tuple(cfg.get("adam_betas", [0.9, 0.999]))
        )

        total_steps = self.trainer.estimated_stepping_batches
        epochs = cfg["epochs"]
        warmup_steps = int(total_steps * (num_warmup_epochs / epochs)) if epochs > 0 else 0
        return _warmup_then(
            optimizer, total_steps, warmup_steps,
            cfg.get("decay_type", "none"), float(cfg.get("min_lr", 1e-7)),
        )

    def forward(self, inputs, targets=None):
        return self.model(inputs, targets)

    def _prepare_targets(self, inputs, targets):
        device = inputs[self.modalities[0]].device
        return [{"labels": t["labels"].to(device), "boxes": t["boxes"].to(device)} for t in targets]

    def training_step(self, batch, batch_idx):
        inputs, raw_targets = batch
        loss_dict = self(inputs, self._prepare_targets(inputs, raw_targets))
        losses = sum(loss for loss in loss_dict.values())

        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"det_loss/{k}", v)
        return losses

    def validation_step(self, batch, batch_idx):
        inputs, raw_targets = batch

        # torchvision detectors return losses only in train mode, predictions only in eval mode
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(inputs, self._prepare_targets(inputs, raw_targets))
            val_loss = sum(loss_dict.values())
        self.model.eval()
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        outputs = self.model(inputs)
        self._update_map_metric(self.map_metric, inputs, raw_targets, outputs)
        if batch_idx == 0:
            self.log_detections(inputs, raw_targets, outputs, "val_detections")

    def test_step(self, batch, batch_idx):
        inputs, raw_targets = batch
        outputs = self.model(inputs)
        self._update_map_metric(self.test_map_metric, inputs, raw_targets, outputs)

    def predict_step(self, batch, batch_idx):
        inputs, raw_targets = batch
        outputs = self.model(inputs)
        self._update_map_metric(self.test_map_metric, inputs, raw_targets, outputs)

        map_dicts = []
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        for target, output in zip(raw_targets, outputs):
            self._update_map_metric(metric, inputs, [target], [output])
            metrics = metric.compute()
            map_dicts.append({k: metrics[k].item() for k in ("map", "map_50", "map_75")})
            metric.reset()
        return outputs, raw_targets, map_dicts

    def _update_map_metric(self, metric, inputs, raw_targets, outputs):
        device = inputs[self.modalities[0]].device
        formatted_preds = []
        formatted_targets = []
        for i, pred in enumerate(outputs):
            valid = self.filter_boxes(pred["boxes"]) & (pred["labels"] > 0)
            formatted_preds.append({
                "boxes": pred["boxes"][valid],
                "scores": pred["scores"][valid],
                "labels": pred["labels"][valid] - 1,
            })
            formatted_targets.append({
                "boxes": raw_targets[i]["boxes"].to(device),
                "labels": (raw_targets[i]["labels"] - 1).to(device),
            })
        metric.update(formatted_preds, formatted_targets)

    def on_train_epoch_end(self) -> None:
        groups = {g.get("name", ""): g["lr"] for g in self.trainer.optimizers[0].param_groups}
        if "head" in groups:
            self.log("lr/head", groups["head"], on_epoch=True, prog_bar=True)

        layer_names = sorted(
            (k for k in groups if k.startswith("backbone_layer_")), key=lambda n: int(n.split("_")[-1])
        )
        if layer_names:
            self.log("lr/backbone_first", groups[layer_names[0]], on_epoch=True)
            self.log("lr/backbone_last", groups[layer_names[-1]], on_epoch=True, prog_bar=True)
        if "backbone_embeddings" in groups:
            self.log("lr/backbone_embeddings", groups["backbone_embeddings"], on_epoch=True)

    def on_validation_epoch_end(self):
        self._log_map_metrics(self.map_metric, prefix="val")

    def on_test_epoch_end(self):
        self._log_map_metrics(self.test_map_metric, prefix="test")
        metrics = self.test_map_metric.compute()
        print(f"\nTest mAP: {metrics['map']:.4f}, mAP_50: {metrics['map_50']:.4f}\n")

    def on_predict_epoch_end(self):
        metrics = self.test_map_metric.compute()
        print(f"\nTest mAP: {metrics['map']:.4f}, mAP_50: {metrics['map_50']:.4f}, "
              f"mAP_75: {metrics['map_75']:.4f}\n")

    def _log_map_metrics(self, metric, prefix):
        metrics = metric.compute()
        self.log(f"{prefix}_map", metrics["map"], prog_bar=True)
        self.log(f"{prefix}_map_50", metrics["map_50"], prog_bar=True)
        for key in ("map_75", "map_small", "map_medium", "map_large"):
            self.log(f"{prefix}_{key}", metrics[key])

        if "map_per_class" in metrics:
            per_class_map = metrics["map_per_class"]
            if per_class_map.ndim == 0:
                per_class_map = per_class_map.unsqueeze(0)
            for i, score in enumerate(per_class_map):
                class_name = self.class_map.get(i + 1, f"class_{i + 1}")
                self.log(f"{prefix}_map_per_class/{class_name}", score)
        metric.reset()

    def filter_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        if self.min_box_diagonal is None or self.min_box_side is None:
            return torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)

        x_min, y_min, x_max, y_max = boxes.unbind(dim=1)
        widths = x_max - x_min
        heights = y_max - y_min
        diagonals = torch.sqrt(widths**2 + heights**2)
        valid_side = (widths > self.min_box_side) & (heights > self.min_box_side)
        return valid_side & (diagonals > self.min_box_diagonal)

    @staticmethod
    def denormalize_rgb(rgb_image):
        mean = torch.tensor(RGB_MEAN, device=rgb_image.device).view(3, 1, 1)
        std = torch.tensor(RGB_STD, device=rgb_image.device).view(3, 1, 1)
        return (torch.clamp(rgb_image * std + mean, 0, 1) * 255).byte()

    def visualize_pred_vs_gt(self, rgb_image, raw_target, output):
        """Ground truth (green) above predictions with score > 0.5 (red)."""
        img_rgb = self.denormalize_rgb(rgb_image).cpu()
        img_gt = draw_bounding_boxes(img_rgb, raw_target["boxes"].cpu(), colors="green", width=2)

        pred_boxes = output["boxes"][output["scores"] > 0.5]
        if len(pred_boxes) > 0:
            img_pred = draw_bounding_boxes(img_rgb, pred_boxes.cpu(), colors="red", width=2)
        else:
            img_pred = img_rgb
        return torch.cat([img_gt, img_pred], dim=1)

    def log_detections(self, inputs, raw_targets, outputs, tag):
        if "rgb" in inputs:
            images = inputs["rgb"]
        else:
            images = None
            for key in inputs:
                if inputs[key].shape[1] == 1:
                    images = inputs[key]
            if images is None:
                return

        vis_images = []
        for i in range(min(images.shape[0], 4)):
            image = images[i] if "rgb" in inputs else images[i].repeat(3, 1, 1)
            vis_images.append(self.visualize_pred_vs_gt(image, raw_targets[i], outputs[i]))

        grid = make_grid(vis_images, nrow=1)
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(tag, grid, self.global_step)
            elif isinstance(logger, WandbLogger):
                logger.experiment.log(
                    {tag: wandb.Image(grid, caption=f"sample_{self.current_epoch}")},
                    step=self.global_step,
                )
