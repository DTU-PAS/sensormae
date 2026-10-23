"""Fine-tune a backbone on one benchmark.

    python train.py --config configs/<task>/<dataset>/<experiment>.yaml [--seed N]
"""

import argparse
import os
from glob import glob

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from src.experiment import class_map, get_loaders, load_config, setup_torch
from src.system import DetectionSystem, SegmentationSystem


def main():
    setup_torch()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiment config (.yaml)")
    parser.add_argument("--seed", type=int, default=None, help="overrides the config seed (default 42)")
    parser.add_argument("--machine_config", default="machine_config.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="initialize from the latest checkpoint of the experiment")
    args = parser.parse_args()

    config = load_config(args.config, args.machine_config)
    exp_name = config.get("experiment_name", "default")
    task = config.get("task", "segmentation")

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    config["seed"] = seed
    pl.seed_everything(seed, workers=True)

    logs_folder = config["logs_folder"]

    train_loader, val_loader, _ = get_loaders(config)
    if task == "detection":
        config["class_map"] = class_map(train_loader)
        SystemClass, monitor_metric = DetectionSystem, "val_map"
    else:
        SystemClass, monitor_metric = SegmentationSystem, "val_mIoU"

    ckpt_files = glob(f"{logs_folder}/{exp_name}/version_*/checkpoints/*.ckpt") if args.resume else []
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=lambda x: int(x.split("-")[0].split("=")[1]))
        system = SystemClass.load_from_checkpoint(latest_ckpt, config=config)
    else:
        system = SystemClass(config)

    os.makedirs(logs_folder, exist_ok=True)
    loggers = [TensorBoardLogger(save_dir=logs_folder, name=exp_name)]
    if config["wandb"]:
        loggers.append(WandbLogger(
            name=exp_name,
            project=config.get("wandb_project"),
            save_dir=logs_folder,
            config=config,
            entity=config.get("wandb_entity"),
        ))

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=os.path.join(logs_folder, exp_name, "checkpoints"),
        filename="{epoch}-{" + monitor_metric + ":.4f}",
        save_top_k=1,
        mode="max",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu",
        devices=1,
        logger=loggers,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        gradient_clip_val=config["gradient_clip_val"],
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        precision="bf16-mixed",
    )
    trainer.fit(system, train_loader, val_loader)

    best = checkpoint_callback.best_model_score
    print(f"Best {monitor_metric}: {best if best is not None else 0.0:.4f}")


if __name__ == "__main__":
    main()
