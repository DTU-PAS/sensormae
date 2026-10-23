"""Config loading and dataset selection shared by the entry points."""

import argparse
import json
import os

import torch
import yaml

from src.data.bdd100k import get_bdd100k_loaders
from src.data.cityscapes import get_cityscapes_loaders
from src.data.dsec_det import get_dsec_det_loaders
from src.data.dsec_seg import get_dsec_seg_loaders
from src.data.fmb import get_fmb_loaders
from src.data.kitti import get_kitti_loaders
from src.data.llvip import get_llvip_loaders
from src.data.mfnet import get_mfnet_loaders
from src.data.stf import get_stf_loaders
from src.system import DetectionSystem, SegmentationSystem
from src.utils import find_best_checkpoint

MACHINE_KEYS = ("logs_folder", "data_root", "wandb", "wandb_project", "wandb_entity")

# Dataset keyword (matched against experiment_name, then data_root) -> loader factory
LOADERS = {
    "detection": [
        ("dsec", get_dsec_det_loaders),
        ("llvip", get_llvip_loaders),
        ("kitti", get_kitti_loaders),
        ("stf", get_stf_loaders),
    ],
    "segmentation": [
        ("dsec", get_dsec_seg_loaders),
        ("fmb", get_fmb_loaders),
        ("mfnet", get_mfnet_loaders),
        ("cityscapes", get_cityscapes_loaders),
        ("bdd100k", get_bdd100k_loaders),
    ],
}


def load_config(config_path, machine_config_path="machine_config.yaml"):
    """Experiment config, with logs/data/W&B settings overridden by the machine config."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if os.path.exists(machine_config_path):
        with open(machine_config_path) as f:
            machine = yaml.safe_load(f) or {}
        config.update({k: machine[k] for k in MACHINE_KEYS if k in machine})

    config.setdefault("logs_folder", "logs")
    config.setdefault("wandb", False)
    return config


def setup_torch():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")


def get_loaders(config):
    """(train, val, test) loaders of the dataset named in experiment_name / data_root.
    Cityscapes and BDD100K have no labelled test split: their val loader is used."""
    name = config["experiment_name"].lower()
    root = config.get("data_root", "").lower()
    for keyword, factory in LOADERS[config["task"]]:
        if keyword in name or keyword in root:
            loaders = factory(config["data_root"], config)
            return loaders if len(loaders) == 3 else (*loaders, loaders[1])
    raise ValueError(f"No dataset found for experiment {config['experiment_name']}")


def class_map(loader):
    """{class id: class name} of a detection dataset."""
    cls_map = loader.dataset.class_map
    if isinstance(next(iter(cls_map.keys())), str):
        cls_map = {v: k for k, v in cls_map.items()}
    return cls_map


def test_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiment config (.yaml)")
    parser.add_argument("--ckpt", default=None, help="default: best checkpoint in logs_folder")
    parser.add_argument("--gpu", type=int, default=1, help="number of GPUs")
    parser.add_argument("--machine_config", default="machine_config.yaml")
    return parser


def load_for_testing(args):
    """Config (batch size 1), fine-tuned model, test loader and checkpoint path."""
    config = load_config(args.config, args.machine_config)
    config["batch_size"] = 1

    _, _, test_loader = get_loaders(config)
    if config.get("task", "segmentation") == "detection":
        config["class_map"] = class_map(test_loader)
        system_class = DetectionSystem
    else:
        system_class = SegmentationSystem

    ckpt_path = args.ckpt or find_best_checkpoint(
        config["logs_folder"], config.get("experiment_name", "default")
    )
    print(f"Checkpoint: {ckpt_path}")
    model = system_class.load_from_checkpoint(ckpt_path, config=config, strict=False)
    model.eval()
    return config, model, test_loader, ckpt_path


def save_results(config, ckpt_path, results, prefix):
    """Write results to <logs_folder>/<experiment>/test_results/<prefix><checkpoint>.json."""
    out_dir = os.path.join(config["logs_folder"], config.get("experiment_name", "default"), "test_results")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_name = os.path.basename(ckpt_path).split(".ckpt")[0]
    out_path = os.path.join(out_dir, f"{prefix}{ckpt_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {out_path}")


def round_results(results):
    return [{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()} for res in results]
