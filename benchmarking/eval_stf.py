"""SeeingThroughFog evaluation: KITTI 2D AP (R40, easy / moderate / hard) per weather
condition (clear, light fog, dense fog, rain / snow).

    python eval_stf.py --config configs/object_detection/stf/<experiment>.yaml [--ckpt path.ckpt]
"""

import argparse
import os
from itertools import chain

import lightning.pytorch as pl
import numpy as np
from PIL import Image

from src.data.stf import get_stf_loaders
from src.eval.kitti_2d_detection_eval import get_official_eval_result
from src.experiment import class_map, load_config, save_results, setup_torch
from src.system import DetectionSystem
from src.utils import find_best_checkpoint

EVAL_SETS = {
    "val": ["val_clear_day.txt", "val_clear_night.txt"],
    "clear": ["test_clear_day.txt", "test_clear_night.txt"],
    "light_fog": ["light_fog_day.txt", "light_fog_night.txt"],
    "dense_fog": ["dense_fog_day.txt", "dense_fog_night.txt"],
    "rain/snow": ["rain.txt", "snow_day.txt", "snow_night.txt"],
}
KITTI_CLASSES = ["Car", "Pedestrian", "Cyclist"]
LABEL_TO_KITTI = {1: "Pedestrian", 2: "Car", 3: "Cyclist", -1: "DontCare"}


def to_kitti_format(annos):
    kitti_annos = []
    for annos_i in annos:
        kitti_anno = {"bbox": annos_i["boxes"].numpy().astype(np.float64)}
        if "scores" in annos_i.keys():
            kitti_anno["score"] = annos_i["scores"].numpy().astype(np.float64)
        else:
            kitti_anno["occluded"] = annos_i["occluded"].numpy().astype(np.float64)
            kitti_anno["truncated"] = annos_i["truncated"].numpy().astype(np.float64)
        kitti_anno["name"] = np.array([LABEL_TO_KITTI[label.item()] for label in annos_i["labels"]])
        kitti_anno["alpha"] = np.zeros(kitti_anno["bbox"].shape[0]).astype(np.float64)
        kitti_annos.append(kitti_anno)
    return kitti_annos


def save_visualizations(model, test_loader, predictions, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filenames = [d["file_name"] for d in test_loader.dataset.datapoints]
    count = 0
    for i, (inputs, raw_targets) in enumerate(test_loader):
        bbs, _, map_dicts = predictions[i]
        for image, raw_target, bb, map_dict in zip(inputs["rgb"], raw_targets, bbs, map_dicts):
            vis = model.visualize_pred_vs_gt(image, raw_target, bb)
            vis = vis.cpu().numpy().transpose(1, 2, 0).astype("uint8")
            Image.fromarray(vis).save(
                os.path.join(out_dir, f"pred_{filenames[count]}_map_{map_dict['map']}.png")
            )
            count += 1


def main():
    setup_torch()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiment config (.yaml)")
    parser.add_argument("--ckpt", default=None, help="default: best checkpoint in logs_folder")
    parser.add_argument("--gpu", type=int, default=1, help="number of GPUs")
    parser.add_argument("--machine_config", default="machine_config.yaml")
    parser.add_argument("--do_visualize", action="store_true", help="save prediction images")
    parser.add_argument("--visualizations_path", default="visualizations/predictions")
    args = parser.parse_args()

    config = load_config(args.config, args.machine_config)
    config["batch_size"] = 1
    exp_name = config.get("experiment_name", "default")

    _, _, test_loader = get_stf_loaders(config["data_root"], config)
    config["class_map"] = class_map(test_loader)

    ckpt_path = args.ckpt or find_best_checkpoint(config["logs_folder"], exp_name, pattern="*val_map*.ckpt")
    print(f"Checkpoint: {ckpt_path}")
    model = DetectionSystem.load_from_checkpoint(ckpt_path, config=config, strict=False)
    model.eval()

    trainer = pl.Trainer(accelerator="gpu", devices=args.gpu, logger=False, precision="bf16-mixed")

    result_dict = {}
    for eval_set_name, eval_set in EVAL_SETS.items():
        print(f"Evaluating on: {eval_set}")
        config["test_splits"] = eval_set
        _, _, test_loader = get_stf_loaders(config["data_root"], config)

        predictions = trainer.predict(model, dataloaders=test_loader)
        dt_annos = to_kitti_format(list(chain.from_iterable(batch[0] for batch in predictions)))
        gt_annos = to_kitti_format(list(chain.from_iterable(batch[1] for batch in predictions)))
        kitti_results = get_official_eval_result(gt_annos, dt_annos, KITTI_CLASSES)
        result_dict[eval_set_name] = {metric: f"{value:.2f}" for metric, value in kitti_results.items()}
        print(result_dict[eval_set_name])

        if args.do_visualize:
            save_visualizations(
                model, test_loader, predictions, os.path.join(args.visualizations_path, eval_set_name)
            )

    save_results(config, ckpt_path, result_dict, prefix="kitti_test_results_")


if __name__ == "__main__":
    main()
