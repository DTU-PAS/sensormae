"""Evaluate a fine-tuned checkpoint on the test split (mIoU / COCO mAP).

    python test.py --config configs/<task>/<dataset>/<experiment>.yaml [--ckpt path.ckpt]

Without --ckpt the best checkpoint of the experiment in logs_folder is used.
Results are written to <logs_folder>/<experiment>/test_results/.
"""

import lightning.pytorch as pl

from src.experiment import load_for_testing, round_results, save_results, setup_torch, test_arguments


def main():
    setup_torch()
    args = test_arguments().parse_args()
    config, model, test_loader, ckpt_path = load_for_testing(args)

    trainer = pl.Trainer(accelerator="gpu", devices=args.gpu, logger=False, precision="bf16-mixed")
    results = round_results(trainer.test(model, dataloaders=test_loader))
    save_results(config, ckpt_path, results, prefix="test_results_")


if __name__ == "__main__":
    main()
