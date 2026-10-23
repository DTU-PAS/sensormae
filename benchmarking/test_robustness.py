"""Missing-modality robustness: test a checkpoint with every subset of its input
modalities, the missing ones replaced by zeros (black images).

    python test_robustness.py --config configs/<task>/<dataset>/<experiment>.yaml [--ckpt path.ckpt]
"""

from itertools import combinations

import lightning.pytorch as pl
import torch

from src.experiment import load_for_testing, round_results, save_results, setup_torch, test_arguments


class ModalityMaskedLoader:
    """Zeroes the inputs of the modalities not in `active_modalities`."""

    def __init__(self, loader, active_modalities):
        self.loader = loader
        self.active_modalities = set(active_modalities)

    def __iter__(self):
        for inputs, targets in self.loader:
            yield {
                modality: tensor if modality in self.active_modalities else torch.zeros_like(tensor)
                for modality, tensor in inputs.items()
            }, targets

    def __len__(self):
        return len(self.loader)


def modality_subsets(modalities):
    """All non-empty subsets, largest first."""
    return [s for r in range(len(modalities), 0, -1) for s in combinations(modalities, r)]


def main():
    setup_torch()
    args = test_arguments().parse_args()
    config, model, test_loader, ckpt_path = load_for_testing(args)

    trainer = pl.Trainer(accelerator="gpu", devices=args.gpu, logger=False, precision="bf16-mixed")

    modalities = config["modalities"]
    all_results = {}
    for subset in modality_subsets(modalities):
        name = "all" if len(subset) == len(modalities) else "+".join(subset)
        print(f"\nModalities: {name}")
        results = trainer.test(model, dataloaders=ModalityMaskedLoader(test_loader, subset))
        all_results[name] = {
            "active_modalities": list(subset),
            "missing_modalities": [m for m in modalities if m not in subset],
            "metrics": round_results(results),
        }

    save_results(config, ckpt_path, all_results, prefix="test_results_robustness_")


if __name__ == "__main__":
    main()
