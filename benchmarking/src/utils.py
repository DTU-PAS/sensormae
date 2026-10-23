import glob
import os


def find_best_checkpoint(logs_folder, exp_name, pattern="*.ckpt"):
    """Most recently written checkpoint in `logs_folder/exp_name/checkpoints` (preferring
    `pattern`, never last.ckpt). Training keeps only the best one (save_top_k=1)."""
    ckpt_dir = os.path.join(logs_folder, exp_name, "checkpoints")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    def candidates(glob_pattern):
        return [
            c for c in glob.glob(os.path.join(ckpt_dir, glob_pattern))
            if "last" not in os.path.basename(c)
        ]

    found = candidates(pattern) or candidates("*.ckpt")
    if not found:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(found, key=os.path.getmtime)
