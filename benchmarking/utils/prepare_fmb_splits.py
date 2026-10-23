"""Carve the FMB validation split (5% of train, seed 42) out of the train folder, in place.

    python utils/prepare_fmb_splits.py --data-root data/fmb
"""

import argparse
import random
import shutil
from pathlib import Path

VAL_PERCENT = 0.05
SEED = 42


def prepare_splits(root):
    root = Path(root)
    source_train = root / "train"
    new_val = root / "val"

    if new_val.exists() and any(new_val.iterdir()):
        print(f"Validation folder {new_val} already exists. Skipping split.")
        return

    print(f"Creating validation split from {source_train}...")

    subfolders = ["Visible", "Infrared", "Label"]

    rgb_dir = source_train / "Visible"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"Could not find {rgb_dir}")

    all_files = sorted([f.name for f in rgb_dir.iterdir() if f.is_file()])
    total_files = len(all_files)

    random.seed(SEED)
    random.shuffle(all_files)

    val_count = int(total_files * VAL_PERCENT)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    print(f"Total: {total_files} | Train: {len(train_files)} | Val: {len(val_files)}")

    for folder_name in subfolders:
        src_subdir = source_train / folder_name
        dst_subdir = new_val / folder_name

        dst_subdir.mkdir(parents=True, exist_ok=True)

        print(f"Moving {len(val_files)} files from {src_subdir} to {dst_subdir}...")

        for fname in val_files:
            src_path = src_subdir / fname
            dst_path = dst_subdir / fname

            if src_path.exists():
                shutil.move(str(src_path), str(dst_path))
            else:
                print(f"Warning: {src_path} not found (skipping)")

    print("Done! Dataset structure is now:")
    print(f"  {root}/train (Reduced)")
    print(f"  {root}/val   (New 5%)")
    print(f"  {root}/test  (Original)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Carve a validation split out of FMB's train set (moves files in place)."
    )
    parser.add_argument(
        "--data-root", type=str, required=True, help="Path to the FMB dataset root (contains train/test)."
    )
    args = parser.parse_args()
    prepare_splits(args.data_root)
