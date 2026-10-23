"""Build the split manifests in `splits/` from a built collection.

Used once to create the shipped split files; end users do not need it. For every dataset and split it
stores the base sample ids (before the 640px slicing) and, for the few samples that
lack a modality the dataset normally has, which modalities are missing.

    python tools/build_splits.py /path/to/sensormae_sharded_mix_segm [OUT_DIR]
"""

import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the LZ4 filter)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensormae_prep.datasets import DATASETS  # noqa: E402

SLICE_SUFFIX = re.compile(r"_(\d+)$")


def main(shard_dir, out_dir=None):
    shard_dir = Path(shard_dir)
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parents[1] / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    # manifest[dataset][split][base_id] = set of modalities (segm excluded)
    manifest = defaultdict(lambda: defaultdict(dict))

    for shard in tqdm(sorted(shard_dir.glob("*.h5"))):
        split = shard.name.split("_")[0]
        with h5py.File(shard, "r") as f:
            for key in f:
                name, sample_id = key.split("_", 1)
                if DATASETS[name].sliced:
                    sample_id = SLICE_SUFFIX.sub("", sample_id)
                mods = set(json.loads(f[key].attrs["shapes"])) - {"segm"}
                manifest[name][split][sample_id] = mods

    for name, splits in manifest.items():
        out = {}
        for split, samples in splits.items():
            full = Counter(m for mods in samples.values() for m in mods)
            full = sorted(full)
            missing = {
                sid: sorted(set(full) - mods)
                for sid, mods in samples.items()
                if set(full) - mods
            }
            out[split] = {
                "modalities": full,
                "ids": sorted(samples),
                "missing": missing,
            }
            print(f"{name:20s} {split:5s} {len(samples):7d} ids, "
                  f"{len(missing)} with missing modalities")
        with gzip.open(out_dir / f"{name}.json.gz", "wt") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main(*sys.argv[1:3])
