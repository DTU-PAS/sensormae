"""Turn one dataset/split into shards, following its split manifest.

Output layout (same as the paper collection):
    OUT/{split}_{n}.h5, OUT/{split}_index.json
plus OUT/_build/{split}_{dataset}_{index,report}.json for resuming and status.
"""

import gzip
import json
import time
from pathlib import Path

from tqdm import tqdm

from .shards import ShardWriter, split_sample

SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"


def load_manifest(name):
    with gzip.open(SPLITS_DIR / f"{name}.json.gz", "rt") as f:
        return json.load(f)


def build_paths(out_dir, split, name):
    d = Path(out_dir) / "_build"
    return d / f"{split}_{name}_index.json", d / f"{split}_{name}_report.json"


def build_split(ds, split, entry, out_dir, segmenter, limit=None):
    """Process the manifest ids of one split. Writes shards, a partial index and a report."""
    ids = entry["ids"][:limit] if limit else entry["ids"]
    keep = set(entry["modalities"]) | set(ds.extra_modalities.get(split, ()))
    index_path, report_path = build_paths(out_dir, split, ds.name)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(index_path.read_text()) if index_path.exists() else {}
    writer = ShardWriter(out_dir, split, replace={v["file"] for v in previous.values()})
    failed, pending = {}, []

    def flush():
        masks = segmenter([s["rgb"] for _, s in pending])
        for (key, sample), mask in zip(pending, masks):
            writer.write(key, {**sample, "segm": mask})
        pending.clear()

    for sample_id in tqdm(ids, desc=f"{ds.name}/{split}", unit="sample"):
        try:
            sample = ds.load(split, sample_id)
        except Exception as e:  # missing / unreadable raw file: report and continue
            failed[sample_id] = f"{type(e).__name__}: {e}"
            continue
        wanted = keep - set(entry["missing"].get(sample_id, ()))
        lacking = wanted - set(sample)
        if lacking:
            failed[sample_id] = f"missing modalities {sorted(lacking)}"
            continue
        sample = {m: a for m, a in sample.items() if m in wanted}
        pending += split_sample(f"{ds.name}_{sample_id}", sample, ds.sliced)
        if len(pending) >= segmenter.batch_size:
            flush()
    if pending:
        flush()

    index = writer.close()
    index_path.write_text(json.dumps(index))
    report = {"expected": len(ids), "written": len(ids) - len(failed), "samples": len(index),
              "failed": failed, "complete": limit is None, "time": time.strftime("%F %T")}
    report_path.write_text(json.dumps(report, indent=1))
    return report


def merge_indexes(out_dir):
    """Merge the per-dataset indexes into OUT/{train,val,test}_index.json."""
    out_dir = Path(out_dir)
    for split in ("train", "val", "test"):
        merged = {}
        for part in sorted((out_dir / "_build").glob(f"{split}_*_index.json")):
            merged.update(json.loads(part.read_text()))
        if merged:
            merged = dict(sorted(merged.items()))
            (out_dir / f"{split}_index.json").write_text(json.dumps(merged, indent=2))
            print(f"{split}_index.json: {len(merged)} samples")
