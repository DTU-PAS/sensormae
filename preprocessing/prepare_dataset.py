"""Build the multi-sensor pretraining collection from the original public datasets.

    python prepare_dataset.py status            # what is there, what is downloadable, what is built
    python prepare_dataset.py download [--datasets ...]
    python prepare_dataset.py build [--datasets ...] [--test]

By default everything lives in the `data/` folder of the repository:
    data/raw/<dataset>/                  original datasets (see README for the layouts)
    data/models/                         FoundationStereo ONNX files
    data/work/                           scratch space (extracted ROS bags)
    data/sensormae_sharded_mix_segm/     the collection ({split}_{n}.h5 + {split}_index.json)
    data/sensormae_test/                 output of --test
"""

import argparse
import json
import shutil
from pathlib import Path

from sensormae_prep.build import build_paths, build_split, load_manifest, merge_indexes
from sensormae_prep.datasets import DATASETS

SPLITS = ("train", "val", "test")
DATA = Path(__file__).resolve().parents[1] / "data"


def selected(args):
    names = args.datasets or list(DATASETS)
    unknown = set(names) - set(DATASETS)
    if unknown:
        raise SystemExit(f"unknown datasets: {sorted(unknown)} (choose from {list(DATASETS)})")
    return [DATASETS[n](Path(args.data_root) / n, Path(args.work_dir) / n) for n in names]


def status(args):
    print(f"{'dataset':20s} {'raw data':10s} {'download':9s} " + " ".join(f"{s:>17s}" for s in SPLITS))
    for ds in selected(args):
        missing = ds.missing()
        manifest = load_manifest(ds.name)
        cells = []
        for split in SPLITS:
            if split not in manifest:
                cells.append(f"{'-':>17s}")
                continue
            expected = len(manifest[split]["ids"])
            report = build_paths(args.out, split, ds.name)[1]
            if report.exists():
                r = json.loads(report.read_text())
                cells.append(f"{r['written']:>7d}/{expected:<7d}{'' if r['complete'] else '*':2s}")
            else:
                cells.append(f"{0:>7d}/{expected:<7d}  ")
        raw = "ok" if not missing else "MISSING"
        print(f"{ds.name:20s} {raw:10s} {'auto' if ds.downloadable else 'manual':9s} " + " ".join(cells))
        if missing and args.verbose:
            print(f"    missing: {', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''}")
            print(f"    get it from: {ds.manual}")
    print(f"\nwritten/expected samples per split in {args.out} (* = partial run, --test or --limit).\n"
          f"Ids that failed are listed in {args.out}/_build/<split>_<dataset>_report.json.")


def download(args):
    for ds in selected(args):
        if ds.downloadable:
            ds.fetch()
            print(f"[{ds.name}] done")
        else:
            print(f"[{ds.name}] manual download: {ds.manual}")


def build(args):
    from sensormae_prep.depth import DepthModels
    from sensormae_prep.shards import Segmenter

    out = Path(args.out)
    if args.test and out.exists():
        shutil.rmtree(out)  # test runs always start from an empty folder
    out.mkdir(parents=True, exist_ok=True)
    depth = DepthModels(args.models, args.device)
    segmenter = Segmenter(device=args.device, batch_size=args.batch_size)
    for ds in selected(args):
        missing = ds.missing()
        if missing:
            print(f"[{ds.name}] skipped, raw data missing: {missing[:4]}\n    get it from: {ds.manual}")
            continue
        manifest = load_manifest(ds.name)
        if ds.depth_model:
            ds.stereo = depth[ds.depth_model]
        todo = []
        for split in args.splits or SPLITS:
            if split not in manifest:
                continue
            report = build_paths(out, split, ds.name)[1]
            if report.exists() and json.loads(report.read_text())["complete"] and not args.force:
                print(f"[{ds.name}/{split}] already built, skipping (use --force to redo)")
                continue
            todo.append(split)
        ds.prepare([i for s in todo for i in manifest[s]["ids"][:args.limit]])
        for split in todo:
            r = build_split(ds, split, manifest[split], out, segmenter, args.limit)
            print(f"[{ds.name}/{split}] {r['written']}/{r['expected']} samples, "
                  f"{r['samples']} after slicing, {len(r['failed'])} failed")
    merge_indexes(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("status", "download", "build", "index"))
    p.add_argument("--data-root", default=DATA / "raw", help="folder with one sub-folder per dataset")
    p.add_argument("--out", help="output folder (default data/sensormae_sharded_mix_segm, "
                                 "data/sensormae_test with --test)")
    p.add_argument("--models", default=DATA / "models", help="folder with the FoundationStereo ONNX files")
    p.add_argument("--work-dir", default=DATA / "work", help="scratch space for extracted ROS bags")
    p.add_argument("--datasets", nargs="+", help=f"subset of {list(DATASETS)}")
    p.add_argument("--splits", nargs="+", choices=SPLITS)
    p.add_argument("--test", nargs="?", type=int, const=10, metavar="N",
                   help="test run: the whole pipeline on the first N (default 10) samples per split")
    p.add_argument("--limit", type=int, help="only the first N samples per split")
    p.add_argument("--batch-size", type=int, default=32, help="Mask2Former batch size")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--force", action="store_true", help="rebuild splits that are already complete")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    if args.test:
        args.limit = args.limit or args.test
        args.force = True
    args.out = Path(args.out or DATA / ("sensormae_test" if args.test else "sensormae_sharded_mix_segm"))
    {"status": status, "download": download, "build": build,
     "index": lambda a: merge_indexes(a.out)}[args.command](args)


if __name__ == "__main__":
    main()
