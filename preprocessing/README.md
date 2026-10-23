# Multi-sensor pretraining collection

Code to rebuild the pretraining collection from the
16 original public datasets: ~826k aligned samples / 2.82M images with RGB, LWIR, NIR
(SWIR), Event, Stereo Depth and Segmentation. The collection itself cannot be
redistributed, so every user rebuilds it locally.

The exact sample selection and train/val/test split of the paper is shipped in
[`splits/`](splits/), one gzipped JSON file per dataset:

```text
{"train": {"modalities": ["depth", "event", "rgb"],   // modalities of the dataset
           "ids": ["00000001", "00000002", ...],       // sample ids, before 640px slicing
           "missing": {"00000123": ["event"], ...}},   // samples lacking a modality
 "val": {...}, "test": {...}}
```

Nothing else is stored there: no image data, only names. The pipeline processes exactly
these ids, which fixes the temporal subsampling, the validation draws and the frames
left out of every dataset.

## Pipeline

For every sample of a manifest:

1. **Dataset-specific processing** (`sensormae_prep/datasets/`): read the raw files,
   undistort / rectify, synchronize sensors (nearest frame within 50 ms), build event
   frames, crop all sensors to their common field of view and resize them to one grid.
2. **Depth** (`sensormae_prep/depth.py`): FoundationStereo (ViT-S) on the stereo RGB pair,
   at roughly half resolution, resized back and min-max normalized per frame (float16).
3. **Thermal normalization**: per-image min-max to uint16 + CLAHE (clip 2, 8x8 tiles), for
   LWIR, NIR and SWIR.
4. **Slicing**: images wider than 800 px are cut into overlapping 640 px wide slices
   (`<id>_0`, `<id>_1`, ...).
5. **Segmentation**: Mask2Former Swin-L (Cityscapes semantic) on every RGB slice, 19
   Cityscapes train ids (uint8).
6. **Packing** into LZ4-compressed HDF5 shards plus `{train,val,test}_index.json`.

Event frames always hold the events of the 50 ms before the image timestamp, binned into
5 slices of 10 ms, one set per polarity: `(H, W, 10)` uint8 counts.

## Installation

```bash
python -m venv .env && source .env/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended (FoundationStereo and Mask2Former); `--device cpu`
works for tests. The Mask2Former weights are fetched from the Hugging Face hub on first use.

### FoundationStereo ONNX models

Depth uses the ViT-small FoundationStereo checkpoint (`11-33-40`) exported to ONNX at a
fixed input size per dataset:

```bash
git clone https://github.com/NVlabs/FoundationStereo && cd FoundationStereo
# download the 11-33-40 checkpoint folder (README of FoundationStereo) into pretrained_models/
for spec in s_w1024:512:1024 s_640:480:640 s_ms2:192:608 s_sthereo:288:640 nsavp:512:608; do
  IFS=: read name h w <<< "$spec"
  XFORMERS_DISABLED=1 python scripts/make_onnx.py --ckpt_dir pretrained_models/11-33-40/model_best_bp2.pth \
      --save_path ../data/models/foundation_stereo_$name.onnx --height $h --width $w --valid_iters 16
done
```

| ONNX file | input (h x w) | used for |
|---|---|---|
| `foundation_stereo_s_w1024.onnx` | 512 x 1024 | Cityscapes |
| `foundation_stereo_s_640.onnx` | 480 x 640 | DSEC, ECMD |
| `foundation_stereo_s_ms2.onnx` | 192 x 608 | MS2 |
| `foundation_stereo_s_sthereo.onnx` | 288 x 640 | STheReO |
| `foundation_stereo_nsavp.onnx` | 512 x 608 | NSAVP |

The collection was generated with `valid_iters=16` (the default of `make_onnx.py`; the
FoundationStereo README and Appendix C mention 20).

## Usage

Everything lives in the `data/` folder at the root of the repository (its content is not
versioned; symlinks are fine):

```
data/raw/<dataset>/                 original datasets, layouts in the table below
data/models/                        FoundationStereo ONNX files
data/work/                          scratch space (frames extracted from ROS bags)
data/sensormae_sharded_mix_segm/    the collection
data/sensormae_test/                output of test runs
```

Run from `preprocessing/`:

```bash
# what is present, what can be downloaded automatically, what is already built
python prepare_dataset.py status -v

# fetch the datasets that can be downloaded by script (DSEC, FR-T, KAIST, MS2, plus
# Cityscapes with a cityscapes-dataset.com account and RASMD with a Hugging Face login)
python prepare_dataset.py download --datasets dsec frt kaist-ms-pedestrian ms2 cityscapes rasmd

# test run: the whole pipeline on the first 10 samples of every split of every dataset
# present, written to data/sensormae_test/
python prepare_dataset.py build --test

# full build (resumes per dataset/split)
python prepare_dataset.py build
```

Useful options: `--datasets` / `--splits` to restrict, `--test N` for N samples per split,
`--device cpu`, `--batch-size` (Mask2Former), `--force` to rebuild finished splits, and
`--data-root`, `--models`, `--work-dir`, `--out` to use other folders.

Output (`data/sensormae_sharded_mix_segm/`): `{train,val,test}_{n}.h5` shards (~1.8 GB)
and `{train,val,test}_index.json`. Each dataset/split is
written to its own shards, so it can be rebuilt alone. `_build/` keeps, per dataset and
split, its index and a report with the ids that could not be built (missing or
unreadable raw files) and why; `status` reads these reports. `python prepare_dataset.py
index` only regenerates the merged index files.

## Datasets

| Dataset | Modalities | Download | Expected layout under `DATA/<name>/` |
|---|---|---|---|
| `bdd` BDD100K | RGB | manual, [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu) (100K images) | `images/100k/{train,val,test}/*.jpg` |
| `cityscapes` | RGB, Depth | auto with an account (`csDownload`), [cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/): left+right 8bit trainvaltest and trainextra (~110 GB) | `{left,right}Img8bit/{train,train_extra,val,test}/<city>/` |
| `dsec` | RGB, Event, Depth | auto | `{train,test}/<scene>/{images,events,calibration}` (official download script layout) |
| `ecmd` | RGB, LWIR, Event, Depth | manual, [ECMD](https://arclab-hku.github.io/ecmd/) rosbags | `**/*.bag` |
| `fmb` | RGB, LWIR | manual, [FMB / SegMiF](https://github.com/JinyuanLiu-CV/SegMiF) | `{train,test}/{Visible,Infrared}/*.png` |
| `frt` Freiburg Thermal | RGB, LWIR | auto | `train/seq_*/NN/{fl_rgb,fl_ir_aligned}`, `test/{day,night}/...` |
| `kaist-ms-pedestrian` | RGB, LWIR | auto (Hugging Face) | `setNN/VNNN/{visible,lwir}/*.jpg` |
| `llvip` | RGB, LWIR | manual, [LLVIP](https://bupt-ai-cz.github.io/LLVIP/) | `{visible,infrared}/{train,test}/*.jpg` |
| `mfnet` | RGB, LWIR | manual, [MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) `ir_seg_dataset` | `images/*.png`, `{train,val,test}.txt` |
| `ms2` | RGB, NIR, LWIR, Depth | auto (Dropbox links of the official script) | `sync_data/<sequence>/{rgb,nir,thr}/...` |
| `nsavp` | RGB, LWIR, Event, Depth | manual, [NSAVP](https://umautobots.github.io/nsavp) (Globus; format in [nsavp_tools](https://github.com/umautobots/nsavp_tools)) | `<SEQ>/<SEQ>_{rgb_left,rgb_right,adk_left,dvxplorer_left}.h5`, `<SEQ>_C1_calibration_results.yaml` |
| `ntu4dradlm` | RGB, LWIR | manual, [NTU4DRadLM](https://github.com/junzhang2016/NTU4DRadLM) rosbags | `{garden,loop1,loop2,loop3,smoke_cp}/*.bag` |
| `rasmd` | RGB, SWIR (as NIR) | auto after accepting the terms on [Hugging Face](https://huggingface.co/datasets/STL-Yonsei/RASMD) and `huggingface-cli login` (`RASMD_aligned.zip`, 46 GB) | `**/croped_data/{rgb,swir}/*.jpg` |
| `smod` | RGB, LWIR | manual, [SMOD](https://github.com/zizhaochen6/SMOD-Dataset) | `{day,night}/*_{rgb,tir}.jpg` |
| `sthereo` | RGB, LWIR, Depth | manual (request access), [STheReO](https://sites.google.com/view/rpmsthereo/) | `<sequence>/image/{stereo_left,stereo_right,stereo_thermal_14_left}` |
| `vividpp` Vivid++ | RGB, LWIR, Event | manual, [Vivid++](https://visibilitydataset.github.io/) driving rosbags | `{driving_full,driving_vision}/<sequence>.bag` (or `<sequence>/*.bag`) |

Processing notes:

- Cityscapes `train` includes `train_extra`.
- DSEC: frames with fewer than 200 events in the window have no event frame.
- FR-T: in the test split the IR file names carry a 9-digit nanosecond field and the RGB
  ones a 10-digit field; pairs are matched on the truncated field.
- NSAVP: depth is predicted on the undistorted (not rectified) stereo pair, given to the
  network in BGR channel order. The camera order of the calibration file is checked at
  run time.
- RASMD: sample ids are the running number of the pairs after a seeded shuffle (seed 42),
  walking the `croped_data` folders in sorted order.
- Vivid++: driving-setup calibration of the release (`calib/vividpp.yaml`), 23 driving
  sequences (table in `sensormae_prep/datasets/vividpp.py`); no stereo pair, so no depth.

## Output format

Every sample is one uint8 HDF5 dataset holding a msgpack dict `{modality: raw bytes}`,
with `shapes` and `dtypes` (JSON) as attributes:

| key | shape | dtype |
|---|---|---|
| `rgb` | (H, W, 3) | uint8 |
| `lwir`, `nir`, `swir` | (H, W, 1) | uint16 (min-max + CLAHE) |
| `depth` | (H, W) | float16, normalized disparity in [0, 1] |
| `event` | (H, W, 10) | uint8 |
| `segm` | (H, W) | uint8, Cityscapes train ids |

```python
import h5py, hdf5plugin, json
from sensormae_prep.shards import read_sample

out = "../data/sensormae_sharded_mix_segm"
index = json.load(open(f"{out}/train_index.json"))
entry = index["ms2_01_000213_0"]
with h5py.File(f"{out}/{entry['file']}") as f:
    sample = read_sample(f, entry["path"])   # {"rgb": ..., "depth": ..., ...}
```

## Tools

- `tools/build_splits.py COLLECTION [OUT_DIR]` regenerates `splits/` from a built
  collection (used once to create the shipped split files).
