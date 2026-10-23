# Downstream benchmarking

Fine-tuning and evaluation code for the downstream experiments of the paper:
semantic segmentation (mIoU) and 2D object detection (COCO mAP; KITTI-style AP for
SeeingThroughFog) on eight datasets, from six ViT initializations.

| Dataset | Task | Modalities |
| :--- | :--- | :--- |
| Cityscapes | Segmentation | RGB |
| BDD100K | Segmentation | RGB |
| DSEC | Segmentation, Detection | RGB, Depth, Event |
| FMB | Segmentation | RGB, LWIR |
| MFNet | Segmentation | RGB, LWIR |
| KITTI | Detection | RGB, Depth |
| LLVIP | Detection | RGB, LWIR |
| SeeingThroughFog (STF) | Detection | RGB, SNIR (gated), Depth, LiDAR, Radar |

## Model

Every modality has its own `Conv2d` patch embedding, positional embedding and modality
embedding. The tokens of all modalities are concatenated and encoded by one shared ViT
(Hugging Face `ViTModel`); the encoded tokens are split back per modality and fused with a
linear projection.

- **Segmentation** ([src/models/segmenter.py](src/models/segmenter.py)): Segmenter mask
  transformer decoder (2 layers) over learnable class embeddings.
- **Detection** ([src/models/detector.py](src/models/detector.py)): Cascade R-CNN on a
  simple feature pyramid (as in ViTDet) built from the fused tokens. Anchors, RPN and RoI
  pooling come from torchvision's Faster R-CNN; its RoI heads are replaced by the cascade
  heads of [src/models/cascade_roi_heads.py](src/models/cascade_roi_heads.py)
  (`head_type: cascade_rcnn`, used by all detection experiments).

## Setup

```bash
python -m venv .env && source .env/bin/activate
pip install -r requirements.txt
cp machine_config.example.yaml machine_config.yaml
```

`requirements.txt` pins the versions the paper results were produced with (Python 3.13,
CUDA GPU). `machine_config.yaml` (not versioned) sets `logs_folder`, `data_root` and the
Weights & Biases options, and overrides them in every experiment config.

### Backbones

| Config suffix | Initialization | `backbone` |
| :--- | :--- | :--- |
| `_vit` | ViT-B/16, supervised ImageNet | `google/vit-base-patch16-224` |
| `_mae` | MAE ViT-B/16 | `facebook/vit-mae-base` |
| `_dino` | DINO ViT-B/16 | `facebook/dino-vitb16` |
| `_multimae` | MultiMAE ViT-B/16 | `weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth` |
| `_sensormae_base_vit` | SensorMAE ViT-B, pretraining initialized from ViT | `weights/sensormae-base-vit.pth` |
| `_sensormae_base_mm` | SensorMAE ViT-B, pretraining initialized from MultiMAE | `weights/sensormae-base-mm.pth` |
| `_sensormae_small_vit` | SensorMAE ViT-S, pretraining initialized from ViT | `weights/sensormae-small-vit.pth` |
| `_sensormae_tiny_vit` | SensorMAE ViT-Ti, pretraining initialized from ViT | `weights/sensormae-tiny-vit.pth` |

Hugging Face models are downloaded on first use. Local files go in `weights/` (not
versioned): the SensorMAE weights from the Hugging Face Hub and the official
[MultiMAE](https://github.com/EPFL-VILAB/MultiMAE) checkpoint, see
[Pretrained weights](../README.md#pretrained-weights) for the download commands. Local
weights are read with `torch.load(weights_only=True)`, so only tensors are deserialized.

How an initialization is mapped onto the model
([src/weight_adapters.py](src/weight_adapters.py)):

- **Hugging Face ViTs**: the RGB patch embedding is copied to the RGB adapter; every other
  modality gets the channel mean of the RGB kernel, repeated over its input channels.
- **MultiMAE**: the RGB and depth adapters come from the corresponding MultiMAE input
  adapters, the other modalities get the channel mean of the RGB one.
- **SensorMAE**: adapters, positional and modality embeddings are taken from the
  pretrained modality they correspond to (LWIR for LWIR, SNIR for NIR / SNIR / SWIR,
  depth for depth / LiDAR / radar).

### Data

`data_root` is a folder with one sub-folder per dataset (a symlink works:
`ln -s /path/to/datasets data`). The datasets are not redistributed; the loaders in
[src/data/](src/data/) read the following layouts.

```text
data/
├── cityscapes/          leftImg8bit/{train,val}/<city>/  gtFine/{train,val}/<city>/
├── bdd100k/             images/10k/{train,val}/          labels/sem_seg/{train,val}/
├── mfnet/               images/  labels/  {train,val,test,test_day,test_night}.txt  black_list.txt
├── fmb/                 {train,val,test}/{Visible,Infrared,Label}/
├── dsec/                {train,test}/<sequence>.h5       (rgb, event, depth, boxes, segmentation)
├── kitti/               {train,val}/{rgb,depth,labels}/
├── llvip/               {train,test}/{rgb,lwir}/  {train,test}/data_coco.json
└── stf/                 cam_stereo_left_lut/  gated_full_acc_rect/  depth_maps/
                         lidar_samples/yzi/lidar_hdl64_strongest/  radar_samples/yzv/
                         gt_labels/  splits/
```

FMB has no official validation split. The one used in the paper is created from the
training set with:

```bash
python utils/prepare_fmb_splits.py --data-root ./data/fmb
```

Cityscapes, BDD100K and KITTI are tested on their validation split.

Files derived from the official releases are created with the scripts in
[data_preparation/](data_preparation/):

```bash
# LLVIP: COCO annotations from the VOC files
python data_preparation/llvip.py --root ./data/llvip

# STF: LiDAR / radar projected onto the RGB crop of the gated camera
git clone https://github.com/princeton-computational-imaging/SeeingThroughFog
pip install pyquaternion
python data_preparation/stf.py --root ./data/stf --stf_repo SeeingThroughFog
```

The STF splits are the official ones of the SeeingThroughFog repository.

DSEC is converted to one h5 file per sequence from the official DSEC images, events and
calibration, the [DSEC-Det](https://github.com/uzh-rpg/dsec-det) labels and the
[DSEC-Semantic](https://dsec.ifi.uzh.ch/dsec-semantic/) labels; depth is computed with the
FoundationStereo ONNX model of the [preprocessing](../preprocessing/README.md) (needs
`onnxruntime-gpu`). See the header of [data_preparation/dsec.py](data_preparation/dsec.py)
for the expected folders and the content of each sample.

```bash
python data_preparation/dsec.py --dsec /path/to/dsec --semantic /path/to/dsec_semantic \
    --stereo_model foundation_stereo_s_640.onnx --out ./data/dsec
```

## Experiments

One YAML file per experiment in [configs/](configs/), named
`<number of modalities>_<dataset>_<modalities>_<backbone>.yaml`:

```text
configs/segmentation/{cityscapes,bdd100k,dsec,fmb,mfnet}/
configs/object_detection/{dsec,kitti,llvip,stf}/
```

The main fields are `task` (`segmentation` / `detection`), `backbone`, `modalities`,
`modality_channels`, `image_size`, the optimization settings (`lr`, `backbone_lr`,
`layer_decay`, `weight_decay`, `epochs`, `num_warmup_epochs`, `decay_type`,
`batch_size`, `accumulate_grad_batches`) and the augmentation settings. The dataset is
selected from `experiment_name`. All runs use seed 42 unless `seed` is set in the config
or passed with `--seed`.

### Training

```bash
python train.py --config configs/segmentation/mfnet/2_mfnet_rgb_lwir_sensormae_base_mm.yaml
```

Checkpoints, TensorBoard logs and results are written to
`<logs_folder>/<experiment_name>/`. The best checkpoint on the validation metric is kept.

### Testing

```bash
python test.py --config <config>.yaml [--ckpt <checkpoint>.ckpt]
```

Without `--ckpt` the best checkpoint of the experiment is used. The test metrics are saved
to `<logs_folder>/<experiment_name>/test_results/test_results_<checkpoint>.json`.

### SeeingThroughFog

STF detection is reported as KITTI 2D AP (R40; easy, moderate, hard; Car, Pedestrian,
Cyclist) per weather condition: clear, light fog, dense fog, rain / snow.

```bash
python eval_stf.py --config configs/object_detection/stf/<experiment>.yaml [--ckpt <checkpoint>.ckpt]
python utils/stf_results_to_excel.py --logs-dir logs
```

The additional STF seeds (43, 44) are trained and evaluated with `bash run_stf_seeds.sh`.

### Missing modalities

```bash
python test_robustness.py --config <config>.yaml [--ckpt <checkpoint>.ckpt]
python utils/report_robustness.py --logs_folder logs
```

`test_robustness.py` tests a multimodal model with every non-empty subset of its input
modalities, the missing ones set to zero. `report_robustness.py` collects the results of all
backbones into one table per experiment.

## Layout

| Path | Content |
| :--- | :--- |
| `train.py`, `test.py`, `eval_stf.py`, `test_robustness.py` | Entry points |
| `run_stf_seeds.sh` | STF runs with additional seeds |
| `src/experiment.py` | Config loading, dataset selection, checkpoint testing helpers |
| `src/system.py` | Lightning modules: optimization, metrics, logging |
| `src/models/` | Multimodal ViT, Segmenter, Cascade R-CNN detector and heads |
| `src/weight_adapters.py` | Loading of the pretrained initializations |
| `src/data/` | Datasets, augmentations, loaders |
| `src/eval/` | KITTI 2D evaluation (from OpenPCDet) |
| `data_preparation/` | Files derived from the official dataset releases |
| `utils/` | FMB validation split, result tables |

## License

MIT, see [LICENSE](../LICENSE). `src/eval/` is adapted from
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (Apache-2.0) and
`src/models/cascade_roi_heads.py` follows the Cascade R-CNN of
[Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0).
