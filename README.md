
# Do Pretrained Backbones Transfer to Physical Sensors?

<p align="center">
  <a href="https://wacv.thecvf.com/"><img src="https://img.shields.io/badge/WACV-2027-1f6feb" alt="WACV 2027"></a>
  <a href="https://huggingface.co/dtu-pcas/sensormae"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-weights-ffcc4d" alt="Weights on Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.13-3776ab?logo=python&logoColor=white" alt="Python 3.13">
  <img src="https://img.shields.io/badge/PyTorch-2.8-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch 2.8">
</p>

Official repository of the WACV 2027 paper *"Do Pretrained Backbones Transfer to Physical
Sensors? A Systematic Study of Multimodal Initialization Across the Sensor Stack"*.

We pretrain **SensorMAE**, a multimodal masked autoencoder over the sensors found on
robots and vehicles (RGB, thermal LWIR/MWIR, reflective NIR/SWIR, event cameras, stereo
depth, plus semantic segmentation), and compare it with RGB-only and multimodal
initializations (ViT, MAE, DINO, MultiMAE) on downstream segmentation and detection with
RGB, thermal, NIR, event, depth, LiDAR and radar inputs.

<p align="center">
  <img src="assets/banner_side.png" width="48.9%" alt="SensorMAE reconstructions of RGB, NIR, LWIR, depth, event and segmentation from a few visible patches of the available modalities">
  <img src="assets/main_figure.png" width="49.1%" alt="SensorMAE architecture: per-modality tokenizers, a shared Vision Transformer, an MLP bottleneck and per-modality decoders">
</p>


## Contents

| Folder | What it contains |
| :--- | :--- |
| [`preprocessing/`](preprocessing/) | Rebuilds the multi-sensor pretraining collection (16 public datasets, ~826k aligned samples) from the original releases, with the exact sample selection and splits of the paper. |
| [`benchmarking/`](benchmarking/) | Fine-tuning and evaluation on the downstream benchmarks: 256 experiment configs over Cityscapes, BDD100K, DSEC, FMB, MFNet, KITTI, LLVIP and SeeingThroughFog. |

Each folder has its own README, requirements and instructions.

## Pretrained weights

The SensorMAE encoders are released on the Hugging Face Hub:
[`dtu-pcas/sensormae`](https://huggingface.co/dtu-pcas/sensormae).

| File | Encoder | Parameters | Pretraining initialized from |
| :--- | :--- | ---: | :--- |
| `sensormae-base-mm` | ViT-B/16 | 101.7M | MultiMAE ViT-B |
| `sensormae-base-vit` | ViT-B/16 | 101.7M | ViT-B (ImageNet-21k) |
| `sensormae-small-vit` | ViT-S/16 | 29.6M | ViT-S (ImageNet-21k) |
| `sensormae-tiny-vit` | ViT-Ti/16 | 9.5M | ViT-Ti (ImageNet-21k) |

Every file holds the encoder only: one patch embedding, positional embedding and modality
embedding per pretraining modality (`rgb`, `lmwir`, `snir`, `depth`, `event`, `segm`), and
the shared transformer blocks. They are provided as `.pth` (read by the benchmarking code)
and `.safetensors` (same tensors).

Download them where the benchmarking configs expect them:

```bash
pip install -U huggingface_hub
hf download dtu-pcas/sensormae --include "*.pth" --local-dir benchmarking/weights
```

The MultiMAE baseline uses the official MultiMAE checkpoint:

```bash
wget -P benchmarking/weights https://github.com/EPFL-VILAB/MultiMAE/releases/download/pretrained-weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth
```

The checkpoints contain tensors only and can be read with
`torch.load(path, weights_only=True)` or `safetensors.torch.load_file(path)`.

## Quick start

```bash
git clone https://github.com/DTU-PAS/sensormae && cd sensormae/benchmarking
python -m venv .env && source .env/bin/activate
pip install -r requirements.txt
cp machine_config.example.yaml machine_config.yaml
hf download dtu-pcas/sensormae --include "*.pth" --local-dir weights

# prepare a dataset under data/ (see benchmarking/README.md), then for example
python train.py --config configs/segmentation/mfnet/2_mfnet_rgb_lwir_sensormae_base_mm.yaml
python test.py  --config configs/segmentation/mfnet/2_mfnet_rgb_lwir_sensormae_base_mm.yaml
```

## Citation

The paper has been accepted at WACV 2027; the BibTeX entry will be added here with the
proceedings.

## License

The code and the SensorMAE weights are released under the [MIT License](LICENSE).
The datasets are not redistributed and keep their original licenses. Third-party code
adapted in `benchmarking/` (OpenPCDet, Detectron2) is listed in its
[README](benchmarking/README.md#license).
