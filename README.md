# SFG-SwinSR

Spatial-Frequency Gated Swin Transformer for remote sensing single-image super-resolution.

## Overview

SFG-SwinSR is a Swin2SR-based super-resolution model for remote sensing imagery. The main architectural change is the replacement of the standard transformer feed-forward network with a **Spatial-Frequency Gated FFN (SFG-FFN)** designed to separate low-frequency structure from high-frequency residual detail and reinject useful detail through lightweight gating.

This repository contains:

- the released model implementation in `model.py`
- training and evaluation scripts under `spacenet/`
- ablation utilities under `ablations/`
- release-safe README assets under `docs/images/`

## Paper

- arXiv: `Add your arXiv link here`

## Architecture

<p align="center">
  <img src="docs/images/architecture.png" alt="SFG-SwinSR architecture" width="900">
</p>

The model keeps the Swin2SR attention backbone and modifies the FFN inside each transformer block. The SFG-FFN estimates low-frequency content with depthwise blur, extracts high-frequency residuals, refines them spatially, and applies adaptive gating before projection back to the token space.

## SpaceNet Pair Generation

<p align="center">
  <img src="docs/images/hr-to-lr-pipeline.png" alt="HR to LR simulation pipeline" width="900">
</p>

For SpaceNet, low-resolution inputs are synthetically generated from high-resolution imagery using the paper's degradation pipeline: blur, bicubic downsampling, and noise injection.

## Headline Results

The following metrics are taken from the paper results currently included in the LaTeX sources under `SingleSR/`.

| Dataset | Model | Params (M) | PSNR (dB) | SSIM | MAE |
|---|---|---:|---:|---:|---:|
| SpaceNet | Swin2SR | 12.09 | 43.56 | 0.9780 | 0.0039 |
| SpaceNet | Swin2-MoSE | 11.45 | 44.61 | 0.9816 | 0.0034 |
| SpaceNet | **SFG-SwinSR** | **13.73** | **45.19** | **0.9852** | **0.0031** |
| SEN2VENuS x2 | Swin2SR | 12.09 | 48.43 | 0.9932 | 0.0029 |
| SEN2VENuS x2 | Swin2-MoSE | 11.45 | 48.97 | 0.9937 | 0.0026 |
| SEN2VENuS x2 | **SFG-SwinSR** | **13.73** | **49.35** | **0.9949** | **0.0025** |
| SEN2VENuS x4 | Swin2SR | 12.09 | 44.12 | 0.9792 | 0.0049 |
| SEN2VENuS x4 | Swin2-MoSE | 11.49 | 45.12 | 0.9806 | 0.0045 |
| SEN2VENuS x4 | **SFG-SwinSR** | **13.73** | **45.52** | **0.9837** | **0.0042** |

Key takeaways reported in the paper:

- `+1.63 dB` PSNR over Swin2SR on SpaceNet
- `+0.92 dB` PSNR over Swin2SR on SEN2VENuS x2
- `+1.40 dB` PSNR over Swin2SR on SEN2VENuS x4

## Qualitative Results

### SpaceNet

<p align="center">
  <img src="docs/images/spacenet-qualitative.png" alt="SpaceNet qualitative comparison" width="1000">
</p>

Visual comparison on SpaceNet. From left to right: Ground Truth, Bicubic, Swin2SR, Swin2-MoSE, and SFG-SwinSR.

### SEN2VENuS

<p align="center">
  <img src="docs/images/sen2venus-qualitative.png" alt="SEN2VENuS qualitative comparison" width="1000">
</p>

Visual comparison on SEN2VENuS. The paper reports improved boundary recovery, finer structural detail, and lower reconstruction error under the evaluated settings.

## Repository Layout

```text
SFG-SwinSR/
|-- model.py
|-- README.md
|-- .gitignore
|-- docs/
|   `-- images/
|-- spacenet/
|   |-- train.py
|   |-- evaluation.py
|   |-- singleSR_model_train.py
|   |-- config.yml
|   |-- config.json
|   `-- info.txt
`-- ablations/
    |-- ablation_runner.py
    `-- info.txt
```

## Datasets and Configuration

The current repository is organized around experiments on:

- SpaceNet / WorldView-2
- SEN2VENuS

Dataset paths, scale settings, crop sizes, and normalization statistics are defined in:

- `spacenet/config.yml`
- `spacenet/config.json`

Update those paths and values for your local environment before running experiments.

## Main Files

### `model.py`

Contains:

- `SpatialFrequencyGatedFFN`
- the modified Swin2SR layer forward path
- the `MAGSwin2SR` wrapper used by the project

### `spacenet/train.py`

Configuration-driven training entrypoint.

### `spacenet/evaluation.py`

Evaluation script for metrics and super-resolved image export.

### `spacenet/singleSR_model_train.py`

Standalone training script with directly embedded local constants.

### `ablations/ablation_runner.py`

Runner for backbone and loss ablation experiments.

## Environment

The scripts depend mainly on:

- Python 3.8+
- PyTorch
- Transformers
- Rasterio
- NumPy
- Kornia
- scikit-image
- PyYAML
- tqdm

Install these packages in your environment before running the training or evaluation scripts.

## Release Notes

- The paper workspace folder `SingleSR/` is intentionally ignored in `.gitignore`.
- The figures shown in this README are copied into `docs/images/` so the repository can render them without tracking the full paper workspace.
- Dataset locations and output directories remain local configuration items and should be updated before use.

## Citation

If you use this repository, cite the corresponding paper:

```bibtex
@article{hossain2026sfgswinsr,
  title={Spatial-Frequency Gated Swin Transformer for Remote Sensing Single-Image Super-Resolution},
  author={Hossain, Md Aminur and others},
  year={2026}
}
```
