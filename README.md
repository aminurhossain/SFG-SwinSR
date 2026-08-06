<h1 align="center">
  Spatial-Frequency Gated Swin Transformer for
Cross-Sensor Remote Sensing Super-Resolution
</h1>

<p align="center">
  <strong>
    Accepted at the <a href="https://terrabytes-workshop.github.io/">TerraBytes II Workshop, ECCV 2026</a>
  </strong>
</p>

<p align="center">
  Md Aminur Hossain<sup>1</sup>,
  Ayush V. Patel<sup>2</sup>,
  Sanjay K. Singh<sup>1</sup>,
  Yogesh Jethani<sup>3</sup>,
  Biplab Banerjee<sup>2</sup>
</p>

<p align="center">
  <sup>1</sup> Space Applications Centre, ISRO, Ahmedabad, India<br>
  <sup>2</sup> Centre of Studies in Resources Engineering, Indian Institute of Technology Bombay, India<br>
  <sup>3</sup> GLS University, Ahmedabad, India
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.09687"><img src="https://img.shields.io/badge/Preprint-arXiv-b31b1b"></a>
</p>

## Overview

SFG-SwinSR is a Swin2SR-based model for cross-sensor remote sensing super-resolution. It replaces the standard transformer feed-forward network with a **Spatial-Frequency Gated FFN (SFG-FFN)** that estimates low-frequency feature content, derives residual details, and reinjects useful details through lightweight adaptive gating.

This repository contains:

- the released model implementation in `SFGSwinSR.py`
- SpaceNet training and evaluation scripts under `spacenet/`
- training and evaluation scripts under `sen2venμs/`
- ablation utilities under `ablations/`
- release-safe README assets under `docs/images/`

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

The following metrics are taken from the final accepted paper.

| Dataset | Model | Params (M) | PSNR (dB) | SSIM | MAE |
|---|---|---:|---:|---:|---:|
| SEN2VENuS x2 | Swin2SR | 12.09 | 48.43 | 0.9932 | 0.0029 |
| SEN2VENuS x2 | Swin2-MoSE | 11.45 | 48.97 | 0.9937 | 0.0026 |
| SEN2VENuS x2 | **SFG-SwinSR** | 13.73 | **49.35** | **0.9949** | **0.0025** |
| SEN2VENuS x4 | Swin2SR | 12.09 | 44.12 | 0.9792 | 0.0049 |
| SEN2VENuS x4 | Swin2-MoSE | 11.49 | 45.12 | 0.9806 | 0.0045 |
| SEN2VENuS x4 | **SFG-SwinSR** | 13.73 | **45.52** | **0.9837** | **0.0042** |
| OLI2MSI x3 | Swin2SR | 12.09 | 45.32 | 0.9904 | 0.0039 |
| OLI2MSI x3 | Swin2-MoSE | 11.45 | **45.94** | 0.9912 | 0.0035 |
| OLI2MSI x3 | **SFG-SwinSR** | 13.73 | 45.91 | **0.9921** | **0.0032** |
| SEN2NAIP x4 | Swin2SR | 12.09 | 35.91 | 0.9348 | 0.0094 |
| SEN2NAIP x4 | Swin2-MoSE | 11.45 | 36.42 | 0.9397 | 0.0089 |
| SEN2NAIP x4 | **SFG-SwinSR** | 13.73 | **36.95** | **0.9442** | **0.0083** |
| SpaceNet x2 | Swin2SR | 12.09 | 43.56 | 0.9780 | 0.0039 |
| SpaceNet x2 | Swin2-MoSE | 11.45 | 44.61 | 0.9816 | 0.0034 |
| SpaceNet x2 | **SFG-SwinSR** | 13.73 | **45.19** | **0.9852** | **0.0031** |

Key takeaways reported in the paper:

- `+0.92 dB` PSNR over Swin2SR on SEN2VENuS x2
- `+1.40 dB` PSNR over Swin2SR on SEN2VENuS x4
- `+0.59 dB` PSNR over Swin2SR on OLI2MSI x3, while Swin2-MoSE remains `0.03 dB` higher
- `+1.04 dB` PSNR over Swin2SR on SEN2NAIP x4
- `+1.63 dB` PSNR over Swin2SR on SpaceNet x2

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
|-- SFGSwinSR.py
|-- README.md
|-- .gitignore
|-- docs/
|   `-- images/
|-- spacenet/
|   |-- evaluation.py
|   |-- singleSR_model_train.py
|   `-- config.json
|-- sen2venμs/
|   |-- train.py
|   |-- evaluation.py
|   `-- config.yml
`-- ablations/
    `-- ablation_runner.py
```

## Datasets and Configuration

The current repository is organized around experiments on:

- SpaceNet Challenge 3 / WorldView-3
- SEN2VENuS

The final paper additionally reports results on OLI2MSI and SEN2NAIP. This repository currently provides runnable training configurations and sample data for SpaceNet and SEN2VENuS.

The `data/` directory in this repository contains sample files only for smoke testing and command-line verification. It is not the full training or evaluation dataset release.

Original dataset sources:

- SpaceNet Roads Dataset: https://spacenet.ai/spacenet-roads-dataset/
- SEN2VENuS V2: https://zenodo.org/records/14603764

Dataset paths, scale settings, crop sizes, and normalization statistics are defined in:

- `spacenet/config.json`
- `sen2venμs/config.yml`

Standalone local path constants are also kept as placeholders in:

- `spacenet/singleSR_model_train.py`

Update those paths and values for your local environment before running experiments.

## Main Files

### `SFGSwinSR.py`

Contains:

- `SpatialFrequencyGatedFFN`
- the modified Swin2SR layer forward path
- the `MAGSwin2SR` wrapper used by the project

### `spacenet/evaluation.py`

Evaluation script for metrics and super-resolved image export.

### `spacenet/singleSR_model_train.py`

Training script for the proposed `MAGSwin2SR` / `SFG-SwinSR` model with directly embedded local constants.

### `sen2venμs/train.py`

Configuration-driven training entrypoint for the SEN2VENuS x2 and x4 profiles.

### `ablations/ablation_runner.py`

Runner for backbone and loss ablation experiments.

## Training and Evaluation

### SpaceNet

Train SFG-SwinSR:

```bash
python spacenet/singleSR_model_train.py
```

Evaluate:

```bash
python spacenet/evaluation.py \
  --model-type mag \
  --checkpoint path/to/your/checkpoint.pt \
  --lr-dir data/spacenet/train_lr \
  --hr-dir data/spacenet/train_hr \
  --output-dir outputs/spacenet/evaluation
```

### SEN2VENuS

Train:

```bash
python "sen2venμs/train.py" --config "sen2venμs/config.yml"
```

Evaluate:

```bash
python "sen2venμs/evaluation.py" \
  --model-type mag \
  --checkpoint path/to/your/checkpoint.pt \
  --lr-dir data/sen2vnus/b2b3b4b8/10m \
  --hr-dir data/sen2vnus/b2b3b4b8/05m \
  --output-dir outputs/sen2venus/evaluation
```

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

Install these packages before running the training or evaluation scripts:

```bash
pip install -r requirements.txt
```

## Release Notes

- The figures shown in this README are copied into `docs/images/` so the repository can render them without tracking the full paper workspace.
- Dataset locations and output directories remain local configuration items and should be updated before use.

## Citation

If you use this repository, cite the corresponding paper:

The arXiv record currently contains an earlier preprint. The citation below reflects the accepted final paper:

```bibtex
@misc{hossain2026sfgswinsr,
  title={Spatial-Frequency Gated Swin Transformer for Cross-Sensor Remote Sensing Super-Resolution},
  author={Hossain, Md Aminur and Patel, Ayush V. and Singh, Sanjay K. and Jethani, Yogesh and Banerjee, Biplab},
  year={2026},
  note={Accepted at the TerraBytes II Workshop, ECCV 2026},
  url={https://arxiv.org/abs/2605.09687}
}
```
