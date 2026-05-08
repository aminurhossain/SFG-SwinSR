# SFG-SwinSR: Spatial-Frequency Gated Swin Transformer for Remote Sensing Single-Image Super-Resolution

## Overview

**SFG-SwinSR** is a Transformer-based model for **single-image super-resolution (SISR)** of remote sensing imagery. Built on the **Swin2SR backbone**, SFG-SwinSR replaces the standard feed-forward network (FFN) in each transformer block with a **Spatial-Frequency Gated FFN (SFG-FFN)**. This module separates low-frequency structural content from high-frequency residual detail, enhancing fine spatial structures such as roads, rooftops, and land-cover boundaries without modifying the Swin2SR attention mechanism.

The low-frequency branch uses a **depthwise blur filter**, high-frequency residuals are extracted by subtraction, refined through a lightweight spatial branch, and injected adaptively via a **bottleneck gate**.  

Experiments on **SpaceNet/WorldView-2** and **SEN2VENµS** datasets demonstrate improved reconstruction quality over baseline Swin2SR and Swin2-MoSE models.

---

## Features

- Transformer-based RS-SISR with Swin2SR backbone.
- Spatial-Frequency Gated FFN for high-frequency detail enhancement.
- Supports **multi-band imagery** (RGB and 4-band datasets).
- Up to **2× and 4× upscaling**.
- Training scripts with **L1, SSIM, Edge, and Frequency-guided losses**.
- Evaluation scripts for **PSNR, SSIM, and MAE** metrics.
- Ablation support to analyze loss functions and architectural variations.

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/SFG-SwinSR.git
cd SFG-SwinSR

# Install dependencies (Python 3.8+ recommended)
pip install -r requirements.txt
```

**Dependencies:**  
- PyTorch >=1.10  
- Torchvision  
- Numpy, OpenCV  
- tqdm, matplotlib  

---

## Usage

### 1. Training

```bash
python train.py \
  --dataset spacenet \
  --scale 2 \
  --bands 3 \
  --loss l1+ssim+edge+freq \
  --epochs 200 \
  --batch_size 16
```

### 2. Evaluation

```bash
python evaluate.py \
  --dataset spacenet \
  --checkpoint checkpoints/sfg_swin2sr.pth \
  --scale 2
```

### 3. Inference on a single image

```bash
python inference.py --input lr_image.png --output sr_image.png --scale 2
```

---

## Results

- **SpaceNet (2×):** PSNR: 45.19 dB, SSIM: 0.9846  
- **SEN2VENµS (2×):** PSNR: 49.35 dB, SSIM: 0.9949  
- SFG-SwinSR consistently **enhances high-frequency details** compared to Swin2SR and Swin2-MoSE.

**Qualitative example:**  
![Qualitative Result](examples/qualitative_example.png)

---

## Citation

If you use this repository, please cite:

```bibtex
@article{your2026SFGSwinSR,
  title={SFG-SwinSR: Spatial-Frequency Gated Swin Transformer for Remote Sensing Single-Image Super-Resolution},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2026}
}
```

---

## License

MIT License – see [LICENSE](LICENSE)
