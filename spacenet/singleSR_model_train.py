"""
Training script for MAG-Swin2SR using the model defined in `SFGSwinSR.py`.

Notes:
    - Uses paired LR/HR raster images loaded with rasterio.
    - Keeps the same training structure as the original Swin2SR trainer.
    - For local PAN test data, a single channel is repeated into pseudo-RGB.
"""

import os
import random
from math import log10

import kornia
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model import MAGSwin2SR


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")

# LR_DIR = os.path.join(DATA_ROOT, "train_lr")
LR_DIR="/hrnas_user/users/aminur/tdp/Super-Resulution/data/singlesr/train_lr_clean"
HR_DIR="/hrnas_user/users/aminur/tdp/Super-Resulution/data/singlesr/train_hr_clean"

# HR_DIR = os.path.join(DATA_ROOT, "train_hr")
SAVE_DIR = os.path.join(OUTPUTS_ROOT, "checkpoints", "SFG_swinSR_test")
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
BIT_MAX_VALUE = 2047.0
SCALE = 2
LR_CROP_SIZE = 64
BATCH_SIZE = 4
NUM_WORKERS = 8
SPLIT_RATIO = 0.9
TOTAL_EPOCHS = 250
START_EPOCH = 1

LR = 2e-4
WEIGHT_DECAY = 1e-4
ETA_MIN = 1e-6
CLIP_GRAD_NORM = None

LAMBDA_L1 = 1.0
LAMBDA_SSIM = 0.1
LAMBDA_EDGE = 0.0
LAMBDA_FREQ = 0.0


MODEL_CONFIG = {
    "upscale": SCALE,
    "img_size": LR_CROP_SIZE,
    "window_size": 8,
    "depths": [6, 6, 6, 6, 6, 6],
    "num_heads": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_channels": 3,
    "mlp_ratio": 2.0,
    "sfg_kernel_size": 5,
    "drop": 0.0,
    "gate_reduction": 8,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ============================================================
# DATASET
# ============================================================

def list_image_paths(image_dir):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    paths = sorted(
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    )
    if not paths:
        raise ValueError(f"No images found in: {image_dir}")
    return paths


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, lr_crop_size=64, scale=2, bit_max_value=2047.0):
        self.lr_paths = list_image_paths(lr_dir)
        self.hr_paths = list_image_paths(hr_dir)

        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError(
                f"Mismatch in image count: LR={len(self.lr_paths)}, HR={len(self.hr_paths)}"
            )

        combined = list(zip(self.lr_paths, self.hr_paths))
        random.Random(SEED).shuffle(combined)
        self.lr_paths, self.hr_paths = zip(*combined)

        self.lr_crop_size = lr_crop_size
        self.hr_crop_size = lr_crop_size * scale
        self.scale = scale
        self.bit_max_value = float(bit_max_value)

    def read_and_normalize(self, path):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)

        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        elif img.shape[0] > 3:
            img = img[:3]

        img = np.clip(img / self.bit_max_value, 0.0, 1.0)
        return torch.from_numpy(img.copy())

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_tensor = self.read_and_normalize(self.lr_paths[idx])
        hr_tensor = self.read_and_normalize(self.hr_paths[idx])

        _, h_lr, w_lr = lr_tensor.shape
        if h_lr < self.lr_crop_size or w_lr < self.lr_crop_size:
            raise ValueError(
                f"LR image too small for crop: {self.lr_paths[idx]} has {(h_lr, w_lr)} "
                f"but crop is {self.lr_crop_size}"
            )

        i = random.randint(0, h_lr - self.lr_crop_size)
        j = random.randint(0, w_lr - self.lr_crop_size)

        lr_crop = lr_tensor[:, i:i + self.lr_crop_size, j:j + self.lr_crop_size]

        i_hr = i * self.scale
        j_hr = j * self.scale
        hr_crop = hr_tensor[:, i_hr:i_hr + self.hr_crop_size, j_hr:j_hr + self.hr_crop_size]

        if hr_crop.shape[-2:] != (self.hr_crop_size, self.hr_crop_size):
            raise ValueError(
                f"HR crop size mismatch for {self.hr_paths[idx]}. Got {hr_crop.shape[-2:]}, "
                f"expected {(self.hr_crop_size, self.hr_crop_size)}"
            )

        return {"lr": lr_crop, "hr": hr_crop}


def get_dataloaders(
    lr_dir,
    hr_dir,
    batch_size=4,
    split_ratio=0.9,
    num_workers=2,
    lr_crop_size=64,
    scale=2,
    bit_max_value=2047.0,
):
    full_dataset = SRDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        lr_crop_size=lr_crop_size,
        scale=scale,
        bit_max_value=bit_max_value,
    )

    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


# ============================================================
# LOSS
# ============================================================

class DetailGuidedSRLoss(nn.Module):
    def __init__(
        self,
        lambda_l1=1.0,
        lambda_ssim=0.1,
        lambda_edge=0.05,
        lambda_freq=0.01,
        window_size=5,
    ):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = kornia.losses.SSIMLoss(window_size=window_size, reduction="mean")
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_freq = lambda_freq

    @staticmethod
    def sobel_edges(x):
        if x.shape[1] == 3:
            gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        else:
            gray = x.mean(dim=1, keepdim=True)
        return kornia.filters.sobel(gray)

    @staticmethod
    def frequency_loss(sr, hr):
        sr_fft = torch.fft.rfft2(sr, norm="ortho")
        hr_fft = torch.fft.rfft2(hr, norm="ortho")
        sr_mag = torch.log1p(torch.abs(sr_fft))
        hr_mag = torch.log1p(torch.abs(hr_fft))
        return F.l1_loss(sr_mag, hr_mag)

    def forward(self, sr, hr):
        sr = torch.clamp(sr, 0.0, 1.0)
        hr = torch.clamp(hr, 0.0, 1.0)

        l1_loss = self.l1(sr, hr)
        ssim_loss = self.ssim(sr, hr)
        edge_loss = F.l1_loss(self.sobel_edges(sr), self.sobel_edges(hr))
        freq_loss = self.frequency_loss(sr, hr)

        return (
            self.lambda_l1 * l1_loss
            + self.lambda_ssim * ssim_loss
            + self.lambda_edge * edge_loss
            + self.lambda_freq * freq_loss
        )


# ============================================================
# METRICS
# ============================================================

def compute_psnr(sr, hr, max_pixel_value=1.0):
    sr = torch.clamp(sr, 0.0, 1.0)
    hr = torch.clamp(hr, 0.0, 1.0)
    mse = F.mse_loss(sr, hr)
    if mse.item() == 0:
        return float("inf")
    psnr = 20 * log10(max_pixel_value) - 10 * torch.log10(mse)
    return psnr.item()


def compute_joint_metric(output, target, bit_max_value=2047.0):
    output = torch.clamp(output, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    output_np = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    target_np = target.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    output_np = (output_np * bit_max_value).astype(np.float32)
    target_np = (target_np * bit_max_value).astype(np.float32)

    psnr_value = psnr_metric(target_np, output_np, data_range=bit_max_value)
    ssim_value = ssim_metric(
        target_np,
        output_np,
        data_range=bit_max_value,
        channel_axis=-1,
    )
    return 40 * ssim_value + psnr_value


def compute_ssim(output, target, bit_max_value=2047.0):
    output = torch.clamp(output, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    output_np = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    target_np = target.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    output_np = (output_np * bit_max_value).astype(np.float32)
    target_np = (target_np * bit_max_value).astype(np.float32)

    return ssim_metric(
        target_np,
        output_np,
        data_range=bit_max_value,
        channel_axis=-1,
    )


# ============================================================
# MODEL
# ============================================================

def build_model():
    return MAGSwin2SR(**MODEL_CONFIG)


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch in loop:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        outputs = torch.clamp(model(pixel_values=lr), 0.0, 1.0)
        loss = criterion(outputs, hr)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if CLIP_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

        optimizer.step()

        psnr = compute_psnr(outputs.detach(), hr)
        running_loss += loss.item()
        running_psnr += psnr

        loop.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}")

    avg_loss = running_loss / max(len(train_loader), 1)
    avg_psnr = running_psnr / max(len(train_loader), 1)
    return avg_loss, avg_psnr


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    val_joint = 0.0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        outputs = torch.clamp(model(pixel_values=lr), 0.0, 1.0)
        loss = criterion(outputs, hr)
        val_loss += loss.item()

        for i in range(outputs.size(0)):
            pred = outputs[i:i + 1]
            target = hr[i:i + 1]
            val_psnr += compute_psnr(pred, target)
            val_ssim += compute_ssim(pred, target, BIT_MAX_VALUE)
            val_joint += compute_joint_metric(pred, target, BIT_MAX_VALUE)

    n_samples = len(val_loader.dataset)
    return (
        val_loss / max(len(val_loader), 1),
        val_psnr / max(n_samples, 1),
        val_ssim / max(n_samples, 1),
        val_joint / max(n_samples, 1),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        batch_size=BATCH_SIZE,
        split_ratio=SPLIT_RATIO,
        num_workers=NUM_WORKERS,
        lr_crop_size=LR_CROP_SIZE,
        scale=SCALE,
        bit_max_value=BIT_MAX_VALUE,
    )

    model = build_model().to(device)

    criterion = DetailGuidedSRLoss(
        lambda_l1=LAMBDA_L1,
        lambda_ssim=LAMBDA_SSIM,
        lambda_edge=LAMBDA_EDGE,
        lambda_freq=LAMBDA_FREQ,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(TOTAL_EPOCHS - START_EPOCH, 1),
        eta_min=ETA_MIN,
    )

    best_psnr = -1.0

    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{TOTAL_EPOCHS} | LR: {current_lr:.8f}")

        train_loss, train_psnr = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        val_loss, val_psnr, val_ssim, val_joint = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | "
            f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | "
            f"Val SSIM: {val_ssim:.4f} | "
            f"Val JOINT: {val_joint:.2f}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = os.path.join(SAVE_DIR, "best_mag_swin2sr.pt")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path} | Best PSNR: {best_psnr:.2f} dB")

        latest_path = os.path.join(SAVE_DIR, "latest_mag_swin2sr.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "config": {
                    "data_root": DATA_ROOT,
                    "lr_dir": LR_DIR,
                    "hr_dir": HR_DIR,
                    "bit_max_value": BIT_MAX_VALUE,
                    "model": MODEL_CONFIG,
                },
            },
            latest_path,
        )

        if epoch % 5 == 0:
            periodic_path = os.path.join(SAVE_DIR, f"mag_swin2sr_epoch{epoch}.pt")
            torch.save(model.state_dict(), periodic_path)

    final_path = os.path.join(SAVE_DIR, "mag_swin2sr_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")
    print("Training completed.")


if __name__ == "__main__":
    main()
