import argparse
import os
import random
from math import log10

import kornia
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
    from .model import MAGSwin2SR
except ImportError:
    from model import MAGSwin2SR


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "config.yml")
SEED = 42


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_parser():
    parser = argparse.ArgumentParser(description="Train standalone SingleSR Sen2Venus profiles.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-profile", "--dataset_profile", dest="dataset_profile", default=None)
    parser.add_argument("--lr-dir", "--lr_dir", dest="lr_dir", default=None)
    parser.add_argument("--hr-dir", "--hr_dir", dest="hr_dir", default=None)
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--num-channels", "--num_channels", dest="num_channels", type=int, default=None)
    parser.add_argument("--lr-crop-size", "--lr_crop_size", dest="lr_crop_size", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--split-ratio", "--split_ratio", dest="split_ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--start-epoch", "--start_epoch", dest="start_epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--eta-min", "--eta_min", dest="eta_min", type=float, default=None)
    parser.add_argument("--clip-grad-norm", "--clip_grad_norm", dest="clip_grad_norm", type=float, default=None)
    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    profiles = data_cfg.get("profiles", {})
    profile_name = args.dataset_profile or data_cfg.get("dataset_profile", "sen2venus_2x")
    if profile_name not in profiles:
        raise ValueError(
            f"Unknown dataset profile '{profile_name}'. Available: {sorted(profiles.keys())}"
        )

    profile_cfg = profiles[profile_name]
    training_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {})

    dataset_root = profile_cfg["dataset_root"]
    if args.lr_dir is None:
        args.lr_dir = os.path.join(PROJECT_ROOT, dataset_root, profile_cfg["lr_subdir"])
    if args.hr_dir is None:
        args.hr_dir = os.path.join(PROJECT_ROOT, dataset_root, profile_cfg["hr_subdir"])
    if args.save_dir is None:
        args.save_dir = os.path.join(PROJECT_ROOT, profile_cfg["save_dir"])
    if args.scale is None:
        args.scale = profile_cfg["scale"]
    if args.num_channels is None:
        args.num_channels = profile_cfg["num_channels"]
    if args.lr_crop_size is None:
        args.lr_crop_size = profile_cfg["lr_crop_size"]
    if args.batch_size is None:
        args.batch_size = profile_cfg["batch_size"]
    if args.num_workers is None:
        args.num_workers = profile_cfg["num_workers"]
    if args.split_ratio is None:
        args.split_ratio = profile_cfg["split_ratio"]

    if args.epochs is None:
        args.epochs = training_cfg.get("epochs", 2)
    if args.start_epoch is None:
        args.start_epoch = training_cfg.get("start_epoch", 0)
    if args.lr is None:
        args.lr = training_cfg.get("lr", 2e-4)
    if args.weight_decay is None:
        args.weight_decay = training_cfg.get("weight_decay", 1e-4)
    if args.eta_min is None:
        args.eta_min = training_cfg.get("eta_min", 1e-6)
    if args.clip_grad_norm is None:
        args.clip_grad_norm = training_cfg.get("clip_grad_norm", None)

    args.dataset_profile = profile_name
    args.lr_suffix = profile_cfg.get("lr_suffix")
    args.hr_suffix = profile_cfg.get("hr_suffix")
    args.normalization = profile_cfg["normalization"]
    args.loss_config = loss_cfg
    args.model_config = model_cfg
    return args


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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


def build_paired_paths(lr_dir, hr_dir, lr_suffix=None, hr_suffix=None):
    lr_paths = list_image_paths(lr_dir)
    hr_paths = list_image_paths(hr_dir)

    def pair_key(path, suffix):
        name = os.path.basename(path)
        if suffix and not name.endswith(suffix):
            raise ValueError(f"File {name} does not end with expected suffix {suffix}")
        return name[: -len(suffix)] if suffix else os.path.splitext(name)[0]

    lr_map = {pair_key(path, lr_suffix): path for path in lr_paths}
    hr_map = {pair_key(path, hr_suffix): path for path in hr_paths}

    common_keys = sorted(lr_map.keys() & hr_map.keys())
    if not common_keys:
        raise ValueError(f"No paired files found between {lr_dir} and {hr_dir}")

    missing_lr = sorted(hr_map.keys() - lr_map.keys())
    missing_hr = sorted(lr_map.keys() - hr_map.keys())
    if missing_lr or missing_hr:
        raise ValueError(
            f"Pair mismatch between {lr_dir} and {hr_dir}. "
            f"Missing LR keys: {missing_lr[:5]}, Missing HR keys: {missing_hr[:5]}"
        )

    return [(lr_map[key], hr_map[key]) for key in common_keys]


class SRDataset(Dataset):
    def __init__(
        self,
        lr_dir,
        hr_dir,
        lr_crop_size,
        scale,
        normalization,
        num_channels,
        lr_suffix=None,
        hr_suffix=None,
    ):
        combined = build_paired_paths(lr_dir, hr_dir, lr_suffix=lr_suffix, hr_suffix=hr_suffix)
        random.Random(SEED).shuffle(combined)
        self.lr_paths, self.hr_paths = zip(*combined)
        self.lr_crop_size = lr_crop_size
        self.hr_crop_size = lr_crop_size * scale
        self.scale = scale
        self.normalization = normalization
        self.num_channels = int(num_channels)

    def _select_channels(self, img, path):
        if img.shape[0] < self.num_channels:
            raise ValueError(
                f"Image {path} has {img.shape[0]} channels, expected at least {self.num_channels}"
            )
        if img.shape[0] > self.num_channels:
            img = img[: self.num_channels]
        return img

    def _normalize(self, img, role):
        mean = np.asarray(self.normalization[f"{role}_mean"], dtype=np.float32)[: self.num_channels]
        std = np.asarray(self.normalization[f"{role}_std"], dtype=np.float32)[: self.num_channels]
        return (img - mean[:, None, None]) / std[:, None, None]

    def read_and_normalize(self, path, role):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
        img = self._select_channels(img, path)
        img = self._normalize(img, role)
        return torch.from_numpy(img.copy())

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_tensor = self.read_and_normalize(self.lr_paths[idx], role="lr")
        hr_tensor = self.read_and_normalize(self.hr_paths[idx], role="hr")

        _, h_lr, w_lr = lr_tensor.shape
        if h_lr < self.lr_crop_size or w_lr < self.lr_crop_size:
            raise ValueError(
                f"LR image too small for crop: {self.lr_paths[idx]} has {(h_lr, w_lr)} "
                f"but crop is {self.lr_crop_size}"
            )

        i = random.randint(0, h_lr - self.lr_crop_size)
        j = random.randint(0, w_lr - self.lr_crop_size)
        lr_crop = lr_tensor[:, i : i + self.lr_crop_size, j : j + self.lr_crop_size]

        i_hr = i * self.scale
        j_hr = j * self.scale
        hr_crop = hr_tensor[:, i_hr : i_hr + self.hr_crop_size, j_hr : j_hr + self.hr_crop_size]

        expected_size = (self.hr_crop_size, self.hr_crop_size)
        if hr_crop.shape[-2:] != expected_size:
            raise ValueError(
                f"HR crop size mismatch for {self.hr_paths[idx]}. Got {hr_crop.shape[-2:]}, "
                f"expected {expected_size}"
            )

        return {"lr": lr_crop, "hr": hr_crop}


def get_dataloaders(args):
    full_dataset = SRDataset(
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        lr_crop_size=args.lr_crop_size,
        scale=args.scale,
        normalization=args.normalization,
        num_channels=args.num_channels,
        lr_suffix=args.lr_suffix,
        hr_suffix=args.hr_suffix,
    )
    train_size = int(args.split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Invalid split_ratio={args.split_ratio} for dataset size={len(full_dataset)}. "
            "Need non-empty train and validation splits."
        )

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    return train_loader, val_loader


def denormalize_tensor(tensor, mean, std):
    mean_t = torch.as_tensor(mean, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    std_t = torch.as_tensor(std, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    return tensor * std_t + mean_t


def to_printable_tensor(tensor, min_vals, max_vals):
    min_t = torch.as_tensor(min_vals, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    max_t = torch.as_tensor(max_vals, device=tensor.device, dtype=tensor.dtype).view(1, -1, 1, 1)
    return torch.clamp((tensor - min_t) / (max_t - min_t + 1e-8), 0.0, 1.0)


def to_metric_space(tensor, role, normalization):
    denorm = denormalize_tensor(tensor, normalization[f"{role}_mean"], normalization[f"{role}_std"])
    return to_printable_tensor(denorm, normalization[f"{role}_min"], normalization[f"{role}_max"])


class DetailGuidedSRLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.1, lambda_edge=0.05, lambda_freq=0.01, window_size=5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = kornia.losses.SSIMLoss(window_size=window_size, reduction="mean")
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_freq = lambda_freq

    @staticmethod
    def sobel_edges(x):
        gray = x.mean(dim=1, keepdim=True)
        return kornia.filters.sobel(gray)

    @staticmethod
    def frequency_loss(sr, hr):
        sr_fft = torch.fft.rfft2(sr, norm="ortho")
        hr_fft = torch.fft.rfft2(hr, norm="ortho")
        return F.l1_loss(torch.log1p(torch.abs(sr_fft)), torch.log1p(torch.abs(hr_fft)))

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


def compute_psnr(sr, hr, max_pixel_value=1.0):
    sr = torch.clamp(sr, 0.0, 1.0)
    hr = torch.clamp(hr, 0.0, 1.0)
    mse = F.mse_loss(sr, hr)
    if mse.item() == 0:
        return float("inf")
    return (20 * log10(max_pixel_value) - 10 * torch.log10(mse)).item()


def compute_ssim(output, target, data_range=1.0):
    output_np = torch.clamp(output, 0.0, 1.0).squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    target_np = torch.clamp(target, 0.0, 1.0).squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    return ssim_metric(target_np, output_np, data_range=data_range, channel_axis=-1)


def compute_joint_metric(output, target, data_range=1.0):
    output_np = torch.clamp(output, 0.0, 1.0).squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    target_np = torch.clamp(target, 0.0, 1.0).squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    psnr_value = psnr_metric(target_np, output_np, data_range=data_range)
    ssim_value = ssim_metric(target_np, output_np, data_range=data_range, channel_axis=-1)
    return 40 * ssim_value + psnr_value

from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

def build_model(args):
    model_cfg = dict(args.model_config)
    config = Swin2SRConfig(
        num_channels=args.num_channels,
        upscale=args.scale,
        img_size=args.lr_crop_size,
        window_size=model_cfg.get("window_size", 8),
        depths=model_cfg.get("depths", [6, 6, 6, 6, 6, 6]),
        num_heads=model_cfg.get("num_heads", [6, 6, 6, 6, 6, 6]),
        embed_dim=model_cfg.get("embed_dim", 180),
    )
    return Swin2SRForImageSuperResolution(config)

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, normalization, clip_grad_norm):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    n_samples = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch in loop:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        outputs = model(pixel_values=lr).reconstruction
        pred_metric = to_metric_space(outputs, "hr", normalization)
        target_metric = to_metric_space(hr, "hr", normalization)
        loss = criterion(pred_metric, target_metric)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        for i in range(pred_metric.size(0)):
            running_psnr += compute_psnr(pred_metric[i : i + 1].detach(), target_metric[i : i + 1])
            n_samples += 1
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(len(train_loader), 1)
    avg_psnr = running_psnr / max(n_samples, 1)
    return avg_loss, avg_psnr


@torch.no_grad()
def validate(model, val_loader, criterion, device, normalization):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    val_joint = 0.0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        outputs = model(pixel_values=lr).reconstruction
        pred_metric = to_metric_space(outputs, "hr", normalization)
        target_metric = to_metric_space(hr, "hr", normalization)
        loss = criterion(pred_metric, target_metric)
        val_loss += loss.item()

        for i in range(pred_metric.size(0)):
            pred = pred_metric[i : i + 1]
            target = target_metric[i : i + 1]
            val_psnr += compute_psnr(pred, target)
            val_ssim += compute_ssim(pred, target)
            val_joint += compute_joint_metric(pred, target)

    n_samples = len(val_loader.dataset)
    return (
        val_loss / max(len(val_loader), 1),
        val_psnr / max(n_samples, 1),
        val_ssim / max(n_samples, 1),
        val_joint / max(n_samples, 1),
    )


def main():
    args = parse_args()
    set_seed(SEED)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset profile: {args.dataset_profile}")
    print(f"LR dir: {args.lr_dir}")
    print(f"HR dir: {args.hr_dir}")
    print(f"Save dir: {args.save_dir}")

    train_loader, val_loader = get_dataloaders(args)
    model = build_model(args).to(device)

    criterion = DetailGuidedSRLoss(
        lambda_l1=args.loss_config.get("lambda_l1", 1.0),
        lambda_ssim=args.loss_config.get("lambda_ssim", 0.1),
        lambda_edge=args.loss_config.get("lambda_edge", 0.05),
        lambda_freq=args.loss_config.get("lambda_freq", 0.01),
        window_size=args.loss_config.get("window_size", 5),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.start_epoch, 1), eta_min=args.eta_min)

    best_psnr = -1.0
    for epoch in range(args.start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} | LR: {current_lr:.8f}")

        train_loss, train_psnr = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            normalization=args.normalization,
            clip_grad_norm=args.clip_grad_norm,
        )
        val_loss, val_psnr, val_ssim, val_joint = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            normalization=args.normalization,
        )
        scheduler.step()

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | "
            f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | "
            f"Val SSIM: {val_ssim:.4f} | Val JOINT: {val_joint:.2f}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = os.path.join(args.save_dir, "best_SFG_swinSR.pt")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path} | Best PSNR: {best_psnr:.2f} dB")

        latest_path = os.path.join(args.save_dir, "latest_SFG_swinSR.pt")
        torch.save(
            {
                "epoch": epoch,
                "dataset_profile": args.dataset_profile,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "config": {
                    "lr_dir": args.lr_dir,
                    "hr_dir": args.hr_dir,
                    "save_dir": args.save_dir,
                    "scale": args.scale,
                    "num_channels": args.num_channels,
                    "lr_crop_size": args.lr_crop_size,
                    "normalization": args.normalization,
                    "model": args.model_config,
                },
            },
            latest_path,
        )

        if epoch % 2 == 0:
            periodic_path = os.path.join(args.save_dir, f"SFG_swinSR_epoch{epoch}.pt")
            torch.save(model.state_dict(), periodic_path)

    final_path = os.path.join(args.save_dir, "SFG_swinSR_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")
    print("Training completed.")


if __name__ == "__main__":
    main()
