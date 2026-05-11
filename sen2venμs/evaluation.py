import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import yaml
import piq
from affine import Affine
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import DataLoader, Dataset
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SFGSwinSR import MAGSwin2SR

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CURRENT_DIR / "config.yml"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class StandardSwin2SR(torch.nn.Module):
    def __init__(self, model_config, scale, img_size, num_channels):
        super().__init__()
        config = Swin2SRConfig(
            num_channels=num_channels,
            upscale=scale,
            img_size=img_size,
            window_size=model_config.get("window_size", 8),
            depths=model_config.get("depths", [6, 6, 6, 6, 6, 6]),
            num_heads=model_config.get("num_heads", [6, 6, 6, 6, 6, 6]),
            embed_dim=model_config.get("embed_dim", 180),
            mlp_ratio=model_config.get("mlp_ratio", 2.0),
        )
        self.swin2sr = Swin2SRForImageSuperResolution(config)

    def forward(self, pixel_values):
        return self.swin2sr(pixel_values=pixel_values).reconstruction


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate standalone SingleSR Sen2Venus checkpoints.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-type", "--model_type", dest="model_type", choices=("standard", "mag"), default="standard")
    parser.add_argument("--dataset-profile", "--dataset_profile", dest="dataset_profile", default=None)
    parser.add_argument("--lr-dir", "--lr_dir", dest="lr_dir", default=None)
    parser.add_argument("--hr-dir", "--hr_dir", dest="hr_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metric-backend", "--metric_backend", dest="metric_backend", choices=("skimage", "piq"), default="skimage")
    parser.add_argument("--save-bicubic-images", "--save_bicubic_images", dest="save_bicubic_images", action="store_true")
    parser.add_argument("--no-hr", "--no_hr", dest="no_hr", action="store_true")
    return parser


def resolve_args():
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
    dataset_root = profile_cfg["dataset_root"]

    if args.lr_dir is None:
        args.lr_dir = str(PROJECT_ROOT / dataset_root / profile_cfg["lr_subdir"])
    if args.hr_dir is None and not args.no_hr:
        args.hr_dir = str(PROJECT_ROOT / dataset_root / profile_cfg["hr_subdir"])
    if args.checkpoint is None:
        default_checkpoint_name = "best_SFG_swinSR.pt" if args.model_type == "mag" else "best_Swin2SR.pt"
        args.checkpoint = str(PROJECT_ROOT / profile_cfg["save_dir"] / default_checkpoint_name)
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "outputs" / "evaluation" / f"{args.model_type}_inference_{profile_name}")

    args.dataset_profile = profile_name
    args.scale = int(profile_cfg["scale"])
    args.num_channels = int(profile_cfg["num_channels"])
    args.lr_crop_size = int(profile_cfg["lr_crop_size"])
    args.lr_suffix = profile_cfg.get("lr_suffix")
    args.hr_suffix = profile_cfg.get("hr_suffix")
    args.normalization = profile_cfg["normalization"]
    args.model_config = cfg.get("model", {})
    return args


def list_image_paths(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS)
    if not paths:
        raise ValueError(f"No images found in: {image_dir}")
    return paths


def build_hr_lookup(hr_dir, hr_suffix=None):
    lookup = {}
    for path in list_image_paths(hr_dir):
        if hr_suffix and not path.name.endswith(hr_suffix):
            continue
        key = path.name[: -len(hr_suffix)] if hr_suffix else path.stem
        lookup[key] = path
    return lookup


class EvaluationDataset(Dataset):
    def __init__(
        self,
        lr_dir,
        hr_dir=None,
        scale=2,
        normalization=None,
        num_channels=4,
        lr_suffix=None,
        hr_suffix=None,
        max_samples=None,
    ):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir) if hr_dir else None
        self.scale = int(scale)
        self.normalization = normalization or {}
        self.num_channels = int(num_channels)
        self.lr_suffix = lr_suffix
        self.hr_suffix = hr_suffix
        self.hr_lookup = build_hr_lookup(self.hr_dir, hr_suffix=hr_suffix) if self.hr_dir else {}
        self.samples = self._build_samples()
        if max_samples is not None:
            self.samples = self.samples[: max_samples]
        if not self.samples:
            raise ValueError("No valid evaluation samples found.")

    def _lr_key(self, path):
        if self.lr_suffix and path.name.endswith(self.lr_suffix):
            return path.name[: -len(self.lr_suffix)]
        return path.stem

    def _build_samples(self):
        samples = []
        for lr_path in list_image_paths(self.lr_dir):
            if self.lr_suffix and not lr_path.name.endswith(self.lr_suffix):
                continue
            key = self._lr_key(lr_path)
            hr_path = self.hr_lookup.get(key) if self.hr_lookup else None
            if self.hr_dir and hr_path is None:
                continue
            samples.append({"name": key, "lr_path": lr_path, "hr_path": hr_path})
        return samples

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

    def _read(self, path, role):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
        img = self._select_channels(img, path)
        img = self._normalize(img, role)
        return torch.from_numpy(img.copy())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = {
            "name": sample["name"],
            "lr": self._read(sample["lr_path"], "lr"),
            "lr_path": str(sample["lr_path"]),
            "hr_path": str(sample["hr_path"]) if sample["hr_path"] is not None else "",
        }
        if sample["hr_path"] is not None:
            item["hr"] = self._read(sample["hr_path"], "hr")
        return item


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


def to_physical_space(tensor, role, normalization):
    return denormalize_tensor(tensor, normalization[f"{role}_mean"], normalization[f"{role}_std"])


def upsample_lr_to_hr_metric(lr_tensor, hr_size, normalization, mode):
    lr_physical = denormalize_tensor(
        lr_tensor,
        normalization["lr_mean"],
        normalization["lr_std"],
    )
    upsampled_physical = F.interpolate(
        lr_physical,
        size=hr_size,
        mode=mode,
        align_corners=False,
    )
    return to_printable_tensor(
        upsampled_physical,
        normalization["hr_min"],
        normalization["hr_max"],
    )


def compute_psnr(sr, hr, backend="skimage"):
    sr = torch.clamp(sr, 0.0, 1.0)
    hr = torch.clamp(hr, 0.0, 1.0)
    if backend == "piq":
        return float(piq.psnr(sr, hr, data_range=1.0, reduction="mean").item())
    sr_np = sr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    hr_np = hr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    return float(psnr_metric(hr_np, sr_np, data_range=1.0))


def compute_ssim(sr, hr, backend="skimage"):
    sr = torch.clamp(sr, 0.0, 1.0)
    hr = torch.clamp(hr, 0.0, 1.0)
    if backend == "piq":
        return float(piq.ssim(sr, hr, data_range=1.0, reduction="mean").item())
    sr_np = sr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    hr_np = hr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    return float(ssim_metric(hr_np, sr_np, data_range=1.0, channel_axis=-1))


def compute_mae(sr, hr):
    return float(torch.mean(torch.abs(torch.clamp(sr, 0.0, 1.0) - torch.clamp(hr, 0.0, 1.0))).item())


def build_model(args, device):
    model_kwargs = dict(args.model_config)
    if args.model_type == "mag":
        model_kwargs["img_size"] = args.lr_crop_size
        model_kwargs["upscale"] = args.scale
        model_kwargs["num_channels"] = args.num_channels
        model = MAGSwin2SR(**model_kwargs).to(device)
    else:
        model = StandardSwin2SR(
            model_config=model_kwargs,
            scale=args.scale,
            img_size=args.lr_crop_size,
            num_channels=args.num_channels,
        ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    return model


def save_sr_image(sr_physical_tensor, output_path, lr_path, hr_path, upscale, normalization):
    sr_np = sr_physical_tensor.squeeze(0).detach().cpu().numpy()
    source_path = Path(hr_path) if hr_path else Path(lr_path)
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        source_dtype = np.dtype(profile["dtype"])
        if hr_path:
            out_height, out_width = sr_np.shape[-2:]
            out_transform = transform
        else:
            out_height, out_width = sr_np.shape[-2:]
            out_transform = transform * Affine.scale(1 / upscale, 1 / upscale)

    # Restore the exported raster to the source data's physical domain.
    sr_np = np.clip(
        sr_np,
        np.asarray(normalization["hr_min"], dtype=np.float32)[:, None, None],
        np.asarray(normalization["hr_max"], dtype=np.float32)[:, None, None],
    )

    if np.issubdtype(source_dtype, np.integer):
        dtype_info = np.iinfo(source_dtype)
        sr_np = np.clip(sr_np, dtype_info.min, dtype_info.max)
        sr_np = np.rint(sr_np).astype(source_dtype)
    else:
        sr_np = sr_np.astype(source_dtype)

    profile.update(
        count=sr_np.shape[0],
        dtype=sr_np.dtype,
        height=out_height,
        width=out_width,
        transform=out_transform,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(sr_np)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = resolve_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    print(f"Dataset profile: {args.dataset_profile}")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Metric backend: {args.metric_backend}")

    dataset = EvaluationDataset(
        lr_dir=args.lr_dir,
        hr_dir=None if args.no_hr else args.hr_dir,
        scale=args.scale,
        normalization=args.normalization,
        num_channels=args.num_channels,
        lr_suffix=args.lr_suffix,
        hr_suffix=args.hr_suffix,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args, device)

    output_dir = Path(args.output_dir)
    sr_dir = output_dir / "sr"
    bicubic_dir = output_dir / "bicubic"
    sr_dir.mkdir(parents=True, exist_ok=True)
    if args.save_bicubic_images:
        bicubic_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ssim_values = []
    psnr_values = []
    mae_values = []
    bicubic_psnr_values = []
    bicubic_ssim_values = []
    bicubic_mae_values = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device) if "hr" in batch else None

            sr = model(pixel_values=lr)
            sr_physical = to_physical_space(sr, "hr", args.normalization)
            sr_metric = to_metric_space(sr, "hr", args.normalization)
            hr_metric = to_metric_space(hr, "hr", args.normalization) if hr is not None else None
            bicubic_metric = None
            bicubic_physical = None
            if hr_metric is not None:
                target_size = hr_metric.shape[-2:]
                bicubic_metric = upsample_lr_to_hr_metric(
                    lr,
                    target_size,
                    args.normalization,
                    mode="bicubic",
                )
                bicubic_physical = F.interpolate(
                    to_physical_space(lr, "lr", args.normalization),
                    size=target_size,
                    mode="bicubic",
                    align_corners=False,
                )
            elif args.save_bicubic_images:
                target_size = (
                    lr.shape[-2] * args.scale,
                    lr.shape[-1] * args.scale,
                )
                bicubic_metric = upsample_lr_to_hr_metric(
                    lr,
                    target_size,
                    args.normalization,
                    mode="bicubic",
                )
                bicubic_physical = F.interpolate(
                    to_physical_space(lr, "lr", args.normalization),
                    size=target_size,
                    mode="bicubic",
                    align_corners=False,
                )

            for i, name in enumerate(batch["name"]):
                sr_i = sr_metric[i : i + 1]
                sr_physical_i = sr_physical[i : i + 1]
                hr_i = hr_metric[i : i + 1] if hr_metric is not None else None
                bicubic_i = bicubic_metric[i : i + 1] if bicubic_metric is not None else None
                bicubic_physical_i = bicubic_physical[i : i + 1] if bicubic_physical is not None else None
                lr_path = batch["lr_path"][i]
                hr_path = batch["hr_path"][i]

                output_path = sr_dir / f"{name}_SR.tif"
                save_sr_image(
                    sr_physical_i,
                    output_path,
                    lr_path,
                    hr_path,
                    upscale=args.scale,
                    normalization=args.normalization,
                )
                row = {"name": name, "lr_path": lr_path, "hr_path": hr_path, "sr_path": str(output_path)}

                if args.save_bicubic_images:
                    bicubic_path = bicubic_dir / f"{name}_bicubic.tif"
                    save_sr_image(
                        bicubic_physical_i,
                        bicubic_path,
                        lr_path,
                        hr_path,
                        upscale=args.scale,
                        normalization=args.normalization,
                    )
                    row["bicubic_path"] = str(bicubic_path)

                if hr_i is not None:
                    psnr_value = compute_psnr(sr_i, hr_i, backend=args.metric_backend)
                    ssim_value = compute_ssim(sr_i, hr_i, backend=args.metric_backend)
                    mae_value = compute_mae(sr_i, hr_i)

                    joint_value = 40.0 * ssim_value + psnr_value
                    bicubic_psnr_value = compute_psnr(bicubic_i, hr_i, backend=args.metric_backend)
                    bicubic_ssim_value = compute_ssim(bicubic_i, hr_i, backend=args.metric_backend)
                    bicubic_mae_value = compute_mae(bicubic_i, hr_i)
                    row.update(
                        {
                            "psnr": f"{psnr_value:.6f}",
                            "ssim": f"{ssim_value:.6f}",
                            "mae": f"{mae_value:.6f}",
                            "joint": f"{joint_value:.6f}",
                            "bicubic_psnr": f"{bicubic_psnr_value:.6f}",
                            "bicubic_ssim": f"{bicubic_ssim_value:.6f}",
                            "bicubic_mae": f"{bicubic_mae_value:.6f}",
                        }
                    )
                    psnr_values.append(psnr_value)
                    ssim_values.append(ssim_value)
                    mae_values.append(mae_value)
                    bicubic_psnr_values.append(bicubic_psnr_value)
                    bicubic_ssim_values.append(bicubic_ssim_value)
                    bicubic_mae_values.append(bicubic_mae_value)

                rows.append(row)

    metrics_path = output_dir / "metrics.csv"
    if rows:
        write_csv(metrics_path, rows, list(rows[0].keys()))

    summary = {
        "dataset_profile": args.dataset_profile,
        "checkpoint": args.checkpoint,
        "num_samples": len(rows),
        "with_hr": not args.no_hr,
        "metric_backend": args.metric_backend,
    }
    if psnr_values:
        summary.update(
            {
                "psnr": float(np.mean(psnr_values)),
                "ssim": float(np.mean(ssim_values)),
                "mae": float(np.mean(mae_values)),
                "joint": float(np.mean(psnr_values) + 40.0 * np.mean(ssim_values)),
                "bicubic_psnr": float(np.mean(bicubic_psnr_values)),
                "bicubic_ssim": float(np.mean(bicubic_ssim_values)),
                "bicubic_mae": float(np.mean(bicubic_mae_values)),
            }
        )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")
    if psnr_values:
        print(
            f"PSNR: {summary['psnr']:.4f} | SSIM: {summary['ssim']:.4f} | "
            f"MAE: {summary['mae']:.6f} | JOINT: {summary['joint']:.4f}"
        )
        print(
            f"Bicubic PSNR: {summary['bicubic_psnr']:.4f} | "
            f"Bicubic SSIM: {summary['bicubic_ssim']:.4f} | "
            f"Bicubic MAE: {summary['bicubic_mae']:.6f}"
        )


if __name__ == "__main__":
    main()
