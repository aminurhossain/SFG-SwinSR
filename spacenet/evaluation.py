import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from affine import Affine
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import DataLoader, Dataset
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution
from tqdm import tqdm

from model import MAGSwin2SR

DEFAULT_LR_DIR = "path/to/your/lr_dir"
DEFAULT_HR_DIR = "path/to/your/hr_dir"
DEFAULT_CHECKPOINT = "path/to/your/checkpoint"
DEFAULT_OUTPUT_DIR = "path/to/your/save_dir"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
BIT_MAX = 2047.0
SCALE = 2
IMG_SIZE = 64
WINDOW_SIZE = 8
DEPTHS = [6, 6, 6, 6, 6, 6]
NUM_HEADS = [6, 6, 6, 6, 6, 6]
EMBED_DIM = 180
NUM_CHANNELS = 3
MLP_RATIO = 2.0
EVAL_BATCH_SIZE = 1
PSNR_EPS = 1e-10


def print_model_parameter_count(model, label):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"{label}: total={total_params / 1e6:.2f}M | "
        f"trainable={trainable_params / 1e6:.2f}M"
    )


def strip_lr_variant_suffix(stem):
    return re.sub(r"_\d+$", "", stem)


class StandardSwin2SR(nn.Module):
    def __init__(self):
        super().__init__()
        config = Swin2SRConfig(
            num_channels=NUM_CHANNELS,
            upscale=SCALE,
            img_size=IMG_SIZE,
            window_size=WINDOW_SIZE,
            depths=DEPTHS,
            num_heads=NUM_HEADS,
            embed_dim=EMBED_DIM,
            mlp_ratio=MLP_RATIO,
        )
        self.swin2sr = Swin2SRForImageSuperResolution(config)

    def forward(self, pixel_values):
        return torch.clamp(
            self.swin2sr(pixel_values=pixel_values).reconstruction,
            0.0,
            1.0,
        )


class SingleSRTestDataset(Dataset):
    def __init__(self, lr_dir, hr_dir=None, bit_max=BIT_MAX):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir) if hr_dir else None
        self.bit_max = float(bit_max)

        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR directory not found: {self.lr_dir}")

        self.hr_lookup = self._build_hr_lookup(self.hr_dir) if self.hr_dir else {}
        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError("No valid evaluation samples found.")

    @staticmethod
    def _build_hr_lookup(hr_dir):
        if not hr_dir.exists():
            raise FileNotFoundError(f"HR directory not found: {hr_dir}")

        lookup = {}
        for path in sorted(hr_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTS:
                lookup[path.stem] = path
        return lookup

    def _match_hr_path(self, lr_path):
        if not self.hr_lookup:
            return None

        direct_match = self.hr_lookup.get(lr_path.stem)
        if direct_match is not None:
            return direct_match

        base_name = strip_lr_variant_suffix(lr_path.stem)
        return self.hr_lookup.get(base_name)

    def _build_samples(self):
        samples = []
        for lr_path in sorted(self.lr_dir.iterdir()):
            if not lr_path.is_file() or lr_path.suffix.lower() not in VALID_EXTS:
                continue

            hr_path = self._match_hr_path(lr_path)
            if self.hr_dir and hr_path is None:
                continue

            samples.append(
                {
                    "name": lr_path.stem,
                    "lr_path": lr_path,
                    "hr_path": hr_path,
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def _read_and_normalize(self, path):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)

        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        elif img.shape[0] > 3:
            img = img[:3]

        img = np.clip(img / self.bit_max, 0.0, 1.0)
        return torch.from_numpy(img.copy())

    def __getitem__(self, idx):
        sample = self.samples[idx]
        lr_tensor = self._read_and_normalize(sample["lr_path"])
        hr_tensor = (
            self._read_and_normalize(sample["hr_path"])
            if sample["hr_path"] is not None
            else None
        )

        if hr_tensor is not None:
            expected_h = lr_tensor.shape[1] * SCALE
            expected_w = lr_tensor.shape[2] * SCALE
            hr_tensor = hr_tensor[:, :expected_h, :expected_w]

            if hr_tensor.shape[-2:] != (expected_h, expected_w):
                raise ValueError(
                    f"HR size mismatch for {sample['name']}: got {tuple(hr_tensor.shape)}, "
                    f"expected (3, {expected_h}, {expected_w})"
                )

        return {
            "name": sample["name"],
            "lr": lr_tensor,
            "hr": hr_tensor,
            "lr_path": str(sample["lr_path"]),
            "hr_path": str(sample["hr_path"]) if sample["hr_path"] is not None else "",
        }


def eval_collate(batch):
    first_item = batch[0]
    hr_is_available = first_item["hr"] is not None

    lr_tensors = [item["lr"] for item in batch]
    hr_tensors = [item["hr"] for item in batch] if hr_is_available else None
    lr_shapes = {tuple(tensor.shape) for tensor in lr_tensors}
    hr_shapes = {tuple(tensor.shape) for tensor in hr_tensors} if hr_tensors else set()

    if len(lr_shapes) > 1 or len(hr_shapes) > 1:
        raise ValueError(
            "Evaluation batch contains different image sizes. "
            "Use --batch-size 1 for full-image evaluation."
        )

    return {
        "name": [item["name"] for item in batch],
        "lr": torch.stack(lr_tensors, dim=0),
        "hr": torch.stack(hr_tensors, dim=0) if hr_is_available else None,
        "lr_path": [item["lr_path"] for item in batch],
        "hr_path": [item["hr_path"] for item in batch],
    }


def compute_psnr(sr, hr):
    sr = torch.clamp(sr, 0.0, 1.0)
    hr = torch.clamp(hr, 0.0, 1.0)
    mse = torch.clamp(F.mse_loss(sr, hr), min=PSNR_EPS)
    return float(-10.0 * torch.log10(mse).item())


def compute_ssim(sr, hr, bit_max=BIT_MAX):
    sr_np = sr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    hr_np = hr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    sr_np = np.clip(sr_np, 0.0, 1.0).astype(np.float32) * bit_max
    hr_np = np.clip(hr_np, 0.0, 1.0).astype(np.float32) * bit_max

    return float(ssim_metric(hr_np, sr_np, data_range=bit_max, channel_axis=-1))


def compute_mae(sr, hr):
    return float(F.l1_loss(sr, hr).item())


def build_model(model_type, checkpoint_path, device):
    if model_type == "mag":
        model = MAGSwin2SR(
            upscale=SCALE,
            img_size=IMG_SIZE,
            window_size=WINDOW_SIZE,
            depths=DEPTHS,
            num_heads=NUM_HEADS,
            embed_dim=EMBED_DIM,
            num_channels=NUM_CHANNELS,
            mlp_ratio=MLP_RATIO,
        ).to(device)
    elif model_type == "standard":
        model = StandardSwin2SR().to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print_model_parameter_count(model, "Evaluation model")
    return model, checkpoint


def save_sr_image(sr_tensor, output_path, lr_path, hr_path, bit_max=BIT_MAX, upscale=SCALE):
    sr_np = sr_tensor.squeeze(0).detach().cpu().numpy()
    sr_np = np.clip(sr_np, 0.0, 1.0)
    sr_np = np.rint(sr_np * bit_max).astype(np.uint16)

    source_path = Path(hr_path) if hr_path else Path(lr_path)
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        transform = src.transform

    profile.update(
        dtype="uint16",
        count=sr_np.shape[0],
        height=sr_np.shape[1],
        width=sr_np.shape[2],
    )

    if not hr_path and transform is not None:
        profile["transform"] = transform * Affine.scale(1 / upscale, 1 / upscale)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(sr_np)


def write_metrics(output_dir, per_image_rows, summary):
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_image_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SingleSR checkpoints.")
    parser.add_argument("--model-type", choices=["mag", "standard"], default="mag")
    parser.add_argument("--lr-dir", default=str(DEFAULT_LR_DIR))
    parser.add_argument("--hr-dir", default=str(DEFAULT_HR_DIR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, cpu, etc.")
    parser.add_argument("--save-bicubic-images", action="store_true")
    parser.add_argument(
        "--no-hr",
        action="store_true",
        help="Run inference-only evaluation without HR targets or metrics.",
    )
    return parser.parse_args()


def run_evaluation(args):
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    checkpoint_path = Path(args.checkpoint)
    lr_dir = Path(args.lr_dir)
    hr_dir = None if args.no_hr else Path(args.hr_dir)
    output_dir = Path(args.output_dir)
    sr_dir = output_dir / "sr_images"
    bicubic_dir = output_dir / "bicubic_images"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not lr_dir.exists():
        raise FileNotFoundError(f"LR directory not found: {lr_dir}")
    if hr_dir is not None and not hr_dir.exists():
        raise FileNotFoundError(f"HR directory not found: {hr_dir}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    output_dir.mkdir(parents=True, exist_ok=True)
    sr_dir.mkdir(parents=True, exist_ok=True)
    if args.save_bicubic_images:
        bicubic_dir.mkdir(parents=True, exist_ok=True)

    dataset = SingleSRTestDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        bit_max=BIT_MAX,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eval_collate,
    )

    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"LR dir: {lr_dir}")
    print(f"HR dir: {hr_dir if hr_dir is not None else 'None'}")
    print(f"Output dir: {output_dir}")
    print(f"Found {len(dataset)} samples")

    model, checkpoint = build_model(args.model_type, checkpoint_path, device)
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    per_image_rows = []
    psnr_values = []
    ssim_values = []
    mae_values = []
    joint_values = []

    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating")
        for batch in loop:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device) if batch["hr"] is not None else None

            sr = torch.clamp(model(pixel_values=lr), 0.0, 1.0)
            batch_psnr_values = []
            batch_ssim_values = []

            for i, name in enumerate(batch["name"]):
                sr_i = sr[i : i + 1]
                hr_i = hr[i : i + 1] if hr is not None else None
                lr_i = lr[i : i + 1]
                lr_path = batch["lr_path"][i]
                hr_path = batch["hr_path"][i]

                output_path = sr_dir / f"{name}_SR.tif"
                save_sr_image(sr_i, output_path, lr_path, hr_path)

                row = {
                    "name": name,
                    "lr_path": lr_path,
                    "hr_path": hr_path,
                    "sr_path": str(output_path),
                }

                if args.save_bicubic_images:
                    if hr_i is not None:
                        bicubic = F.interpolate(
                            lr_i,
                            size=hr_i.shape[-2:],
                            mode="bicubic",
                            align_corners=False,
                        )
                    else:
                        bicubic = F.interpolate(
                            lr_i,
                            scale_factor=SCALE,
                            mode="bicubic",
                            align_corners=False,
                        )

                    bicubic = torch.clamp(bicubic, 0.0, 1.0)
                    bicubic_path = bicubic_dir / f"{name}_bicubic.tif"
                    save_sr_image(bicubic, bicubic_path, lr_path, hr_path)
                    row["bicubic_path"] = str(bicubic_path)

                if hr_i is not None:
                    psnr_value = compute_psnr(sr_i, hr_i)
                    ssim_value = compute_ssim(sr_i, hr_i)
                    mae_value = compute_mae(sr_i, hr_i)
                    joint_value = 40.0 * ssim_value + psnr_value

                    row.update(
                        {
                            "psnr": f"{psnr_value:.6f}",
                            "ssim": f"{ssim_value:.6f}",
                            "mae": f"{mae_value:.6f}",
                            "joint": f"{joint_value:.6f}",
                        }
                    )

                    psnr_values.append(psnr_value)
                    ssim_values.append(ssim_value)
                    mae_values.append(mae_value)
                    joint_values.append(joint_value)
                    batch_psnr_values.append(psnr_value)
                    batch_ssim_values.append(ssim_value)

                per_image_rows.append(row)

            if batch_psnr_values:
                loop.set_postfix(
                    batch_psnr=f"{np.mean(batch_psnr_values):.2f}",
                    avg_psnr=f"{np.mean(psnr_values):.2f}",
                    batch_ssim=f"{np.mean(batch_ssim_values):.4f}",
                    avg_ssim=f"{np.mean(ssim_values):.4f}",
                )
            else:
                loop.set_postfix(saved=len(per_image_rows))

    summary = {
        "num_samples": len(per_image_rows),
        "checkpoint": str(checkpoint_path),
        "model_type": args.model_type,
        "lr_dir": str(lr_dir),
        "hr_dir": str(hr_dir) if hr_dir is not None else None,
        "output_dir": str(output_dir),
    }

    if psnr_values:
        summary.update(
            {
                "mean_psnr": float(np.mean(psnr_values)),
                "mean_ssim": float(np.mean(ssim_values)),
                "mean_mae": float(np.mean(mae_values)),
                "mean_joint": float(np.mean(joint_values)),
            }
        )

    write_metrics(output_dir, per_image_rows, summary)

    print("\nEvaluation complete")
    if psnr_values:
        print(
            f"Mean PSNR: {summary['mean_psnr']:.4f} dB | "
            f"Mean SSIM: {summary['mean_ssim']:.4f} | "
            f"Mean MAE: {summary['mean_mae']:.6f} | "
            f"Mean JOINT: {summary['mean_joint']:.4f}"
        )
    print(f"Metrics CSV: {output_dir / 'metrics.csv'}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    return summary


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
