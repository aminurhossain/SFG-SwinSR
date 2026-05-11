import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SFGSwinSR import MAGSwin2SR


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "smoke_tests"
SPACE_LR_DIR = PROJECT_ROOT / "data" / "spacenet" / "train_lr"
SPACE_HR_DIR = PROJECT_ROOT / "data" / "spacenet" / "train_hr"
SEN2_2X_LR_DIR = PROJECT_ROOT / "data" / "sen2vnus" / "b2b3b4b8" / "10m"
SEN2_2X_HR_DIR = PROJECT_ROOT / "data" / "sen2vnus" / "b2b3b4b8" / "05m"
SEN2_4X_LR_DIR = PROJECT_ROOT / "data" / "sen2vnus" / "b5b6b7b8a" / "20m"
SEN2_4X_HR_DIR = PROJECT_ROOT / "data" / "sen2vnus" / "b5b6b7b8a" / "05m"
SEN2_CONFIG_PATH = PROJECT_ROOT / "sen2venμs" / "config.yml"

SPACE_SCALE = 2
SPACE_BIT_MAX = 2047.0
SPACE_LR_CROP = 64
SPACE_MODEL_CONFIG = {
    "upscale": SPACE_SCALE,
    "img_size": SPACE_LR_CROP,
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


def list_rasters(directory):
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".tif", ".tiff"})
    if not paths:
        raise FileNotFoundError(f"No raster files found in {directory}")
    return paths


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_raster(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.profile.copy()


def pair_by_suffix(lr_dir, hr_dir, lr_suffix, hr_suffix):
    lr_map = {}
    for path in list_rasters(lr_dir):
        if path.name.endswith(lr_suffix):
            lr_map[path.name[: -len(lr_suffix)]] = path

    hr_map = {}
    for path in list_rasters(hr_dir):
        if path.name.endswith(hr_suffix):
            hr_map[path.name[: -len(hr_suffix)]] = path

    common = sorted(lr_map.keys() & hr_map.keys())
    if not common:
        raise ValueError(f"No common pairs between {lr_dir} and {hr_dir}")
    key = common[0]
    return lr_map[key], hr_map[key], key


def crop_top_left(img, size):
    if img.shape[1] < size or img.shape[2] < size:
        raise ValueError(f"Image size {img.shape[1:]} is smaller than crop {size}")
    return img[:, :size, :size]


def save_raster(output_path, reference_profile, array_chw):
    profile = reference_profile.copy()
    profile.update(
        count=int(array_chw.shape[0]),
        height=int(array_chw.shape[1]),
        width=int(array_chw.shape[2]),
        dtype=array_chw.dtype,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array_chw)


def prepare_spacenet_sample():
    lr_path = list_rasters(SPACE_LR_DIR)[0]
    hr_path = SPACE_HR_DIR / lr_path.name
    if not hr_path.exists():
        raise FileNotFoundError(f"Matching HR file not found for {lr_path.name}")

    lr_img, _ = read_raster(lr_path)
    hr_img, hr_profile = read_raster(hr_path)

    if lr_img.shape[0] == 1:
        lr_img = np.repeat(lr_img, 3, axis=0)
    if hr_img.shape[0] == 1:
        hr_img = np.repeat(hr_img, 3, axis=0)
    lr_img = lr_img[:3]
    hr_img = hr_img[:3]

    lr_patch = crop_top_left(lr_img, SPACE_LR_CROP)
    hr_patch = crop_top_left(hr_img, SPACE_LR_CROP * SPACE_SCALE)

    lr_tensor = torch.from_numpy(np.clip(lr_patch / SPACE_BIT_MAX, 0.0, 1.0)).unsqueeze(0).float()
    hr_tensor = torch.from_numpy(np.clip(hr_patch / SPACE_BIT_MAX, 0.0, 1.0)).unsqueeze(0).float()

    return {
        "dataset": "spacenet",
        "sample_name": lr_path.stem,
        "lr_path": str(lr_path),
        "hr_path": str(hr_path),
        "lr_tensor": lr_tensor,
        "hr_tensor": hr_tensor,
        "hr_profile": hr_profile,
        "physical_export": "unit_interval_to_uint16_2047",
    }


def prepare_sen2_sample(profile_name, lr_dir, hr_dir, lr_suffix, hr_suffix):
    cfg = load_yaml(SEN2_CONFIG_PATH)
    profile_cfg = cfg["data"]["profiles"][profile_name]
    normalization = profile_cfg["normalization"]
    model_config = cfg["model"]

    lr_path, hr_path, name = pair_by_suffix(lr_dir, hr_dir, lr_suffix, hr_suffix)
    lr_img, _ = read_raster(lr_path)
    hr_img, hr_profile = read_raster(hr_path)

    num_channels = int(profile_cfg["num_channels"])
    scale = int(profile_cfg["scale"])
    lr_crop_size = int(profile_cfg["lr_crop_size"])
    hr_crop_size = lr_crop_size * scale

    lr_patch = crop_top_left(lr_img[:num_channels], lr_crop_size)
    hr_patch = crop_top_left(hr_img[:num_channels], hr_crop_size)

    lr_mean = np.asarray(normalization["lr_mean"], dtype=np.float32)[:num_channels]
    lr_std = np.asarray(normalization["lr_std"], dtype=np.float32)[:num_channels]
    hr_mean = np.asarray(normalization["hr_mean"], dtype=np.float32)[:num_channels]
    hr_std = np.asarray(normalization["hr_std"], dtype=np.float32)[:num_channels]

    lr_tensor = torch.from_numpy((lr_patch - lr_mean[:, None, None]) / lr_std[:, None, None]).unsqueeze(0).float()
    hr_tensor = torch.from_numpy((hr_patch - hr_mean[:, None, None]) / hr_std[:, None, None]).unsqueeze(0).float()

    return {
        "dataset": profile_name,
        "sample_name": name,
        "lr_path": str(lr_path),
        "hr_path": str(hr_path),
        "lr_tensor": lr_tensor,
        "hr_tensor": hr_tensor,
        "hr_profile": hr_profile,
        "normalization": normalization,
        "model_config": model_config,
        "num_channels": num_channels,
        "scale": scale,
        "lr_crop_size": lr_crop_size,
        "physical_export": "denormalized_hr_domain",
    }


def export_spacenet_sr(sr, sample, output_path):
    sr_uint16 = np.rint(np.clip(sr.squeeze(0).numpy(), 0.0, 1.0) * SPACE_BIT_MAX).astype(np.uint16)
    save_raster(output_path, sample["hr_profile"], sr_uint16)


def export_sen2_sr(sr, sample, output_path):
    hr_mean = np.asarray(sample["normalization"]["hr_mean"], dtype=np.float32)[: sample["num_channels"]]
    hr_std = np.asarray(sample["normalization"]["hr_std"], dtype=np.float32)[: sample["num_channels"]]
    hr_min = np.asarray(sample["normalization"]["hr_min"], dtype=np.float32)[: sample["num_channels"]]
    hr_max = np.asarray(sample["normalization"]["hr_max"], dtype=np.float32)[: sample["num_channels"]]

    sr_physical = sr.squeeze(0).numpy() * hr_std[:, None, None] + hr_mean[:, None, None]
    sr_physical = np.clip(sr_physical, hr_min[:, None, None], hr_max[:, None, None])

    reference_dtype = np.dtype(sample["hr_profile"]["dtype"])
    if np.issubdtype(reference_dtype, np.integer):
        info = np.iinfo(reference_dtype)
        sr_physical = np.rint(np.clip(sr_physical, info.min, info.max)).astype(reference_dtype)
    else:
        sr_physical = sr_physical.astype(reference_dtype)

    save_raster(output_path, sample["hr_profile"], sr_physical)


def run_case(case_name, dataset_name, model_name, model, sample, device, exporter):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        sr = model(sample["lr_tensor"].to(device)).cpu()

    output_path = OUTPUT_DIR / f"{case_name}.tif"
    exporter(sr, sample, output_path)

    return {
        "case": case_name,
        "dataset": dataset_name,
        "model": model_name,
        "status": "ok",
        "input_shape": list(sample["lr_tensor"].shape),
        "target_shape": list(sample["hr_tensor"].shape),
        "output_shape": list(sr.shape),
        "lr_path": sample["lr_path"],
        "hr_path": sample["hr_path"],
        "output_path": str(output_path),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spacenet_sample = prepare_spacenet_sample()
    sen2_2x_sample = prepare_sen2_sample(
        profile_name="sen2venus_2x",
        lr_dir=SEN2_2X_LR_DIR,
        hr_dir=SEN2_2X_HR_DIR,
        lr_suffix="_10m.tif",
        hr_suffix="_05m.tif",
    )
    sen2_4x_sample = prepare_sen2_sample(
        profile_name="sen2venus_4x",
        lr_dir=SEN2_4X_LR_DIR,
        hr_dir=SEN2_4X_HR_DIR,
        lr_suffix="_20m.tif",
        hr_suffix="_05m.tif",
    )

    cases = [
        (
            "spacenet_standard_forward",
            "spacenet",
            "StandardSwin2SR",
            StandardSwin2SR(SPACE_MODEL_CONFIG, SPACE_SCALE, SPACE_LR_CROP, 3),
            spacenet_sample,
            export_spacenet_sr,
        ),
        (
            "spacenet_mag_forward",
            "spacenet",
            "MAGSwin2SR",
            MAGSwin2SR(**SPACE_MODEL_CONFIG),
            spacenet_sample,
            export_spacenet_sr,
        ),
        (
            "sen2venus_2x_standard_forward",
            "sen2venus_2x",
            "StandardSwin2SR",
            StandardSwin2SR(
                sen2_2x_sample["model_config"],
                sen2_2x_sample["scale"],
                sen2_2x_sample["lr_crop_size"],
                sen2_2x_sample["num_channels"],
            ),
            sen2_2x_sample,
            export_sen2_sr,
        ),
        (
            "sen2venus_2x_mag_forward",
            "sen2venus_2x",
            "MAGSwin2SR",
            MAGSwin2SR(
                upscale=sen2_2x_sample["scale"],
                img_size=sen2_2x_sample["lr_crop_size"],
                num_channels=sen2_2x_sample["num_channels"],
                **sen2_2x_sample["model_config"],
            ),
            sen2_2x_sample,
            export_sen2_sr,
        ),
        (
            "sen2venus_4x_standard_forward",
            "sen2venus_4x",
            "StandardSwin2SR",
            StandardSwin2SR(
                sen2_4x_sample["model_config"],
                sen2_4x_sample["scale"],
                sen2_4x_sample["lr_crop_size"],
                sen2_4x_sample["num_channels"],
            ),
            sen2_4x_sample,
            export_sen2_sr,
        ),
        (
            "sen2venus_4x_mag_forward",
            "sen2venus_4x",
            "MAGSwin2SR",
            MAGSwin2SR(
                upscale=sen2_4x_sample["scale"],
                img_size=sen2_4x_sample["lr_crop_size"],
                num_channels=sen2_4x_sample["num_channels"],
                **sen2_4x_sample["model_config"],
            ),
            sen2_4x_sample,
            export_sen2_sr,
        ),
    ]

    report = {
        "project_root": str(PROJECT_ROOT),
        "device": str(device),
        "tests": [],
    }

    for case_name, dataset_name, model_name, model, sample, exporter in cases:
        try:
            report["tests"].append(run_case(case_name, dataset_name, model_name, model, sample, device, exporter))
        except Exception as exc:
            report["tests"].append(
                {
                    "case": case_name,
                    "dataset": dataset_name,
                    "model": model_name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report_path = OUTPUT_DIR / "smoke_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSmoke report written to {report_path}")


if __name__ == "__main__":
    main()
