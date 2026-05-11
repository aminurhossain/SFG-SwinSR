import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

from model import MAGSwin2SR
from evaluation import (
    DEFAULT_HR_DIR as DEFAULT_EVAL_HR_DIR,
    DEFAULT_LR_DIR as DEFAULT_EVAL_LR_DIR,
    run_evaluation,
)
from singleSR_model_train import (
    BIT_MAX_VALUE,
    BATCH_SIZE,
    DATA_ROOT,
    DetailGuidedSRLoss,
    ETA_MIN,
    HR_DIR,
    LR,
    LR_CROP_SIZE,
    LR_DIR,
    MODEL_CONFIG,
    NUM_WORKERS,
    SCALE,
    SEED,
    SPLIT_RATIO,
    WEIGHT_DECAY,
    get_dataloaders,
    set_seed,
    train_one_epoch,
    validate,
)


OUTPUTS_ROOT = "outputs"
ABLATIONS_ROOT = "ablations"
DEFAULT_CONFIG_PATH = "path/to/config_file"
os.makedirs(ABLATIONS_ROOT, exist_ok=True)

STANDARD_MODEL_KEYS = {
    "upscale",
    "img_size",
    "window_size",
    "depths",
    "num_heads",
    "embed_dim",
    "num_channels",
    "mlp_ratio",
}


@dataclass(frozen=True)
class AblationExperiment:
    name: str
    model_label: str
    backbone_label: str
    loss_label: str
    model_type: str
    lambda_l1: float
    lambda_ssim: float
    lambda_edge: float
    lambda_freq: float


class StandardSwin2SR(nn.Module):
    def __init__(self, model_config: Dict[str, object]):
        super().__init__()
        standard_config = {
            key: value
            for key, value in model_config.items()
            if key in STANDARD_MODEL_KEYS
        }
        self.swin2sr = Swin2SRForImageSuperResolution(Swin2SRConfig(**standard_config))

    def forward(self, pixel_values):
        return torch.clamp(
            self.swin2sr(pixel_values=pixel_values).reconstruction,
            0.0,
            1.0,
        )


EXPERIMENTS: List[AblationExperiment] = [
    AblationExperiment(
        name="swin2sr_l1",
        model_label="Swin2SR",
        backbone_label="Standard FFN",
        loss_label="L1",
        model_type="standard",
        lambda_l1=1.0,
        lambda_ssim=0.0,
        lambda_edge=0.0,
        lambda_freq=0.0,
    ),
    AblationExperiment(
        name="swin2sr_l1_ssim",
        model_label="Swin2SR",
        backbone_label="Standard FFN",
        loss_label="L1 + SSIM",
        model_type="standard",
        lambda_l1=1.0,
        lambda_ssim=0.1,
        lambda_edge=0.0,
        lambda_freq=0.0,
    ),
    AblationExperiment(
        name="swin2sr_full_loss",
        model_label="Swin2SR",
        backbone_label="Standard FFN",
        loss_label="L1 + SSIM + Edge + Freq",
        model_type="standard",
        lambda_l1=1.0,
        lambda_ssim=0.1,
        lambda_edge=0.05,
        lambda_freq=0.01,
    ),
    AblationExperiment(
        name="mag_swin2sr_l1",
        model_label="MAG-Swin2SR",
        backbone_label="MAG-FFN",
        loss_label="L1",
        model_type="mag",
        lambda_l1=1.0,
        lambda_ssim=0.0,
        lambda_edge=0.0,
        lambda_freq=0.0,
    ),
    AblationExperiment(
        name="mag_swin2sr_l1_ssim",
        model_label="MAG-Swin2SR",
        backbone_label="MAG-FFN",
        loss_label="L1 + SSIM",
        model_type="mag",
        lambda_l1=1.0,
        lambda_ssim=0.1,
        lambda_edge=0.0,
        lambda_freq=0.0,
    ),
    AblationExperiment(
        name="mag_swin2sr_full_loss",
        model_label="MAG-Swin2SR",
        backbone_label="MAG-FFN",
        loss_label="L1 + SSIM + Edge + Freq",
        model_type="mag",
        lambda_l1=1.0,
        lambda_ssim=0.1,
        lambda_edge=0.05,
        lambda_freq=0.01,
    ),
]


def build_model(model_type: str):
    if model_type == "mag":
        return MAGSwin2SR(**MODEL_CONFIG)
    if model_type == "standard":
        return StandardSwin2SR(MODEL_CONFIG)
    raise ValueError(f"Unsupported model type: {model_type}")


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_experiment_table(experiments: List[AblationExperiment]):
    print("\nAblation experiments")
    print("-" * 78)
    print(f"{'Name':<24} {'Model':<14} {'Backbone':<16} {'Loss'}")
    print("-" * 78)
    for exp in experiments:
        print(
            f"{exp.name:<24} "
            f"{exp.model_label:<14} "
            f"{exp.backbone_label:<16} "
            f"{exp.loss_label}"
        )
    print("-" * 78)


def load_config(config_path: str) -> Dict[str, object]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object.")

    return config


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    pre_args, remaining_argv = pre_parser.parse_known_args()

    config = load_config(pre_args.config)
    parser = argparse.ArgumentParser(
        description="Run paper ablations for Swin2SR and MAG-Swin2SR."
    )
    parser.add_argument(
        "--config",
        default=pre_args.config,
        help="Path to config.json for ablation runs.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "train_eval", "eval"],
        default="train",
        help="train: train only, train_eval: train then evaluate on test data, eval: evaluate existing checkpoints only.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        help="Experiment names to run, or 'all'. Use --list to see available names.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs per run.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for train/validation loaders.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Dataloader worker count.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=SPLIT_RATIO,
        help="Train/validation split ratio.",
    )
    parser.add_argument(
        "--train-lr-dir",
        "--lr-dir",
        dest="train_lr_dir",
        default=LR_DIR,
        help="Directory containing low-resolution training images.",
    )
    parser.add_argument(
        "--train-hr-dir",
        "--hr-dir",
        dest="train_hr_dir",
        default=HR_DIR,
        help="Directory containing high-resolution training images.",
    )
    parser.add_argument(
        "--save-root",
        default=ABLATIONS_ROOT,
        help="Root directory where ablation outputs will be saved.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cuda, cuda:0, cpu. Defaults to auto.",
    )
    parser.add_argument(
        "--test-lr-dir",
        "--eval-lr-dir",
        dest="test_lr_dir",
        default=str(DEFAULT_EVAL_LR_DIR),
        help="Directory containing low-resolution evaluation/test images.",
    )
    parser.add_argument(
        "--test-hr-dir",
        "--eval-hr-dir",
        dest="test_hr_dir",
        default=str(DEFAULT_EVAL_HR_DIR),
        help="Directory containing high-resolution evaluation/test images.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation. Keep 1 for variable full-image sizes.",
    )
    parser.add_argument(
        "--eval-checkpoint-name",
        choices=["best", "latest", "final"],
        default="best",
        help="Checkpoint file to evaluate per experiment.",
    )
    parser.add_argument(
        "--eval-save-bicubic-images",
        action="store_true",
        help="Also save bicubic baselines during evaluation.",
    )
    parser.add_argument(
        "--eval-no-hr",
        action="store_true",
        help="Run evaluation/inference without HR targets or test metrics.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available experiments and exit.",
    )

    parser.set_defaults(**config)
    return parser.parse_args(remaining_argv)


def select_experiments(selected_names: List[str]) -> List[AblationExperiment]:
    if not selected_names or selected_names == ["all"]:
        return EXPERIMENTS

    lookup = {exp.name: exp for exp in EXPERIMENTS}
    missing = [name for name in selected_names if name not in lookup]
    if missing:
        raise ValueError(
            "Unknown experiments: "
            + ", ".join(missing)
            + ". Use --list to inspect valid names."
        )
    return [lookup[name] for name in selected_names]


def get_checkpoint_path(exp_dir: str, checkpoint_name: str) -> str:
    checkpoint_map = {
        "best": os.path.join(exp_dir, "best.pt"),
        "latest": os.path.join(exp_dir, "latest.pt"),
        "final": os.path.join(exp_dir, "final.pt"),
    }
    return checkpoint_map[checkpoint_name]


def run_experiment_training(
    experiment: AblationExperiment,
    args,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(SEED)

    exp_dir = os.path.join(args.save_root, experiment.name)
    os.makedirs(exp_dir, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        lr_dir=args.train_lr_dir,
        hr_dir=args.train_hr_dir,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        num_workers=args.num_workers,
        lr_crop_size=LR_CROP_SIZE,
        scale=SCALE,
        bit_max_value=BIT_MAX_VALUE,
    )

    model = build_model(experiment.model_type).to(device)
    criterion = DetailGuidedSRLoss(
        lambda_l1=experiment.lambda_l1,
        lambda_ssim=experiment.lambda_ssim,
        lambda_edge=experiment.lambda_edge,
        lambda_freq=experiment.lambda_freq,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=ETA_MIN,
    )

    history: List[Dict[str, object]] = []
    best_summary = None
    best_psnr = float("-inf")

    print(
        f"\n=== {experiment.name} | {experiment.model_label} | "
        f"{experiment.loss_label} ==="
    )

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} | LR: {current_lr:.8f}")

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

        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_psnr": round(train_psnr, 6),
            "val_loss": round(val_loss, 6),
            "val_psnr": round(val_psnr, 6),
            "val_ssim": round(val_ssim, 6),
            "val_joint": round(val_joint, 6),
            "lr": round(current_lr, 10),
        }
        history.append(epoch_summary)

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | "
            f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | "
            f"Val SSIM: {val_ssim:.4f} | Val JOINT: {val_joint:.2f}"
        )

        latest_path = os.path.join(exp_dir, "latest.pt")

        if epoch % 2 == 0:
            periodic_path = os.path.join(exp_dir, f"model_epoch{epoch}.pt")
            torch.save(model.state_dict(), periodic_path)

        torch.save(
            {
                "epoch": epoch,
                "experiment": asdict(experiment),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": {
                    "data_root": DATA_ROOT,
                    "train_lr_dir": args.train_lr_dir,
                    "train_hr_dir": args.train_hr_dir,
                    "test_lr_dir": args.test_lr_dir,
                    "test_hr_dir": None if args.eval_no_hr else args.test_hr_dir,
                    "bit_max_value": BIT_MAX_VALUE,
                    "scale": SCALE,
                    "model_config": MODEL_CONFIG,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "split_ratio": args.split_ratio,
                },
            },
            latest_path,
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_summary = dict(epoch_summary)
            torch.save(model.state_dict(), os.path.join(exp_dir, "best.pt"))

    torch.save(model.state_dict(), os.path.join(exp_dir, "final.pt"))
    write_csv(
        os.path.join(exp_dir, "history.csv"),
        history,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_psnr",
            "val_loss",
            "val_psnr",
            "val_ssim",
            "val_joint",
            "lr",
        ],
    )

    summary = {
        "name": experiment.name,
        "model": experiment.model_label,
        "backbone": experiment.backbone_label,
        "loss": experiment.loss_label,
        "epochs": args.epochs,
        "best_epoch": best_summary["epoch"],
        "best_val_loss": best_summary["val_loss"],
        "best_val_psnr": best_summary["val_psnr"],
        "best_val_ssim": best_summary["val_ssim"],
        "best_val_joint": best_summary["val_joint"],
        "output_dir": exp_dir,
    }

    with open(os.path.join(exp_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def run_experiment_evaluation(
    experiment: AblationExperiment,
    args,
    exp_dir: str,
) -> Dict[str, object]:
    checkpoint_path = get_checkpoint_path(exp_dir, args.eval_checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint for evaluation not found: {checkpoint_path}. "
            f"Run training first or choose a different --eval-checkpoint-name."
        )

    eval_output_dir = os.path.join(exp_dir, "evaluation")
    eval_args = argparse.Namespace(
        model_type=experiment.model_type,
        lr_dir=args.test_lr_dir,
        hr_dir=args.test_hr_dir,
        checkpoint=checkpoint_path,
        output_dir=eval_output_dir,
        batch_size=args.eval_batch_size,
        device=args.device,
        save_bicubic_images=args.eval_save_bicubic_images,
        no_hr=args.eval_no_hr,
    )

    print(
        f"\n--- Evaluating {experiment.name} using {args.eval_checkpoint_name}.pt "
        f"on test data ---"
    )
    evaluation_summary = run_evaluation(eval_args)
    evaluation_summary["evaluation_output_dir"] = eval_output_dir
    evaluation_summary["evaluation_checkpoint"] = checkpoint_path
    return evaluation_summary


def main():
    args = parse_args()

    if args.list:
        print_experiment_table(EXPERIMENTS)
        return

    if args.mode in {"train", "train_eval"} and args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be at least 1.")

    experiments = select_experiments(args.experiments)

    os.makedirs(args.save_root, exist_ok=True)
    print_experiment_table(experiments)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    if args.mode in {"train", "train_eval"}:
        print(f"Train LR dir: {args.train_lr_dir}")
        print(f"Train HR dir: {args.train_hr_dir}")
    if args.mode in {"train_eval", "eval"}:
        print(f"Test LR dir: {args.test_lr_dir}")
        print(f"Test HR dir: {'None' if args.eval_no_hr else args.test_hr_dir}")
        print(f"Eval checkpoint: {args.eval_checkpoint_name}.pt")
    print(f"Outputs: {args.save_root}")

    summaries = []
    for experiment in experiments:
        exp_dir = os.path.join(args.save_root, experiment.name)
        summary = {
            "name": experiment.name,
            "model": experiment.model_label,
            "backbone": experiment.backbone_label,
            "loss": experiment.loss_label,
            "epochs": args.epochs if args.mode in {"train", "train_eval"} else "",
            "best_epoch": "",
            "best_val_loss": "",
            "best_val_psnr": "",
            "best_val_ssim": "",
            "best_val_joint": "",
            "test_mean_psnr": "",
            "test_mean_ssim": "",
            "test_mean_mae": "",
            "test_mean_joint": "",
            "evaluation_checkpoint": "",
            "evaluation_output_dir": "",
            "output_dir": exp_dir,
        }

        if args.mode in {"train", "train_eval"}:
            train_summary = run_experiment_training(experiment, args, device)
            summary.update(train_summary)

        if args.mode in {"train_eval", "eval"}:
            eval_summary = run_experiment_evaluation(experiment, args, exp_dir)
            summary.update(
                {
                    "test_mean_psnr": eval_summary.get("mean_psnr", ""),
                    "test_mean_ssim": eval_summary.get("mean_ssim", ""),
                    "test_mean_mae": eval_summary.get("mean_mae", ""),
                    "test_mean_joint": eval_summary.get("mean_joint", ""),
                    "evaluation_checkpoint": eval_summary["evaluation_checkpoint"],
                    "evaluation_output_dir": eval_summary["evaluation_output_dir"],
                }
            )

        summaries.append(summary)

    summary_csv_path = os.path.join(args.save_root, "ablation_summary_kite.csv")
    write_csv(
        summary_csv_path,
        summaries,
        fieldnames=[
            "name",
            "model",
            "backbone",
            "loss",
            "epochs",
            "best_epoch",
            "best_val_loss",
            "best_val_psnr",
            "best_val_ssim",
            "best_val_joint",
            "test_mean_psnr",
            "test_mean_ssim",
            "test_mean_mae",
            "test_mean_joint",
            "evaluation_checkpoint",
            "evaluation_output_dir",
            "output_dir",
        ],
    )

    print("\nAblation summary")
    if args.mode == "train":
        print("-" * 92)
        print(
            f"{'Name':<24} {'PSNR':>8} {'SSIM':>8} {'JOINT':>10} {'Best Epoch':>11}"
        )
        print("-" * 92)
        for row in summaries:
            print(
                f"{row['name']:<24} "
                f"{float(row['best_val_psnr']):>8.2f} "
                f"{float(row['best_val_ssim']):>8.4f} "
                f"{float(row['best_val_joint']):>10.2f} "
                f"{int(row['best_epoch']):>11}"
            )
        print("-" * 92)
    else:
        print("-" * 106)
        print(
            f"{'Name':<24} {'Val PSNR':>9} {'Test PSNR':>10} "
            f"{'Test SSIM':>10} {'Test JOINT':>12}"
        )
        print("-" * 106)
        for row in summaries:
            val_psnr = row["best_val_psnr"]
            val_psnr_text = f"{float(val_psnr):.2f}" if val_psnr != "" else "-"
            test_psnr = row["test_mean_psnr"]
            test_psnr_text = f"{float(test_psnr):.2f}" if test_psnr != "" else "-"
            test_ssim = row["test_mean_ssim"]
            test_ssim_text = f"{float(test_ssim):.4f}" if test_ssim != "" else "-"
            test_joint = row["test_mean_joint"]
            test_joint_text = f"{float(test_joint):.2f}" if test_joint != "" else "-"
            print(
                f"{row['name']:<24} "
                f"{val_psnr_text:>9} "
                f"{test_psnr_text:>10} "
                f"{test_ssim_text:>10} "
                f"{test_joint_text:>12}"
            )
        print("-" * 106)
    print(f"Saved summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
