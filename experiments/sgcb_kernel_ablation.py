"""Run SGCB kernel-size ablation variants for the UltraLite-Nano model.

The current manuscript uses a 7x7 SGCB spatial-gating kernel. This script
creates controlled 5x5 and 9x9 variants from the existing UltraLite-Nano YAML
and trains/evaluates them with the same protocol as the full Nano ablation.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = REPO_ROOT / "third_party" / "ultralytics"
if LOCAL_ULTRALYTICS.exists():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


TRAINING_CONFIG = {
    "patience": 10,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.1,
    "rect": False,
    "cos_lr": True,
    "close_mosaic": 10,
    "amp": True,
    "fraction": 1.0,
    "profile": False,
    "deterministic": True,
    "seed": 0,
    "workers": 8,
    "exist_ok": True,
    "plots": True,
    "save": True,
    "save_period": 25,
    "cache": "disk",
    "nbs": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGCB kernel-size ablation runner")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "configs" / "VisDrone.yaml")
    parser.add_argument("--base-cfg", type=Path, default=model_cfg_dir() / "yolov8n-ultraliteattn-local.yaml")
    parser.add_argument("--kernels", type=int, nargs="+", default=[5, 9], help="Odd SGCB kernels to test.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "results" / "sgcb_kernel_ablation")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs" / "generated")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only generate YAML configs and print planned runs.")
    parser.add_argument("--force", action="store_true", help="Retrain even when a variant best.pt already exists.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing best.pt files.")
    return parser.parse_args()


def model_cfg_dir() -> Path:
    return REPO_ROOT / "third_party" / "ultralytics" / "ultralytics" / "cfg" / "models" / "v8"


def validate_kernel(kernel: int) -> None:
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError(f"SGCB kernel must be a positive odd integer, got {kernel}.")


def generate_kernel_config(base_cfg: Path, output_dir: Path, kernel: int) -> Path:
    """Create a model YAML with every explicit SGCB k_attn value replaced."""
    validate_kernel(kernel)
    source = base_cfg.read_text(encoding="utf-8")
    replacement_count = 0
    output_lines = []
    pattern = re.compile(r"(SGCB,\s*\[[^\]]*?,\s*)7(\]\])")

    for line in source.splitlines():
        if "SGCB" in line:
            line, count = pattern.subn(rf"\g<1>{kernel}\2", line)
            replacement_count += count
        output_lines.append(line)

    if replacement_count == 0:
        raise RuntimeError(f"No SGCB k_attn=7 entries were replaced in {base_cfg}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{base_cfg.stem}-k{kernel}.yaml"
    header = [
        f"# Auto-generated from {base_cfg.as_posix()}",
        f"# SGCB spatial-gating kernel: {kernel}x{kernel}",
        "",
    ]
    out_path.write_text("\n".join(header + output_lines) + "\n", encoding="utf-8")
    return out_path


def result_project(args: argparse.Namespace) -> Path:
    suffix = f"ep{args.epochs}_bs{args.batch}_img{args.imgsz}_pat{TRAINING_CONFIG['patience']}"
    return args.project.with_name(f"{args.project.name}_{suffix}")


def run_name(kernel: int) -> str:
    return f"ultralite_nano_sgcb_k{kernel}"


def best_checkpoint(project_dir: Path, kernel: int) -> Path:
    return project_dir / run_name(kernel) / "weights" / "best.pt"


def logged_epoch_count(project_dir: Path, kernel: int) -> int:
    results_csv = project_dir / run_name(kernel) / "results.csv"
    if not results_csv.exists():
        return 0
    with results_csv.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def train_variant(args: argparse.Namespace, cfg_path: Path, kernel: int, project_dir: Path) -> Path:
    from ultralytics import YOLO

    name = run_name(kernel)
    checkpoint = best_checkpoint(project_dir, kernel)
    if checkpoint.exists() and not args.force:
        epochs_logged = logged_epoch_count(project_dir, kernel)
        if args.epochs <= 1 or epochs_logged > 1:
            print(f"[skip] {name}: found existing {checkpoint} with {epochs_logged} logged epoch(s)")
            return checkpoint
        print(f"[rerun] {name}: existing checkpoint has only {epochs_logged} logged epoch; retraining")
    if args.validate_only:
        raise FileNotFoundError(f"--validate-only requested, but checkpoint is missing: {checkpoint}")

    print(f"\n[train] {name}")
    model = YOLO(str(cfg_path))
    train_args = TRAINING_CONFIG.copy()
    train_args.update(
        {
            "data": str(args.data),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "project": str(project_dir),
            "name": name,
            "pretrained": False,
        }
    )
    if args.epochs < train_args["close_mosaic"]:
        train_args["close_mosaic"] = max(1, args.epochs // 10)
    model.train(**train_args)
    return checkpoint


def model_complexity(model, imgsz: int) -> tuple[float, float]:
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params

        params_m = float(get_num_params(model.model)) / 1e6
        gflops = float(get_flops(model.model, imgsz=imgsz))
        return params_m, gflops
    except Exception:
        params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
        return params_m, 0.0


def metric_value(obj, *names: str) -> float:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return float(value)
    return 0.0


def validate_checkpoint(args: argparse.Namespace, checkpoint: Path, kernel: int, project_dir: Path) -> dict[str, object]:
    from ultralytics import YOLO

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for k={kernel}: {checkpoint}")

    print(f"[val] {run_name(kernel)}")
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="val",
        project=str(project_dir),
        name=f"{run_name(kernel)}_val",
        exist_ok=True,
        plots=False,
    )
    params_m, gflops = model_complexity(model, args.imgsz)
    box = getattr(metrics, "box", metrics)
    return {
        "variant": run_name(kernel),
        "sgcb_kernel": kernel,
        "input_size": args.imgsz,
        "batch": args.batch,
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)) if checkpoint.is_relative_to(REPO_ROOT) else str(checkpoint),
        "params_M": round(params_m, 6),
        "GFLOPs": round(gflops, 6),
        "precision": round(metric_value(box, "mp"), 6),
        "recall": round(metric_value(box, "mr"), 6),
        "mAP50": round(metric_value(box, "map50"), 6),
        "mAP50_95": round(metric_value(box, "map"), 6),
    }


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "sgcb_kernel",
        "input_size",
        "batch",
        "checkpoint",
        "params_M",
        "GFLOPs",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved summary: {output_path}")


def main() -> None:
    args = parse_args()
    project_dir = result_project(args)
    summary_path = args.summary or project_dir / "sgcb_kernel_ablation_summary.csv"

    generated = []
    for kernel in args.kernels:
        cfg_path = generate_kernel_config(args.base_cfg, args.config_dir, kernel)
        generated.append((kernel, cfg_path))

    print("\nSGCB kernel ablation plan")
    print(f"Base config: {args.base_cfg}")
    print(f"Data:        {args.data}")
    print(f"Output:      {project_dir}")
    for kernel, cfg_path in generated:
        print(f"  k={kernel}: {cfg_path}")

    if args.dry_run:
        return

    rows = []
    for kernel, cfg_path in generated:
        checkpoint = train_variant(args, cfg_path, kernel, project_dir)
        rows.append(validate_checkpoint(args, checkpoint, kernel, project_dir))
    write_summary(rows, summary_path)


if __name__ == "__main__":
    main()
