"""Run the Small-scale component ablation for Ultra-Lite YOLOv8.

This mirrors the Nano ablation but uses the practical Small-scale setting:
YOLOv8s baseline, P2-only, SGCB-only, and full P2+SGCB UltraLite-S.
Existing Exp2 checkpoints can be reused for the baseline and full model so
only the missing P2-only and SGCB-only variants need to be trained.
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
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics-config"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


TRAINING_CONFIG = {
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
    "patience": 10,
}


def find_source_cfg(filename: str) -> Path:
    candidates = [
        REPO_ROOT / "configs" / "generated" / "small_ablation" / filename,
        REPO_ROOT / "configs" / filename,
        REPO_ROOT / "third_party" / "ultralytics" / "ultralytics" / "cfg" / "models" / "v8" / filename,
        REPO_ROOT / "ultralytics" / "cfg" / "models" / "v8" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find {filename} in candidate paths: {[str(c) for c in candidates]}")


def generated_dir() -> Path:
    path = REPO_ROOT / "configs" / "generated" / "small_ablation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_scaled_yaml(source: Path, destination: Path, scale: str = "s") -> Path:
    if scale != "s":
        raise ValueError("This runner is intentionally scoped to the Small ('s') scale.")
    text = source.read_text(encoding="utf-8")
    text = force_visdrone_small_scale(text)
    with destination.open("w", encoding="utf-8") as file:
        file.write(text)
    return destination


def force_visdrone_small_scale(text: str) -> str:
    text = re.sub(r"(?m)^nc:\s*\d+.*$", "nc: 10", text, count=1)
    text = re.sub(
        r"(?m)^scales:[^\n]*\n(?:^[ \t]{2}.*\n)+",
        "scales:\n  n: [0.33, 0.50, 1024]\nscale: n\n",
        text,
        count=1,
    )
    return text


def build_variant_configs() -> dict[str, dict[str, object]]:
    out = generated_dir()
    p2_only = out / "small_ablation_p2_only.yaml"
    if not p2_only.exists():
        p2_src = find_source_cfg("yolov8-p2.yaml")
        write_scaled_yaml(p2_src, p2_only)

    full = out / "small_ablation_ultralite_full.yaml"
    if not full.exists():
        full_src = find_source_cfg("yolov8n-ultraliteattn-local.yaml")
        write_scaled_yaml(full_src, full)

    sgcb_only = out / "small_ablation_sgcb_only.yaml"
    if not sgcb_only.exists():
        sgcb_only_source = find_source_cfg("yolov8n-sgcb-only.yaml")
        write_sgcb_only_small_yaml(sgcb_only_source, sgcb_only)

    return {
        "A_Baseline_S": {
            "cfg": "yolov8s.yaml",
            "description": "YOLOv8s baseline",
            "existing_checkpoint": REPO_ROOT
            / "Comparison_Exp2_ep300_bs30_img640_pat10"
            / "yolov8s_baseline"
            / "weights"
            / "best.pt",
        },
        "B_P2_Only_S": {
            "cfg": str(p2_only),
            "description": "YOLOv8s with P2 high-resolution head only",
            "existing_checkpoint": None,
        },
        "C_SGCB_Only_S": {
            "cfg": str(sgcb_only),
            "description": "YOLOv8s with SGCB selective replacement, no P2 head",
            "existing_checkpoint": None,
        },
        "D_UltraLite_S_Full": {
            "cfg": str(full),
            "description": "UltraLite-S full design: P2 head plus selective SGCB",
            "existing_checkpoint": REPO_ROOT
            / "Comparison_Exp2_ep300_bs30_img640_pat10"
            / "ultralite_s_ours"
            / "weights"
            / "best.pt",
        },
    }


def write_sgcb_only_small_yaml(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Missing SGCB-only source config: {source}")
    text = force_visdrone_small_scale(source.read_text(encoding="utf-8"))
    with destination.open("w", encoding="utf-8") as file:
        file.write(text)
    return destination


def metric_value(box: object, name: str) -> float:
    return float(getattr(box, name, 0.0) or 0.0)


def model_stats(model: YOLO, imgsz: int) -> tuple[int, float, float]:
    params = int(get_num_params(model.model))
    gflops = float(get_flops(model.model, imgsz=imgsz))
    return params, round(params / 1_000_000, 6), round(gflops, 6)


def validate_model(model: YOLO, args: argparse.Namespace, name: str) -> dict[str, float]:
    metrics = model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.val_batch,
        device=args.device,
        project=str(args.output_dir / "validation"),
        name=name,
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=args.verbose,
    )
    box = metrics.box
    return {
        "precision": metric_value(box, "mp"),
        "recall": metric_value(box, "mr"),
        "mAP50": metric_value(box, "map50"),
        "mAP50_95": metric_value(box, "map"),
    }


def train_variant(cfg: str, name: str, args: argparse.Namespace) -> Path:
    model = YOLO(cfg)
    training_config = TRAINING_CONFIG.copy()
    training_config.update(
        {
            "data": str(args.data),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "project": str(args.output_dir),
            "name": name,
            "pretrained": False,
        }
    )
    if args.epochs < training_config["close_mosaic"]:
        training_config["close_mosaic"] = max(1, args.epochs // 10)
    if args.no_cache:
        training_config["cache"] = False

    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_key_value_config(run_dir / "training_config.yaml", training_config)

    model.train(**training_config)
    checkpoint = run_dir / "weights" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training finished but best checkpoint is missing: {checkpoint}")
    return checkpoint


def write_key_value_config(path: Path, config: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for key, value in config.items():
            file.write(f"{key}: {value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "configs" / "VisDrone.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=30)
    parser.add_argument("--val-batch", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "results" / "ablation_small")
    parser.add_argument("--models", nargs="*", default=None, help="Variant names to run/validate.")
    parser.add_argument(
        "--only-missing",
        "--only-left-out",
        dest="only_missing",
        action="store_true",
        help="Run only the Small ablation variants without reused checkpoints: B_P2_Only_S and C_SGCB_Only_S.",
    )
    parser.add_argument("--list-models", action="store_true", help="Print available variant names and exit.")
    parser.add_argument("--train-all", action="store_true", help="Retrain baseline and full model too.")
    parser.add_argument("--dry-run", action="store_true", help="Build configs and print model stats only.")
    parser.add_argument("--no-cache", action="store_true", help="Disable Ultralytics dataset caching.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    suffix = f"ep{args.epochs}_bs{args.batch}_img{args.imgsz}_pat{TRAINING_CONFIG['patience']}"
    args.output_dir = REPO_ROOT / f"{args.project}_{suffix}"
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = build_variant_configs()
    if args.list_models:
        print("Available Small ablation variants:")
        for name, spec in variants.items():
            checkpoint = spec["existing_checkpoint"]
            reuse = bool(checkpoint and Path(checkpoint).exists())
            suffix = "reuses existing checkpoint" if reuse else "requires training"
            print(f"  {name}: {spec['description']} ({suffix})")
        return

    if args.only_missing and args.models:
        raise ValueError("Use either --only-missing or --models, not both.")

    if args.only_missing:
        selected = {
            name
            for name, spec in variants.items()
            if not spec["existing_checkpoint"]
        }
    else:
        selected = set(args.models or variants.keys())
    unknown = selected.difference(variants)
    if unknown:
        raise KeyError(f"Unknown variant(s): {', '.join(sorted(unknown))}")

    rows: list[dict[str, object]] = []
    for name, spec in variants.items():
        if name not in selected:
            continue

        cfg = str(spec["cfg"])
        checkpoint = spec["existing_checkpoint"]
        output_checkpoint = args.output_dir / name / "weights" / "best.pt"
        mode = "trained"
        if output_checkpoint.exists() and not args.train_all:
            model_source = str(output_checkpoint)
            mode = "reused_trained_checkpoint"
        elif checkpoint and Path(checkpoint).exists() and not args.train_all:
            model_source = str(checkpoint)
            mode = "reused_existing_checkpoint"
        else:
            model_source = cfg

        print("\n" + "=" * 80)
        print(f"{name}: {spec['description']}")
        print(f"source: {model_source}")
        print("=" * 80)

        model = YOLO(model_source)
        total_params, params_m, gflops = model_stats(model, args.imgsz)
        if args.dry_run:
            metrics = {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50_95": 0.0}
            final_checkpoint = model_source
        else:
            if mode == "trained":
                final_checkpoint = train_variant(cfg, name, args)
                model = YOLO(str(final_checkpoint))
                total_params, params_m, gflops = model_stats(model, args.imgsz)
            else:
                final_checkpoint = model_source
            metrics = validate_model(model, args, name)

        rows.append(
            {
                "variant": name,
                "description": spec["description"],
                "mode": mode if not args.dry_run else "dry_run",
                "checkpoint": str(final_checkpoint),
                "total_params": total_params,
                "params_M": params_m,
                "GFLOPs": gflops,
                **metrics,
                "epochs": args.epochs,
                "batch": args.batch,
                "imgsz": args.imgsz,
            }
        )

    summary_path = args.output_dir / "small_ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSmall ablation summary")
    print(f"Saved: {summary_path}")
    for row in rows:
        print(
            f"{row['variant']}: params={row['params_M']:.2f}M, "
            f"GFLOPs={row['GFLOPs']:.2f}, "
            f"mAP50={float(row['mAP50']) * 100:.1f}, "
            f"mAP50-95={float(row['mAP50_95']) * 100:.1f}, "
            f"mode={row['mode']}"
        )


if __name__ == "__main__":
    main()
