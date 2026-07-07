"""Validate AU-AIR Exp1 best checkpoints on the AU-AIR validation split."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics-config"))

ULTRALYTICS_PATH = REPO_ROOT / "third_party" / "ultralytics"
if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


AU_AIR_EXP1_MODELS = [
    {
        "model_name": "yolov8n_baseline",
        "input_size": 384,
        "weight_path": "Comparison_AU-AIR_Exp1_ep300_bs30_img384_pat10/yolov8n_baseline/weights/best.pt",
    },
    {
        "model_name": "ultralite_nano_ours",
        "input_size": 384,
        "weight_path": "Comparison_AU-AIR_Exp1_ep300_bs30_img384_pat10/yolov8n_local_attn_ultra/weights/best.pt",
    },
    {
        "model_name": "yolo11n_baseline",
        "input_size": 384,
        "weight_path": "Comparison_AU-AIR_Exp1_ep300_bs30_img384_pat10/yolo11n_baseline/weights/best.pt",
    },
]


def as_float(value: object) -> float:
    return float(value) if value is not None else 0.0


def validate_checkpoint(entry: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    weight_path = args.run_root / str(entry["weight_path"])
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weight_path}")

    imgsz = int(entry["input_size"])
    model = YOLO(str(weight_path))
    params = int(get_num_params(model.model))
    gflops = float(get_flops(model.model, imgsz=imgsz))

    metrics = model.val(
        data=str(args.data),
        split="val",
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=f"{entry['model_name']}_img{imgsz}",
        exist_ok=True,
        plots=args.plots,
        save_json=False,
        verbose=args.verbose,
    )

    box = metrics.box
    return {
        "model_name": entry["model_name"],
        "input_size": imgsz,
        "checkpoint": str(entry["weight_path"]),
        "total_params": params,
        "params_M": round(params / 1_000_000, 6),
        "GFLOPs": round(gflops, 6),
        "precision": as_float(getattr(box, "mp", 0.0)),
        "recall": as_float(getattr(box, "mr", 0.0)),
        "mAP50": as_float(getattr(box, "map50", 0.0)),
        "mAP50_95": as_float(getattr(box, "map", 0.0)),
        "selection_rule": "best_pt_validation",
        "data": str(args.data),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "AU-AIR" / "yolo-auair8" / "dataset.yaml")
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "Comparison_AU-AIR_Exp1_ep300_bs30_img384_pat10"
        / "auair_exp1_bestpt_val_summary.csv",
    )
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "results" / "auair_exp1_bestpt_val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=60)
    parser.add_argument("--models", nargs="*", default=None, help="Optional model_name filter for smoke tests.")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Missing data yaml: {args.data}")

    selected_models = set(args.models) if args.models else None
    rows = []
    for entry in AU_AIR_EXP1_MODELS:
        if selected_models and entry["model_name"] not in selected_models:
            continue
        print(f"Validating {entry['model_name']} at imgsz={entry['input_size']}...")
        rows.append(validate_checkpoint(entry, args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "input_size",
        "checkpoint",
        "total_params",
        "params_M",
        "GFLOPs",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "selection_rule",
        "data",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved validation summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
