"""Validate AU-AIR Exp1 and Exp2 best checkpoints on the AU-AIR validation or test split."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics-config"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ULTRALYTICS_PATH = REPO_ROOT / "third_party" / "ultralytics"
if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


ALL_AU_AIR_MODELS = [
    # Exp 1 (Nano @ 384)
    {
        "group": "Exp1_Nano_384",
        "model_name": "yolov8n_baseline",
        "input_size": 384,
        "candidate_paths": [
            "downloaded_results/Comparison_AU-AIR_Exp1_ep300_bs30_img384_pat10/yolov8n_baseline/weights/best.pt",
            "Comparison_AU-AIR_Exp1_ep300_bs30_img384_pat10/yolov8n_baseline/weights/best.pt",
        ],
    },
    {
        "group": "Exp1_Nano_384",
        "model_name": "yolo11n_baseline",
        "input_size": 384,
        "candidate_paths": [
            "downloaded_results/Comparison_AU-AIR_Exp1_ep300_bs30_img384_pat10/yolo11n_baseline/weights/best.pt",
            "Comparison_AU-AIR_Exp1_ep300_bs30_img384_pat10/yolo11n_baseline/weights/best.pt",
        ],
    },
    {
        "group": "Exp1_Nano_384",
        "model_name": "ultralite_nano_ours",
        "input_size": 384,
        "candidate_paths": [
            "downloaded_results/Comparison_AU-AIR_Exp1_ep300_bs30_img384_pat10/yolov8n_local_attn_ultra/weights/best.pt",
            "Comparison_AU-AIR_Exp1_ep300_bs30_img384_pat10/yolov8n_local_attn_ultra/weights/best.pt",
        ],
    },
    # Exp 2 (Small & X @ 640)
    {
        "group": "Exp2_Scaling_640",
        "model_name": "yolov8s_baseline",
        "input_size": 640,
        "candidate_paths": [
            "remote_results/auair_exp2_combined/Comparison_AU-AIR_Exp2_ep300_bs30_img640_pat10/yolov8s_baseline/weights/best.pt",
            "Comparison_AU-AIR_Exp2_ep300_bs30_img640_pat10/yolov8s_baseline/weights/best.pt",
        ],
    },
    {
        "group": "Exp2_Scaling_640",
        "model_name": "yolo11s_baseline",
        "input_size": 640,
        "candidate_paths": [
            "remote_results/auair_exp2_combined/Comparison_AU-AIR_Exp2_ep300_bs30_img640_pat10/yolo11s_baseline/weights/best.pt",
            "Comparison_AU-AIR_Exp2_ep300_bs30_img640_pat10/yolo11s_baseline/weights/best.pt",
        ],
    },
    {
        "group": "Exp2_Scaling_640",
        "model_name": "ultralite_s_ours",
        "input_size": 640,
        "candidate_paths": [
            "remote_results/auair_exp2_combined/Comparison_AU-AIR_Exp2_ep300_bs30_img640_pat10/ultralite_s_ours/weights/best.pt",
            "Comparison_AU-AIR_Exp2_ep300_bs30_img640_pat10/ultralite_s_ours/weights/best.pt",
        ],
    },
    {
        "group": "Exp2_Scaling_640",
        "model_name": "ultralite_x_ours",
        "input_size": 640,
        "candidate_paths": [
            "remote_results/auair_exp2_combined/Comparison_AU-AIR_Exp2_ep300_bs30_img640_pat10/ultralite_x_ours/weights/best.pt",
            "Comparison_AU-AIR_Exp2_ep300_bs30_img640_pat10/ultralite_x_ours/weights/best.pt",
        ],
    },
]


def as_float(value: object) -> float:
    return float(value) if value is not None else 0.0


def find_weight_path(entry: dict[str, object], run_root: Path) -> Path:
    for rel in entry["candidate_paths"]:
        p = run_root / str(rel)
        if p.exists():
            return p
    raise FileNotFoundError(f"Checkpoint not found for {entry['model_name']} in candidate paths: {entry['candidate_paths']}")


def validate_checkpoint(entry: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    weight_path = find_weight_path(entry, args.run_root)
    imgsz = int(entry["input_size"])
    model = YOLO(str(weight_path))
    params = int(get_num_params(model.model))
    gflops = float(get_flops(model.model, imgsz=imgsz))

    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=f"{entry['model_name']}_img{imgsz}_{args.split}",
        exist_ok=True,
        plots=args.plots,
        save_json=False,
        verbose=args.verbose,
    )

    box = metrics.box
    return {
        "group": entry["group"],
        "model_name": entry["model_name"],
        "input_size": imgsz,
        "checkpoint": str(weight_path.relative_to(args.run_root)),
        "split": args.split,
        "total_params": params,
        "params_M": round(params / 1_000_000, 6),
        "GFLOPs": round(gflops, 6),
        "precision": as_float(getattr(box, "mp", 0.0)),
        "recall": as_float(getattr(box, "mr", 0.0)),
        "mAP50": as_float(getattr(box, "map50", 0.0)),
        "mAP50_95": as_float(getattr(box, "map", 0.0)),
        "data": str(args.data),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "AU-AIR" / "yolo-auair8" / "dataset.yaml")
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--split", choices=["val", "test", "train"], default="test", help="Dataset split to evaluate (default: test)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "results" / "auair_eval")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--group", choices=["all", "exp1", "exp2"], default="all", help="Model group to evaluate")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model_name filter.")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = REPO_ROOT / "results" / f"auair_{args.split}_summary.csv"

    if not args.data.exists():
        raise FileNotFoundError(f"Missing data yaml: {args.data}")

    selected_models = set(args.models) if args.models else None
    rows = []
    for entry in ALL_AU_AIR_MODELS:
        if args.group == "exp1" and "Exp1" not in str(entry["group"]):
            continue
        if args.group == "exp2" and "Exp2" not in str(entry["group"]):
            continue
        if selected_models and entry["model_name"] not in selected_models:
            continue

        print(f"\n[{entry['group']}] Validating {entry['model_name']} on '{args.split}' at imgsz={entry['input_size']}...")
        rows.append(validate_checkpoint(entry, args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "model_name",
        "input_size",
        "checkpoint",
        "split",
        "total_params",
        "params_M",
        "GFLOPs",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "data",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n================================================================================")
    print(f"AU-AIR {args.split.upper()} RESULTS SUMMARY")
    print(f"================================================================================")
    for r in rows:
        print(f"[{r['group']}] {r['model_name']:20s} | Params: {r['params_M']}M | GFLOPs: {r['GFLOPs']:6.2f} | P: {r['precision']*100:4.1f}% | R: {r['recall']*100:4.1f}% | mAP50: {r['mAP50']*100:4.1f}% | mAP50-95: {r['mAP50_95']*100:4.1f}%")
    print(f"Saved results to: {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
