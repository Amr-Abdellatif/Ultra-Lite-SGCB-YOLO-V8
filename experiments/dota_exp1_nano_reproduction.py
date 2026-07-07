from __future__ import annotations

import argparse
from pathlib import Path

from dota_common import BASE_CONFIG, REPO_ROOT, model_cfg_dir, print_multi_comparison, save_comparison_csv, train_model


def main() -> int:
    parser = argparse.ArgumentParser(description="DOTA-HBB Experiment 1: Nano controlled comparison")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--project", type=str, default="Comparison_DOTA_Exp1")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--ultralite-only", action="store_true")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "DOTA" / "yolo-dota-v1.5-hbb" / "dataset.yaml")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Missing DOTA-HBB dataset YAML: {args.data}")

    suffix_epochs = 1 if args.quick_test else args.epochs
    suffix = f"ep{suffix_epochs}_bs{args.batch}_img{args.imgsz}_pat{BASE_CONFIG['patience']}"
    project_name = f"{args.project}_{suffix}"

    v8_dir = model_cfg_dir() / "v8"
    configs = [
        ("yolov8n.yaml", "yolov8n_baseline"),
        (v8_dir / "yolov8n-ultraliteattn-local.yaml", "yolov8n_local_attn_ultra"),
        ("yolo11n.yaml", "yolo11n_baseline"),
    ]

    if args.baseline_only:
        configs = [(path, name) for path, name in configs if "ultra" not in name]
    if args.ultralite_only:
        configs = [(path, name) for path, name in configs if "ultra" in name]

    rows = []
    for model_path, model_name in configs:
        rows.append(
            train_model(
                model_path=model_path,
                model_name=model_name,
                data_yaml=args.data,
                epochs=args.epochs,
                device=args.device,
                batch=args.batch,
                project_name=project_name,
                imgsz=args.imgsz,
                quick_test=args.quick_test,
                resume=args.resume,
            )
        )

    if rows:
        print_multi_comparison(rows)
        save_comparison_csv(rows, project_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
