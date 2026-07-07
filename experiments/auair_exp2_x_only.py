from __future__ import annotations

import argparse
from pathlib import Path

from auair_common import (
    BASE_CONFIG,
    REPO_ROOT,
    create_scaled_ultralite_yaml,
    print_multi_comparison,
    save_comparison_csv,
    train_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="AU-AIR Experiment 2: UltraLite-X only")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default="Comparison_AU-AIR_Exp2")
    parser.add_argument("--patience", type=int, default=BASE_CONFIG["patience"])
    parser.add_argument("--lr0", type=float, default=BASE_CONFIG["lr0"])
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "AU-AIR" / "yolo-auair8" / "dataset.yaml")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Missing AU-AIR dataset YAML: {args.data}")

    BASE_CONFIG["patience"] = args.patience
    BASE_CONFIG["lr0"] = args.lr0

    suffix_epochs = 1 if args.quick_test else args.epochs
    suffix = f"ep{suffix_epochs}_bs{args.batch}_img{args.imgsz}_pat{BASE_CONFIG['patience']}"
    project_name = f"{args.project}_{suffix}"

    ultralite_x_path = create_scaled_ultralite_yaml("x")
    try:
        row = train_model(
            model_path=ultralite_x_path,
            model_name="ultralite_x_ours",
            data_yaml=args.data,
            epochs=args.epochs,
            device=args.device,
            batch=args.batch,
            project_name=project_name,
            imgsz=args.imgsz,
            quick_test=args.quick_test,
        )
    finally:
        if ultralite_x_path.exists():
            ultralite_x_path.unlink()

    print_multi_comparison([row])
    save_comparison_csv([row], project_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
