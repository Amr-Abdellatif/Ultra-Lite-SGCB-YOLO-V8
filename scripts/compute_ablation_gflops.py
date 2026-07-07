from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def add_local_ultralytics(repo_root: Path) -> None:
    local_ultralytics = repo_root / "third_party" / "ultralytics"
    if local_ultralytics.exists():
        sys.path.insert(0, str(local_ultralytics))


def model_gflops(weight_path: Path, imgsz: int) -> tuple[float, float]:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    model = YOLO(str(weight_path))
    params_m = float(get_num_params(model.model)) / 1e6
    gflops = float(get_flops(model.model, imgsz=imgsz))
    return params_m, gflops


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute GFLOPs for completed ablation runs.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/ablation_full_ep300_bs30_img384_pat10"),
        help="Ablation result directory containing variant subfolders.",
    )
    parser.add_argument("--imgsz", type=int, default=384, help="Input image size used for GFLOPs.")
    parser.add_argument("--weights", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    add_local_ultralytics(repo_root)

    try:
        import thop  # noqa: F401
    except ImportError:
        print("ERROR: thop is missing. Install it with: pip install ultralytics-thop", file=sys.stderr)
        return 2

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "comparison_summary.csv"
    if not summary_path.exists():
        print(f"ERROR: summary CSV not found: {summary_path}", file=sys.stderr)
        return 1

    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        variant = row["variant"]
        weight_path = run_dir / variant / "weights" / args.weights
        if not weight_path.exists():
            print(f"ERROR: missing weights for {variant}: {weight_path}", file=sys.stderr)
            return 1
        params_m, gflops = model_gflops(weight_path, args.imgsz)
        row["params_M"] = f"{params_m:.6f}"
        row["GFLOPs"] = f"{gflops:.6f}"
        print(f"{variant}: params={params_m:.6f}M, GFLOPs={gflops:.6f}")

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
