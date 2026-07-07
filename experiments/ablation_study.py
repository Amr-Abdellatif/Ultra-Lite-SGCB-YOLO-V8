import argparse
import csv
import os
import sys
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - YAML export is a convenience.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = REPO_ROOT / "third_party" / "ultralytics"
if LOCAL_ULTRALYTICS.exists():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ablation protocol mirrors the controlled Exp1 training setup so that the
# architecture variants differ only in model definition.
ABLATION_CONFIG = {
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


def clean_key(key):
    return key.strip().lower().replace(" ", "").replace("_", "")


def get_row_value(row, names):
    normalized = {clean_key(k): v for k, v in row.items()}
    for name in names:
        value = normalized.get(clean_key(name))
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                return 0.0
    return 0.0


def extract_metrics(run_dir, train_results=None):
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "mAP50": 0.0,
        "mAP50-95": 0.0,
    }
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            metrics["precision"] = get_row_value(last, ["metrics/precision(B)", "metrics/precision"])
            metrics["recall"] = get_row_value(last, ["metrics/recall(B)", "metrics/recall"])
            metrics["mAP50"] = get_row_value(last, ["metrics/mAP50(B)", "metrics/mAP50"])
            metrics["mAP50-95"] = get_row_value(last, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
            return metrics

    result_dict = getattr(train_results, "results_dict", {}) or {}
    fallback_keys = {
        "precision": ["metrics/precision(B)", "metrics/precision"],
        "recall": ["metrics/recall(B)", "metrics/recall"],
        "mAP50": ["metrics/mAP50(B)", "metrics/mAP50", "metrics/m_ap50"],
        "mAP50-95": ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/m_ap50_95"],
    }
    for metric, keys in fallback_keys.items():
        for key in keys:
            if key in result_dict:
                metrics[metric] = float(result_dict[key])
                break
    return metrics


def get_model_complexity(model, imgsz):
    try:
        info = model.info(imgsz=imgsz, verbose=False)
        params = float(info[1]) / 1e6
        gflops = float(info[3]) if len(info) > 3 else 0.0
        return params, gflops
    except Exception:
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
        return params, 0.0


def save_training_config(run_dir, training_config):
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "training_config.yaml"
    if yaml is None:
        with config_path.with_suffix(".txt").open("w", encoding="utf-8") as f:
            for key, value in sorted(training_config.items()):
                f.write(f"{key}: {value}\n")
        return
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(training_config, f, sort_keys=False)


def run_ablation():
    parser = argparse.ArgumentParser(description="Ultra-Lite YOLO ablation study")
    parser.add_argument("--data", type=str, default="configs/VisDrone.yaml", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=30, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=384, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id")
    parser.add_argument("--project", type=str, default="results/ablation_full", help="Base output directory")
    args = parser.parse_args()

    cfg_dir = REPO_ROOT / "third_party" / "ultralytics" / "ultralytics" / "cfg" / "models" / "v8"
    variants = [
        {
            "name": "A_Baseline",
            "cfg": str(cfg_dir / "yolov8n.yaml"),
            "desc": "Standard YOLOv8n",
        },
        {
            "name": "B_Baseline_P2",
            "cfg": str(cfg_dir / "yolov8-p2.yaml"),
            "desc": "YOLOv8n with P2 high-resolution head",
        },
        {
            "name": "C_Baseline_SGCB",
            "cfg": str(REPO_ROOT / "configs" / "yolov8n-sgcb-only.yaml"),
            "desc": "YOLOv8n with SGCB selective replacement, no P2 head",
        },
        {
            "name": "D_UltraLite_Full",
            "cfg": str(cfg_dir / "yolov8n-ultraliteattn-local.yaml"),
            "desc": "Full UltraLite design: SGCB plus P2 head",
        },
    ]

    create_sgcb_only_config(REPO_ROOT)

    suffix = f"ep{args.epochs}_bs{args.batch}_img{args.imgsz}_pat{ABLATION_CONFIG['patience']}"
    output_dir = REPO_ROOT / f"{args.project}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("STARTING ABLATION STUDY")
    print(f"Results will be saved to: {output_dir}")
    print(f"Protocol: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, patience={ABLATION_CONFIG['patience']}")
    print("=" * 80)

    results = []
    for var in variants:
        print(f"\nRunning {var['name']} ({var['desc']})")
        model = YOLO(var["cfg"])

        training_config = ABLATION_CONFIG.copy()
        training_config.update(
            {
                "data": args.data,
                "epochs": args.epochs,
                "batch": args.batch,
                "imgsz": args.imgsz,
                "device": args.device,
                "project": str(output_dir),
                "name": var["name"],
                "pretrained": False,
            }
        )
        if args.epochs < ABLATION_CONFIG["close_mosaic"]:
            training_config["close_mosaic"] = max(1, args.epochs // 10)

        run_dir = output_dir / var["name"]
        save_training_config(run_dir, training_config)
        train_results = model.train(**training_config)

        params, gflops = get_model_complexity(model, args.imgsz)
        metrics = extract_metrics(run_dir, train_results)
        results.append(
            {
                "variant": var["name"],
                "description": var["desc"],
                "params_M": params,
                "GFLOPs": gflops,
                **metrics,
            }
        )

    summary_path = output_dir / "comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "description", "params_M", "GFLOPs", "precision", "recall", "mAP50", "mAP50-95"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<20} | {'Params (M)':<10} | {'GFLOPs':<8} | {'mAP50':<8} | {'mAP50-95':<10}")
    print("-" * 70)
    for result in results:
        print(
            f"{result['variant']:<20} | "
            f"{result['params_M']:<10.2f} | "
            f"{result['GFLOPs']:<8.2f} | "
            f"{result['mAP50']:<8.4f} | "
            f"{result['mAP50-95']:<10.4f}"
        )
    print(f"\nSaved summary: {summary_path}")


def create_sgcb_only_config(repo_root):
    """Create a config with SGCB selective replacement but no P2 head."""
    cfg_path = repo_root / "configs" / "yolov8n-sgcb-only.yaml"
    if cfg_path.exists():
        return

    content = """# YOLOv8 with SGCB Only (No P2 Head)
nc: 10
scales:
  n: [0.33, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]             # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]            # 1-P2/4
  - [-1, 3, C2f, [128, True]]             # 2
  - [-1, 1, Conv, [256, 3, 2]]            # 3-P3/8
  - [-1, 6, SGCB, [256, True, 1, 0.5, 7]] # 4
  - [-1, 1, Conv, [512, 3, 2]]            # 5-P4/16
  - [-1, 6, SGCB, [512, True, 1, 0.5, 7]] # 6
  - [-1, 1, Conv, [1024, 3, 2]]           # 7-P5/32
  - [-1, 3, C2f, [1024, True]]            # 8
  - [-1, 1, SPPF, [1024, 5]]              # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, nearest]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, SGCB, [512]]                  # 12

  - [-1, 1, nn.Upsample, [None, 2, nearest]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, SGCB, [256]]                  # 15

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, SGCB, [512]]                  # 18

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]]                  # 21

  - [[15, 18, 21], 1, Detect, [nc]]
"""
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_ablation()
