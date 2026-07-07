from __future__ import annotations

import csv
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics-config"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ULTRALYTICS_PATH = REPO_ROOT / "third_party" / "ultralytics"
if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


BASE_CONFIG = {
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


SUMMARY_FIELDS = [
    "model_name",
    "dataset",
    "input_size",
    "epochs_trained",
    "total_params",
    "params_M",
    "GFLOPs",
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
    "selected_epoch",
    "selection_rule",
    "checkpoint",
    "training_config",
    "timestamp",
]


def model_cfg_dir() -> Path:
    return REPO_ROOT / "third_party" / "ultralytics" / "ultralytics" / "cfg" / "models"


def create_scaled_ultralite_yaml(scale: str) -> Path:
    source = model_cfg_dir() / "v8" / "yolov8n-ultraliteattn-local.yaml"
    with source.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if scale not in data.get("scales", {}):
        raise KeyError(f"Scale '{scale}' is not defined in {source}")
    data["scales"] = {"n": data["scales"][scale]}

    fd, path = tempfile.mkstemp(prefix=f"auair_ultralite_{scale}_", suffix=".yaml", dir=REPO_ROOT)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return Path(path)


def row_float(row: dict[str, str], candidates: list[str]) -> float:
    normalized = {key.strip().lower().replace(" ", ""): value for key, value in row.items()}
    for candidate in candidates:
        value = normalized.get(candidate.strip().lower().replace(" ", ""))
        if value not in (None, ""):
            return float(value)
    return 0.0


def parse_best_metrics(results_csv: Path) -> dict[str, object]:
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "selected_epoch": "",
        "selection_rule": "best_ultralytics_fitness",
    }
    if not results_csv.exists():
        return metrics

    with results_csv.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return metrics

    def fitness(row: dict[str, str]) -> float:
        map50 = row_float(row, ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"])
        map50_95 = row_float(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50_95"])
        return 0.1 * map50 + 0.9 * map50_95

    best_row = max(rows, key=fitness)
    metrics.update(
        {
            "precision": row_float(best_row, ["metrics/precision(B)", "metrics/precision", "precision"]),
            "recall": row_float(best_row, ["metrics/recall(B)", "metrics/recall", "recall"]),
            "mAP50": row_float(best_row, ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"]),
            "mAP50_95": row_float(best_row, ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50_95"]),
            "selected_epoch": best_row.get("epoch", ""),
        }
    )
    return metrics


def train_model(
    model_path: str | Path,
    model_name: str,
    data_yaml: str | Path,
    epochs: int,
    device: str,
    batch: int,
    project_name: str | Path,
    imgsz: int,
    quick_test: bool = False,
    resume: bool = False,
) -> dict[str, object]:
    actual_epochs = 1 if quick_test else epochs
    project_path = Path(project_name)
    run_dir = project_path / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"TRAINING AU-AIR: {model_name}")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {actual_epochs}")
    print(f"Batch: {batch}")
    print(f"Image size: {imgsz}")

    model = YOLO(str(model_path))
    config = BASE_CONFIG.copy()
    config.update(
        {
            "data": str(data_yaml),
            "epochs": actual_epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device,
            "project": str(project_path),
            "name": model_name,
            "pretrained": False,
            "resume": resume,
        }
    )
    if actual_epochs < config["close_mosaic"]:
        config["close_mosaic"] = max(1, actual_epochs // 10)
    if quick_test:
        config.update({"plots": False, "save_period": 1})

    config_path = run_dir / "training_config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    model.train(**config)

    params = int(get_num_params(model.model))
    try:
        gflops = float(get_flops(model.model, imgsz=imgsz))
    except Exception:
        gflops = 0.0

    metrics = parse_best_metrics(run_dir / "results.csv")
    return {
        "model_name": model_name,
        "dataset": str(data_yaml),
        "input_size": imgsz,
        "epochs_trained": actual_epochs,
        "total_params": params,
        "params_M": round(params / 1_000_000, 6),
        "GFLOPs": round(gflops, 6),
        **metrics,
        "checkpoint": str(run_dir / "weights" / "best.pt"),
        "training_config": str(config_path),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_comparison_csv(rows: list[dict[str, object]], project_name: str | Path, prefix: str = "comparison_summary") -> Path:
    project_path = Path(project_name)
    project_path.mkdir(parents=True, exist_ok=True)
    csv_path = project_path / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to {csv_path}")
    return csv_path


def print_multi_comparison(rows: list[dict[str, object]]) -> None:
    print("\n" + "=" * 100)
    print("AU-AIR COMPARISON SUMMARY")
    print("=" * 100)
    header = f"{'Metric':<18}" + "".join(f" {row['model_name']:>22}" for row in rows)
    print(header)
    print("-" * len(header))
    for label, key in [
        ("Params", "params_M"),
        ("GFLOPs", "GFLOPs"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("mAP50", "mAP50"),
        ("mAP50-95", "mAP50_95"),
    ]:
        print(f"{label:<18}" + "".join(f" {row.get(key, 0):>22}" for row in rows))
    print("=" * 100)
