"""Compute object-size-stratified recall for YOLO detection checkpoints.

The script bins ground-truth boxes by their original-image pixel size, runs
existing best.pt checkpoints, and reports class-aware recall at a fixed IoU.
It is intended for reviewer-facing analysis of extremely small objects without
retraining models.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics-config"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ULTRALYTICS_PATH = REPO_ROOT / "third_party" / "ultralytics"
if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


DEFAULT_MODELS = [
    {
        "name": "yolov8n_baseline",
        "imgsz": 384,
        "weights": "Comparison_Exp1_ep300_bs30_img384_pat10/yolov8n_baseline/weights/best.pt",
    },
    {
        "name": "yolo11n_baseline",
        "imgsz": 384,
        "weights": "Comparison_Exp1_ep300_bs30_img384_pat10/yolo11n_baseline/weights/best.pt",
    },
    {
        "name": "ultralite_nano_ours",
        "imgsz": 384,
        "weights": "Comparison_Exp1_ep300_bs30_img384_pat10/yolov8n_local_attn_ultra/weights/best.pt",
    },
    {
        "name": "yolov8s_baseline",
        "imgsz": 640,
        "weights": "Comparison_Exp2_ep300_bs30_img640_pat10/yolov8s_baseline/weights/best.pt",
    },
    {
        "name": "yolo11s_baseline",
        "imgsz": 640,
        "weights": "Comparison_Exp2_ep300_bs30_img640_pat10/yolo11s_baseline/weights/best.pt",
    },
    {
        "name": "ultralite_s_ours",
        "imgsz": 640,
        "weights": "Comparison_Exp2_ep300_bs30_img640_pat10/ultralite_s_ours/weights/best.pt",
    },
    {
        "name": "ultralite_x_ours",
        "imgsz": 640,
        "weights": "Comparison_Exp2_ep300_bs30_img640_pat10/ultralite_x_ours/weights/best.pt",
    },
]


@dataclass
class GroundTruth:
    cls: int
    xyxy: tuple[float, float, float, float]
    size_px: float
    bin_label: str


@dataclass
class Prediction:
    cls: int
    conf: float
    xyxy: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "configs" / "VisDrone.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "journal_submission" / "visdrone_size_stratified_recall.csv")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001, help="Prediction confidence threshold used before matching.")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold for true-positive matching.")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA devices to reduce memory.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument("--size-metric", choices=["sqrt_area", "max_side", "min_side"], default="sqrt_area")
    parser.add_argument(
        "--bins",
        default="0,10,20,32,inf",
        help="Comma-separated bin edges in pixels. Default gives <10, 10-20, 20-32, >=32.",
    )
    parser.add_argument("--models", nargs="*", default=None, help="Optional default model-name filter.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=WEIGHTS:IMGSZ",
        help="Custom model entry. Example: ours=runs/model/weights/best.pt:640",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke tests.")
    parser.add_argument("--agnostic", action="store_true", help="Ignore class labels during matching.")
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_data_config(data_yaml: Path) -> dict:
    cfg = {}
    with data_yaml.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line or line.startswith(" ") or line.startswith("\t"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key in {"path", "train", "val", "test"}:
                cfg[key] = value
    data_root = Path(cfg.get("path", ""))
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    cfg["_root"] = data_root
    return cfg


def read_image_list(entry: object, data_root: Path, limit: int | None) -> list[Path]:
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []
    for item in entries:
        item_path = Path(str(item))
        resolved = resolve_path(item_path, data_root)
        if resolved.is_file() and resolved.suffix.lower() == ".txt":
            with resolved.open("r", encoding="utf-8") as file:
                for line in file:
                    value = line.strip()
                    if not value:
                        continue
                    path = Path(value)
                    images.append(path if path.is_absolute() else (data_root / path).resolve())
        elif resolved.is_dir():
            for suffix in IMAGE_SUFFIXES:
                images.extend(resolved.rglob(f"*{suffix}"))
        elif resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES:
            images.append(resolved)
        else:
            raise FileNotFoundError(f"Could not resolve image source: {item}")

    images = sorted(dict.fromkeys(images))
    return images[:limit] if limit else images


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def parse_bins(raw_bins: str) -> list[float]:
    edges: list[float] = []
    for item in raw_bins.split(","):
        value = item.strip().lower()
        edges.append(math.inf if value in {"inf", "infinity"} else float(value))
    if len(edges) < 2:
        raise ValueError("At least two bin edges are required.")
    if edges != sorted(edges):
        raise ValueError("Bin edges must be sorted ascending.")
    return edges


def bin_label(value: float, edges: list[float]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return edge_label(lo, hi)
    return f">={edges[-2]:g}px"


def edge_label(lo: float, hi: float) -> str:
    if math.isinf(hi):
        return f">={lo:g}px"
    if lo == 0:
        return f"<{hi:g}px"
    return f"{lo:g}-{hi:g}px"


def compute_size(width_px: float, height_px: float, metric: str) -> float:
    if metric == "sqrt_area":
        return math.sqrt(width_px * height_px)
    if metric == "max_side":
        return max(width_px, height_px)
    if metric == "min_side":
        return min(width_px, height_px)
    raise ValueError(f"Unknown size metric: {metric}")


def load_ground_truths(label_path: Path, image_width: int, image_height: int, size_metric: str, edges: list[float]) -> list[GroundTruth]:
    if not label_path.exists():
        return []

    gts: list[GroundTruth] = []
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, width, height = [float(value) for value in parts[1:5]]
            width_px = width * image_width
            height_px = height * image_height
            x1 = (xc - width / 2.0) * image_width
            y1 = (yc - height / 2.0) * image_height
            x2 = (xc + width / 2.0) * image_width
            y2 = (yc + height / 2.0) * image_height
            size_px = compute_size(width_px, height_px, size_metric)
            gts.append(GroundTruth(cls=cls, xyxy=(x1, y1, x2, y2), size_px=size_px, bin_label=bin_label(size_px, edges)))
    return gts


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def extract_predictions(result) -> list[Prediction]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy().tolist()
    classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
    confs = result.boxes.conf.cpu().numpy().tolist()
    return [Prediction(cls=cls, conf=float(conf), xyxy=tuple(map(float, box))) for cls, conf, box in zip(classes, confs, boxes)]


def match_by_bin(gts: list[GroundTruth], preds: list[Prediction], iou_threshold: float, class_agnostic: bool) -> dict[str, tuple[int, int]]:
    totals: dict[str, int] = {}
    matched: dict[str, int] = {}
    for gt in gts:
        totals[gt.bin_label] = totals.get(gt.bin_label, 0) + 1

    used_gt: set[int] = set()
    for pred in sorted(preds, key=lambda item: item.conf, reverse=True):
        best_iou = 0.0
        best_idx: int | None = None
        for idx, gt in enumerate(gts):
            if idx in used_gt:
                continue
            if not class_agnostic and pred.cls != gt.cls:
                continue
            iou = box_iou(pred.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            used_gt.add(best_idx)
            label = gts[best_idx].bin_label
            matched[label] = matched.get(label, 0) + 1

    return {label: (totals.get(label, 0), matched.get(label, 0)) for label in totals}


def select_models(args: argparse.Namespace) -> list[dict[str, object]]:
    models = DEFAULT_MODELS
    if args.models:
        wanted = set(args.models)
        models = [model for model in models if str(model["name"]) in wanted]

    custom = []
    for item in args.model:
        if "=" not in item or ":" not in item:
            raise ValueError("--model must be formatted as NAME=WEIGHTS:IMGSZ")
        name, rest = item.split("=", 1)
        weights, imgsz = rest.rsplit(":", 1)
        custom.append({"name": name, "weights": weights, "imgsz": int(imgsz)})

    models = [*models, *custom]
    if not models:
        raise ValueError("No models selected.")
    return models


def evaluate_model(model_entry: dict[str, object], image_paths: list[Path], args: argparse.Namespace, edges: list[float]) -> list[dict[str, object]]:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency while importing Ultralytics: {exc.name}. "
            "Install the Ultralytics runtime dependencies in pytorch_env first."
        ) from exc

    name = str(model_entry["name"])
    imgsz = int(model_entry["imgsz"])
    weights = resolve_path(Path(str(model_entry["weights"])), REPO_ROOT)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights for {name}: {weights}")

    print(f"Evaluating {name}: imgsz={imgsz}, weights={weights}")
    model = YOLO(str(weights))
    counts = {edge_label(lo, hi): [0, 0] for lo, hi in zip(edges[:-1], edges[1:])}

    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=0.7,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        stream=True,
        verbose=False,
    )

    iterator = results
    progress = None
    if not args.no_progress:
        try:
            from tqdm import tqdm

            progress = tqdm(total=len(image_paths), desc=name, unit="img")
        except ModuleNotFoundError:
            progress = None

    for idx, result in enumerate(iterator, start=1):
        image_path = Path(result.path)
        image_height, image_width = result.orig_shape
        gts = load_ground_truths(label_path_for_image(image_path), image_width, image_height, args.size_metric, edges)
        preds = extract_predictions(result)
        image_counts = match_by_bin(gts, preds, args.iou, args.agnostic)
        for label, (total_gt, matched_gt) in image_counts.items():
            counts.setdefault(label, [0, 0])
            counts[label][0] += total_gt
            counts[label][1] += matched_gt
        if progress is not None:
            progress.update(1)
        elif not args.no_progress and (idx == 1 or idx % 50 == 0 or idx == len(image_paths)):
            print(f"{name}: {idx}/{len(image_paths)} images")

    if progress is not None:
        progress.close()

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass

    rows = []
    for label, (total_gt, matched_gt) in counts.items():
        recall = matched_gt / total_gt if total_gt else 0.0
        rows.append(
            {
                "model_name": name,
                "input_size": imgsz,
                "size_metric": args.size_metric,
                "size_bin": label,
                "total_gt": total_gt,
                "matched_gt": matched_gt,
                "recall": round(recall, 6),
                "iou_threshold": args.iou,
                "conf_threshold": args.conf,
                "class_agnostic": bool(args.agnostic),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    cfg = load_data_config(args.data)
    image_paths = read_image_list(cfg[args.split], cfg["_root"], args.limit)
    if not image_paths:
        raise FileNotFoundError(f"No images found for split '{args.split}' in {args.data}")

    edges = parse_bins(args.bins)
    rows: list[dict[str, object]] = []
    for model_entry in select_models(args):
        rows.extend(evaluate_model(model_entry, image_paths, args, edges))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "input_size",
        "size_metric",
        "size_bin",
        "total_gt",
        "matched_gt",
        "recall",
        "iou_threshold",
        "conf_threshold",
        "class_agnostic",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved size-stratified recall to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
