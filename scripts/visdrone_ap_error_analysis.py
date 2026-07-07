"""Compute per-class AP and COCO-style area AP for VisDrone checkpoints.

The script uses existing YOLO `best.pt` checkpoints only. It does not retrain
models. Per-class AP is read from Ultralytics validation output, while
small/medium/large AP is computed from validation predictions using COCO area
bins on original-image ground-truth boxes.
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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics_settings"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ULTRALYTICS_PATH = REPO_ROOT / "third_party" / "ultralytics"
if ULTRALYTICS_PATH.exists():
    sys.path.insert(0, str(ULTRALYTICS_PATH))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.metrics import compute_ap  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


CONTROLLED_MODELS = [
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


@dataclass(frozen=True)
class GroundTruth:
    cls: int
    xyxy: tuple[float, float, float, float]
    area: float


@dataclass(frozen=True)
class Prediction:
    image_id: int
    cls: int
    conf: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    image_path: Path
    gts: tuple[GroundTruth, ...]
    preds: tuple[Prediction, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "configs" / "VisDrone.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation, equivalent to --device cpu.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--val-batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model-name filter.")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke tests.")
    parser.add_argument("--skip-per-class", action="store_true", help="Skip Ultralytics per-class AP validation.")
    parser.add_argument("--skip-area", action="store_true", help="Skip small/medium/large area AP calculation.")
    parser.add_argument("--skip-errors", action="store_true", help="Skip FP/FN/localization/class-confusion analysis.")
    parser.add_argument("--error-iou", type=float, default=0.50, help="IoU threshold used for TP/FP/FN matching.")
    parser.add_argument(
        "--localization-iou",
        type=float,
        default=0.10,
        help="Minimum same-class IoU for counting an unmatched prediction as a localization error.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=REPO_ROOT / "results" / "visdrone_ap_error_analysis_val",
    )
    parser.add_argument(
        "--per-class-output",
        type=Path,
        default=REPO_ROOT / "journal_submission" / "visdrone_per_class_ap.csv",
    )
    parser.add_argument(
        "--area-output",
        type=Path,
        default=REPO_ROOT / "journal_submission" / "visdrone_coco_area_ap.csv",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=REPO_ROOT / "journal_submission" / "visdrone_error_summary.csv",
    )
    parser.add_argument(
        "--confusion-output",
        type=Path,
        default=REPO_ROOT / "journal_submission" / "visdrone_confusion_pairs.csv",
    )
    args = parser.parse_args()
    if args.cpu:
        args.device = "cpu"
        args.half = False
    return args


def resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_data_config(data_yaml: Path) -> dict[str, object]:
    cfg: dict[str, object] = {"names": {}}
    current_key: str | None = None
    with data_yaml.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line:
                continue
            if line.startswith(" ") or line.startswith("\t"):
                if current_key == "names" and ":" in line:
                    key, value = line.split(":", 1)
                    cfg["names"][int(key.strip())] = value.strip().strip("'\"")  # type: ignore[index]
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            current_key = key.strip()
            value = value.strip().strip("'\"")
            if current_key in {"path", "train", "val", "test"}:
                cfg[current_key] = value

    data_root = Path(str(cfg.get("path", "")))
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    cfg["_root"] = data_root
    return cfg


def absolute_data_yaml(data_yaml: Path, cfg: dict[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = output_dir / f"{data_yaml.stem}.absolute.yaml"
    names = cfg.get("names", {})
    with runtime_yaml.open("w", encoding="utf-8") as file:
        file.write(f"path: {Path(str(cfg['_root'])).as_posix()}\n")
        for key in ("train", "val", "test"):
            if key in cfg:
                file.write(f"{key}: {cfg[key]}\n")
        file.write("names:\n")
        for idx in sorted(names):  # type: ignore[arg-type]
            file.write(f"  {idx}: {names[idx]}\n")  # type: ignore[index]
    return runtime_yaml


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
                    if value:
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


def load_ground_truths(label_path: Path, image_width: int, image_height: int) -> tuple[GroundTruth, ...]:
    if not label_path.exists():
        return tuple()
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
            gts.append(GroundTruth(cls=cls, xyxy=(x1, y1, x2, y2), area=width_px * height_px))
    return tuple(gts)


def extract_predictions(result, image_id: int) -> tuple[Prediction, ...]:
    if result.boxes is None or len(result.boxes) == 0:
        return tuple()
    boxes = result.boxes.xyxy.cpu().numpy().tolist()
    classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
    confs = result.boxes.conf.cpu().numpy().tolist()
    return tuple(
        Prediction(image_id=image_id, cls=cls, conf=float(conf), xyxy=tuple(map(float, box)))
        for cls, conf, box in zip(classes, confs, boxes)
    )


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def area_bins() -> dict[str, tuple[float, float]]:
    return {
        "all": (0.0, math.inf),
        "small": (0.0, 32.0**2),
        "medium": (32.0**2, 96.0**2),
        "large": (96.0**2, math.inf),
    }


def in_area(gt: GroundTruth, limits: tuple[float, float]) -> bool:
    lo, hi = limits
    return lo <= gt.area < hi


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    ap, _, _ = compute_ap(recall, precision)
    return float(ap)


def evaluate_area_bin(
    records: list[ImageRecord],
    class_ids: list[int],
    area_limits: tuple[float, float],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    per_class_rows: list[dict[str, object]] = []

    gt_by_image_class: dict[tuple[int, int], list[tuple[int, GroundTruth]]] = {}
    preds_by_class: dict[int, list[Prediction]] = {class_id: [] for class_id in class_ids}
    gt_count_by_class: dict[int, int] = {class_id: 0 for class_id in class_ids}

    for record in records:
        for gt_idx, gt in enumerate(record.gts):
            gt_by_image_class.setdefault((record.image_id, gt.cls), []).append((gt_idx, gt))
            if gt.cls in gt_count_by_class and in_area(gt, area_limits):
                gt_count_by_class[gt.cls] += 1
        for pred in record.preds:
            if pred.cls in preds_by_class:
                preds_by_class[pred.cls].append(pred)

    for class_id in class_ids:
        npos = gt_count_by_class[class_id]
        detections = sorted(preds_by_class[class_id], key=lambda item: item.conf, reverse=True)
        ap_by_threshold: list[float] = []

        for threshold in IOU_THRESHOLDS:
            matched: set[tuple[int, int]] = set()
            tp: list[float] = []
            fp: list[float] = []

            for pred in detections:
                candidates = gt_by_image_class.get((pred.image_id, class_id), [])
                in_candidates = [(idx, gt) for idx, gt in candidates if in_area(gt, area_limits)]
                out_candidates = [(idx, gt) for idx, gt in candidates if not in_area(gt, area_limits)]

                best_iou = 0.0
                best_gt_idx: int | None = None
                for gt_idx, gt in in_candidates:
                    if (pred.image_id, gt_idx) in matched:
                        continue
                    iou = box_iou(pred.xyxy, gt.xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx is not None and best_iou >= threshold:
                    matched.add((pred.image_id, best_gt_idx))
                    tp.append(1.0)
                    fp.append(0.0)
                    continue

                ignored = any(box_iou(pred.xyxy, gt.xyxy) >= threshold for _, gt in out_candidates)
                if ignored:
                    continue
                tp.append(0.0)
                fp.append(1.0)

            if npos == 0 or not tp:
                ap_by_threshold.append(0.0)
                continue

            tp_cum = np.cumsum(np.array(tp))
            fp_cum = np.cumsum(np.array(fp))
            recall = tp_cum / (npos + 1e-16)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)
            ap_by_threshold.append(average_precision(recall, precision))

        per_class_rows.append(
            {
                "class_id": class_id,
                "gt_instances": npos,
                "AP50": ap_by_threshold[0],
                "AP50_95": float(np.mean(ap_by_threshold)),
            }
        )

    valid_rows = [row for row in per_class_rows if int(row["gt_instances"]) > 0]
    summary = {
        "gt_instances": float(sum(int(row["gt_instances"]) for row in valid_rows)),
        "AP50": float(np.mean([float(row["AP50"]) for row in valid_rows])) if valid_rows else 0.0,
        "AP50_95": float(np.mean([float(row["AP50_95"]) for row in valid_rows])) if valid_rows else 0.0,
    }
    return per_class_rows, summary


def evaluate_detection_errors(
    records: list[ImageRecord],
    names: dict[int, str],
    match_iou: float,
    localization_iou: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Compact error analysis at one IoU threshold.

    The summary reports standard TP/FP/FN after class-aware greedy matching.
    Unmatched predictions are then categorized as localization errors,
    class-confusion errors, duplicate detections, or background false positives.
    """

    total_gt = 0
    total_pred = 0
    true_positive = 0
    localization_errors = 0
    class_confusion_errors = 0
    duplicate_errors = 0
    background_false_positives = 0
    pedestrian_people_confusions = 0
    vehicle_confusions = 0
    confusion_stats: dict[tuple[int, int], dict[str, float]] = {}
    vehicle_ids = {3, 4, 5, 6, 9}  # car, van, truck, tricycle, motor

    for record in records:
        total_gt += len(record.gts)
        total_pred += len(record.preds)
        sorted_preds = sorted(enumerate(record.preds), key=lambda item: item[1].conf, reverse=True)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        for pred_idx, pred in sorted_preds:
            best_iou = 0.0
            best_gt_idx: int | None = None
            for gt_idx, gt in enumerate(record.gts):
                if gt_idx in matched_gt or gt.cls != pred.cls:
                    continue
                iou = box_iou(pred.xyxy, gt.xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx is not None and best_iou >= match_iou:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                true_positive += 1

        assigned_error_gt: set[int] = set()
        unmatched_gt = set(range(len(record.gts))) - matched_gt
        for pred_idx, pred in sorted_preds:
            if pred_idx in matched_pred:
                continue

            best_same_iou = 0.0
            best_same_idx: int | None = None
            best_diff_iou = 0.0
            best_diff_idx: int | None = None
            best_matched_iou = 0.0

            for gt_idx, gt in enumerate(record.gts):
                iou = box_iou(pred.xyxy, gt.xyxy)
                if gt_idx in matched_gt:
                    best_matched_iou = max(best_matched_iou, iou)
                    continue
                if gt_idx in assigned_error_gt:
                    continue
                if gt.cls == pred.cls:
                    if iou > best_same_iou:
                        best_same_iou = iou
                        best_same_idx = gt_idx
                elif gt_idx in unmatched_gt and iou > best_diff_iou:
                    best_diff_iou = iou
                    best_diff_idx = gt_idx

            if best_same_idx is not None and localization_iou <= best_same_iou < match_iou:
                localization_errors += 1
                assigned_error_gt.add(best_same_idx)
                continue

            if best_diff_idx is not None and best_diff_iou >= match_iou:
                gt = record.gts[best_diff_idx]
                class_confusion_errors += 1
                assigned_error_gt.add(best_diff_idx)
                key = (gt.cls, pred.cls)
                stats = confusion_stats.setdefault(key, {"count": 0.0, "iou_sum": 0.0, "conf_sum": 0.0})
                stats["count"] += 1.0
                stats["iou_sum"] += best_diff_iou
                stats["conf_sum"] += pred.conf
                if {gt.cls, pred.cls} == {0, 1}:
                    pedestrian_people_confusions += 1
                if gt.cls in vehicle_ids and pred.cls in vehicle_ids and gt.cls != pred.cls:
                    vehicle_confusions += 1
                continue

            if best_matched_iou >= match_iou:
                duplicate_errors += 1
            else:
                background_false_positives += 1

    false_positive = total_pred - true_positive
    false_negative = total_gt - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / total_gt if total_gt else 0.0

    summary = {
        "gt_instances": total_gt,
        "predictions": total_pred,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision_at_iou": precision,
        "recall_at_iou": recall,
        "localization_errors": localization_errors,
        "class_confusion_errors": class_confusion_errors,
        "duplicate_errors": duplicate_errors,
        "background_false_positives": background_false_positives,
        "pedestrian_people_confusions": pedestrian_people_confusions,
        "vehicle_confusions": vehicle_confusions,
        "match_iou": match_iou,
        "localization_iou": localization_iou,
    }

    confusion_rows: list[dict[str, object]] = []
    for (actual_cls, predicted_cls), stats in sorted(
        confusion_stats.items(),
        key=lambda item: item[1]["count"],
        reverse=True,
    ):
        count = int(stats["count"])
        confusion_rows.append(
            {
                "actual_class_id": actual_cls,
                "actual_class_name": names.get(actual_cls, str(actual_cls)),
                "predicted_class_id": predicted_cls,
                "predicted_class_name": names.get(predicted_cls, str(predicted_cls)),
                "count": count,
                "mean_iou": stats["iou_sum"] / count if count else 0.0,
                "mean_conf": stats["conf_sum"] / count if count else 0.0,
            }
        )

    return summary, confusion_rows


def selected_models(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.models:
        return CONTROLLED_MODELS
    wanted = set(args.models)
    models = [entry for entry in CONTROLLED_MODELS if str(entry["name"]) in wanted]
    missing = sorted(wanted - {str(entry["name"]) for entry in models})
    if missing:
        raise ValueError(f"Unknown model filters: {', '.join(missing)}")
    return models


def validate_per_class(
    model: YOLO,
    model_name: str,
    imgsz: int,
    data_yaml: Path,
    names: dict[int, str],
    args: argparse.Namespace,
    device: str,
) -> list[dict[str, object]]:
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=imgsz,
        batch=args.val_batch,
        device=device,
        workers=args.workers,
        half=args.half and device != "cpu",
        project=str(args.project),
        name=f"{model_name}_img{imgsz}_per_class",
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=False,
    )

    box = metrics.box
    ap_class_index = [int(value) for value in getattr(box, "ap_class_index", [])]
    p_values = list(getattr(box, "p", []))
    r_values = list(getattr(box, "r", []))
    ap50_values = list(getattr(box, "ap50", []))
    ap_values = list(getattr(box, "ap", []))

    rows: list[dict[str, object]] = []
    for idx, class_id in enumerate(ap_class_index):
        rows.append(
            {
                "model_name": model_name,
                "input_size": imgsz,
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "precision": float(p_values[idx]),
                "recall": float(r_values[idx]),
                "AP50": float(ap50_values[idx]),
                "AP50_95": float(ap_values[idx]),
            }
        )
    return rows


def collect_prediction_records(
    model: YOLO,
    model_name: str,
    imgsz: int,
    image_paths: list[Path],
    args: argparse.Namespace,
    device: str,
) -> list[ImageRecord]:
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        device=device,
        half=args.half and device != "cpu",
        stream=True,
        verbose=False,
    )

    records: list[ImageRecord] = []
    for image_id, result in enumerate(results):
        image_path = Path(result.path)
        image_height, image_width = result.orig_shape
        gts = load_ground_truths(label_path_for_image(image_path), image_width, image_height)
        preds = extract_predictions(result, image_id)
        records.append(ImageRecord(image_id=image_id, image_path=image_path, gts=gts, preds=preds))
        if (image_id + 1) % 50 == 0 or image_id + 1 == len(image_paths):
            print(f"{model_name}: collected predictions for {image_id + 1}/{len(image_paths)} images")
    return records


def cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "outofmemoryerror" in message


def release_memory(model: YOLO | None = None) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_model(
    entry: dict[str, object],
    image_paths: list[Path],
    runtime_data_yaml: Path,
    names: dict[int, str],
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    model_name = str(entry["name"])
    imgsz = int(entry["imgsz"])
    weights = resolve_path(Path(str(entry["weights"])), REPO_ROOT)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights for {model_name}: {weights}")

    print(f"\n[{model_name}] weights={weights}")
    model = YOLO(str(weights))
    per_class_rows = []
    if not args.skip_per_class:
        per_class_rows = validate_per_class(model, model_name, imgsz, runtime_data_yaml, names, args, device)

    area_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    records: list[ImageRecord] | None = None
    if not args.skip_area or not args.skip_errors:
        records = collect_prediction_records(model, model_name, imgsz, image_paths, args, device)

    if not args.skip_area and records is not None:
        class_ids = sorted(names)
        for area_name, limits in area_bins().items():
            _, summary = evaluate_area_bin(records, class_ids, limits)
            area_rows.append(
                {
                    "model_name": model_name,
                    "input_size": imgsz,
                    "area_bin": area_name,
                    "gt_instances": int(summary["gt_instances"]),
                    "AP50": summary["AP50"],
                    "AP50_95": summary["AP50_95"],
                    "area_definition": "COCO: small<32^2, medium=32^2-96^2, large>=96^2 px^2",
                    "prediction_conf": args.conf,
                    "nms_iou": args.nms_iou,
                    "max_det": args.max_det,
                    "device": device,
                }
            )

    if not args.skip_errors and records is not None:
        summary, model_confusions = evaluate_detection_errors(records, names, args.error_iou, args.localization_iou)
        error_rows.append(
            {
                "model_name": model_name,
                "input_size": imgsz,
                **summary,
                "prediction_conf": args.conf,
                "nms_iou": args.nms_iou,
                "max_det": args.max_det,
                "device": device,
            }
        )
        for row in model_confusions:
            confusion_rows.append(
                {
                    "model_name": model_name,
                    "input_size": imgsz,
                    "match_iou": args.error_iou,
                    **row,
                }
            )

    release_memory(model)
    return per_class_rows, area_rows, error_rows, confusion_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    cfg = load_data_config(args.data)
    names = {int(k): str(v) for k, v in dict(cfg["names"]).items()}  # type: ignore[arg-type]
    image_paths = read_image_list(cfg[args.split], Path(str(cfg["_root"])), args.limit)
    if not image_paths:
        raise FileNotFoundError(f"No images found for split '{args.split}' in {args.data}")

    runtime_data_yaml = absolute_data_yaml(args.data, cfg, args.project)
    per_class_rows: list[dict[str, object]] = []
    area_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    per_class_fields = [
        "model_name",
        "input_size",
        "class_id",
        "class_name",
        "precision",
        "recall",
        "AP50",
        "AP50_95",
    ]
    area_fields = [
        "model_name",
        "input_size",
        "area_bin",
        "gt_instances",
        "AP50",
        "AP50_95",
        "area_definition",
        "prediction_conf",
        "nms_iou",
        "max_det",
        "device",
    ]
    error_fields = [
        "model_name",
        "input_size",
        "gt_instances",
        "predictions",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision_at_iou",
        "recall_at_iou",
        "localization_errors",
        "class_confusion_errors",
        "duplicate_errors",
        "background_false_positives",
        "pedestrian_people_confusions",
        "vehicle_confusions",
        "match_iou",
        "localization_iou",
        "prediction_conf",
        "nms_iou",
        "max_det",
        "device",
    ]
    confusion_fields = [
        "model_name",
        "input_size",
        "match_iou",
        "actual_class_id",
        "actual_class_name",
        "predicted_class_id",
        "predicted_class_name",
        "count",
        "mean_iou",
        "mean_conf",
    ]

    for entry in selected_models(args):
        try:
            class_rows, bin_rows, model_error_rows, model_confusion_rows = run_model(
                entry, image_paths, runtime_data_yaml, names, args, args.device
            )
        except RuntimeError as exc:
            if args.device != "cpu" and cuda_oom(exc):
                print(f"CUDA OOM while evaluating {entry['name']}; retrying on CPU.")
                release_memory(None)
                class_rows, bin_rows, model_error_rows, model_confusion_rows = run_model(
                    entry, image_paths, runtime_data_yaml, names, args, "cpu"
                )
            else:
                raise
        per_class_rows.extend(class_rows)
        area_rows.extend(bin_rows)
        error_rows.extend(model_error_rows)
        confusion_rows.extend(model_confusion_rows)
        if per_class_rows and not args.skip_per_class:
            write_csv(args.per_class_output, per_class_rows, per_class_fields)
        if area_rows and not args.skip_area:
            write_csv(args.area_output, area_rows, area_fields)
        if error_rows and not args.skip_errors:
            write_csv(args.error_output, error_rows, error_fields)
            write_csv(args.confusion_output, confusion_rows, confusion_fields)

    if not args.skip_per_class:
        print(f"\nSaved per-class AP to {args.per_class_output}")
    if not args.skip_area:
        print(f"Saved area AP to {args.area_output}")
    if not args.skip_errors:
        print(f"Saved error summary to {args.error_output}")
        print(f"Saved confusion pairs to {args.confusion_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
