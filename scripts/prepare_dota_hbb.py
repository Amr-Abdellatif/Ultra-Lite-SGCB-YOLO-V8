"""Convert DOTA OBB annotations into a YOLO horizontal-box dataset.

This is intended for controlled comparisons with the detect models used in the
manuscript. The script accepts the Ultralytics DOTAv1/DOTAv1.5 archives, which
already contain split images and normalized OBB labels, and also handles raw
DOTA labelTxt-style polygon annotations when present.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

DOTA_V1_NAMES = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
]

DOTA_V15_NAMES = DOTA_V1_NAMES + ["container crane"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/DOTA/raw/DOTAv1.5"),
        help="Source DOTA root containing images/<split> and labels/<split> or labelTxt/<split>.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/DOTA/yolo-dota-v1.5-hbb"),
        help="Output YOLO detect dataset directory.",
    )
    parser.add_argument("--version", choices=("v1", "v1.5"), default="v1.5")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to convert. Training uses train/val; test can be added for prediction-only use.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "none"),
        default="hardlink",
        help="How to place source images into the converted dataset.",
    )
    parser.add_argument(
        "--small-only",
        action="store_true",
        help="Keep only boxes with both width and height <= --max-box-pixels.",
    )
    parser.add_argument(
        "--max-box-pixels",
        type=float,
        default=32.0,
        help="Small-object threshold used only with --small-only.",
    )
    parser.add_argument(
        "--min-box-pixels",
        type=float,
        default=1.0,
        help="Discard boxes with width or height below this pixel size after clipping.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output directory first.")
    return parser.parse_args()


def names_for_version(version: str) -> list[str]:
    return DOTA_V15_NAMES if version == "v1.5" else DOTA_V1_NAMES


def ensure_clean_output(output: Path, overwrite: bool) -> None:
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def place_image(src: Path, dst: Path, mode: str) -> None:
    if mode == "none" or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def find_split_dir(root: Path, base_name: str, split: str) -> Path | None:
    candidates = [
        root / base_name / split,
        root / split / base_name,
        root / split,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_label_file(root: Path, split: str, stem: str) -> Path | None:
    candidates = [
        root / "labels" / split / f"{stem}.txt",
        root / "labelTxt" / split / f"{stem}.txt",
        root / split / "labels" / f"{stem}.txt",
        root / split / "labelTxt" / f"{stem}.txt",
        root / "labels" / f"{stem}.txt",
        root / "labelTxt" / f"{stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix in IMAGE_SUFFIXES


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        try:
            import cv2

            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not read image: {path}")
            height, width = image.shape[:2]
            return width, height
        except Exception as exc:
            raise RuntimeError("Could not read image size. Install Pillow or opencv-python.") from exc


def normalize_hbb(
    points: list[float],
    image_width: int,
    image_height: int,
    already_normalized: bool,
    min_box_pixels: float,
    small_only: bool,
    max_box_pixels: float,
) -> tuple[float, float, float, float] | None:
    xs = points[0::2]
    ys = points[1::2]
    if already_normalized:
        xs_px = [x * image_width for x in xs]
        ys_px = [y * image_height for y in ys]
    else:
        xs_px = xs
        ys_px = ys

    x1 = max(0.0, min(xs_px))
    y1 = max(0.0, min(ys_px))
    x2 = min(float(image_width), max(xs_px))
    y2 = min(float(image_height), max(ys_px))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < min_box_pixels or box_h < min_box_pixels:
        return None
    if small_only and (box_w > max_box_pixels or box_h > max_box_pixels):
        return None

    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    return x_center, y_center, box_w / image_width, box_h / image_height


def parse_label_line(
    line: str,
    names: list[str],
    image_width: int,
    image_height: int,
    min_box_pixels: float,
    small_only: bool,
    max_box_pixels: float,
) -> tuple[int, tuple[float, float, float, float]] | None:
    parts = line.strip().lstrip("\ufeff").split()
    if len(parts) < 9:
        return None

    # Ultralytics OBB labels: class x1 y1 x2 y2 x3 y3 x4 y4, normalized.
    try:
        class_id = int(float(parts[0]))
        points = [float(value) for value in parts[1:9]]
        box = normalize_hbb(points, image_width, image_height, True, min_box_pixels, small_only, max_box_pixels)
        if box is None:
            return None
        return class_id, box
    except ValueError:
        pass

    # Raw DOTA labelTxt: x1 y1 ... x4 y4 class difficulty, pixel coordinates.
    try:
        points = [float(value) for value in parts[:8]]
    except ValueError:
        return None
    class_name = " ".join(parts[8:-1]) if len(parts) > 9 else parts[8]
    if class_name not in names:
        return None
    box = normalize_hbb(points, image_width, image_height, False, min_box_pixels, small_only, max_box_pixels)
    if box is None:
        return None
    return names.index(class_name), box


def convert_split(args: argparse.Namespace, split: str, names: list[str]) -> dict[str, object]:
    images_dir = find_split_dir(args.source, "images", split)
    if images_dir is None:
        raise FileNotFoundError(f"Could not find images for split '{split}' under {args.source}")

    image_paths = sorted(path for path in images_dir.rglob("*") if is_image(path))
    output_images = args.output / "images" / split
    output_labels = args.output / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[int] = Counter()
    missing_labels = 0
    kept_boxes = 0
    skipped_boxes = 0

    for src_image in image_paths:
        dst_image = output_images / src_image.name
        place_image(src_image, dst_image, args.link_mode)

        width, height = image_size(src_image)
        label_file = find_label_file(args.source, split, src_image.stem)
        label_lines: list[str] = []

        if label_file is None:
            missing_labels += 1
        else:
            for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                parsed = parse_label_line(
                    line=line,
                    names=names,
                    image_width=width,
                    image_height=height,
                    min_box_pixels=args.min_box_pixels,
                    small_only=args.small_only,
                    max_box_pixels=args.max_box_pixels,
                )
                if parsed is None:
                    skipped_boxes += 1
                    continue
                class_id, box = parsed
                if class_id < 0 or class_id >= len(names):
                    skipped_boxes += 1
                    continue
                class_counts[class_id] += 1
                kept_boxes += 1
                label_lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in box))

        (output_labels / f"{dst_image.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    return {
        "images": len(image_paths),
        "missing_labels": missing_labels,
        "kept_boxes": kept_boxes,
        "skipped_boxes": skipped_boxes,
        "class_counts": {names[key]: value for key, value in sorted(class_counts.items())},
    }


def write_dataset_yaml(output: Path, names: list[str], splits: list[str]) -> None:
    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    lines = [
        f"path: {output.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if "test" in splits:
        lines.append("test: images/test")
    lines.extend(["names:", names_yaml, ""])
    (output / "dataset.yaml").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)

    ensure_clean_output(args.output, args.overwrite)
    names = names_for_version(args.version)

    split_summaries = {}
    for split in args.splits:
        split_summaries[split] = convert_split(args, split, names)

    write_dataset_yaml(args.output, names, args.splits)
    summary = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "version": args.version,
        "names": names,
        "small_only": args.small_only,
        "max_box_pixels": args.max_box_pixels if args.small_only else None,
        "min_box_pixels": args.min_box_pixels,
        "splits": split_summaries,
        "dataset_yaml": str((args.output / "dataset.yaml").resolve()),
    }
    (args.output / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
