"""Convert AU-AIR annotations to a YOLO/Ultralytics dataset.

The script keeps the original images in one place and hard-links them into the
YOLO dataset folder by default, so the 2.2 GB frame archive is not duplicated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


AU_AIR_NAMES = [
    "Human",
    "Car",
    "Truck",
    "Van",
    "Motorbike",
    "Bicycle",
    "Bus",
    "Trailer",
]

VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

CLASS_MODES = {
    "auair8": {
        "names": AU_AIR_NAMES,
        "mapping": {idx: idx for idx in range(len(AU_AIR_NAMES))},
        "description": "Native AU-AIR 8-class labels for controlled training.",
    },
    "visdrone_vehicle10": {
        "names": VISDRONE_NAMES,
        "mapping": {
            1: 3,  # Car -> car
            2: 5,  # Truck -> truck
            3: 4,  # Van -> van
            4: 9,  # Motorbike -> motor
            5: 2,  # Bicycle -> bicycle
            6: 8,  # Bus -> bus
        },
        "description": (
            "Vehicle-only AU-AIR labels mapped into VisDrone's 10-class index "
            "space for external validation of VisDrone-trained checkpoints."
        ),
    },
    "visdrone10": {
        "names": VISDRONE_NAMES,
        "mapping": {
            0: 1,  # Human -> people
            1: 3,  # Car -> car
            2: 5,  # Truck -> truck
            3: 4,  # Van -> van
            4: 9,  # Motorbike -> motor
            5: 2,  # Bicycle -> bicycle
            6: 8,  # Bus -> bus
            7: 5,  # Trailer -> truck-like heavy vehicle
        },
        "description": (
            "Approximate full AU-AIR to VisDrone mapping for external "
            "validation; human/trailer classes are necessarily imperfect."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/AU-AIR/raw/annotations.json"),
        help="Path to AU-AIR annotations.json.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/AU-AIR/raw/images"),
        help="Directory containing AU-AIR image frames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/AU-AIR/yolo-auair8"),
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--class-mode",
        choices=sorted(CLASS_MODES),
        default="auair8",
        help="Label space to write.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "none"),
        default="hardlink",
        help="How to place images in the YOLO output directory.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Skip missing frames instead of failing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing labels and split files in the output directory.",
    )
    return parser.parse_args()


def clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def image_path(images_dir: Path, image_name: str) -> Path:
    candidate = images_dir / image_name
    if candidate.exists():
        return candidate
    stem = Path(image_name).stem
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return images_dir / image_name


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


def to_yolo_box(bbox: dict, image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    left = float(bbox["left"])
    top = float(bbox["top"])
    width = float(bbox["width"])
    height = float(bbox["height"])

    x1 = max(0.0, min(left, image_width))
    y1 = max(0.0, min(top, image_height))
    x2 = max(0.0, min(left + width, image_width))
    y2 = max(0.0, min(top + height, image_height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    return x_center, y_center, clipped_width / image_width, clipped_height / image_height


def split_records(records: list[dict], val_fraction: float, test_fraction: float, seed: int) -> dict[str, list[dict]]:
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Split fractions must be in [0, 1).")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val-fraction + test-fraction must be less than 1.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_test = round(n_total * test_fraction)
    n_val = round(n_total * val_fraction)
    return {
        "train": shuffled[: n_total - n_val - n_test],
        "val": shuffled[n_total - n_val - n_test : n_total - n_test],
        "test": shuffled[n_total - n_test :],
    }


def write_yaml(path: Path, names: list[str]) -> None:
    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    text = (
        f"path: {path.resolve().as_posix()}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "names:\n"
        f"{names_yaml}\n"
    )
    (path / "dataset.yaml").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.annotations.exists():
        raise FileNotFoundError(args.annotations)
    if not args.images_dir.exists() and not args.allow_missing_images:
        raise FileNotFoundError(args.images_dir)

    output = args.output
    images_out = output / "images"
    labels_out = output / "labels"
    clean_dir(images_out, args.overwrite)
    clean_dir(labels_out, args.overwrite)

    with args.annotations.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    mode = CLASS_MODES[args.class_mode]
    class_mapping: dict[int, int] = mode["mapping"]
    records = payload["annotations"]

    converted_records = []
    source_class_counts: Counter[int] = Counter()
    target_class_counts: Counter[int] = Counter()
    missing_images = []
    ignored_boxes = 0
    clipped_or_invalid_boxes = 0

    for record in records:
        src = image_path(args.images_dir, record["image_name"])
        if not src.exists():
            missing_images.append(record["image_name"])
            if args.allow_missing_images:
                continue
            raise FileNotFoundError(src)

        width = int(record.get("image_width:", record.get("image_width")))
        height = int(record["image_height"])
        label_lines = []
        for bbox in record.get("bbox", []):
            source_class = int(bbox["class"])
            source_class_counts[source_class] += 1
            if source_class not in class_mapping:
                ignored_boxes += 1
                continue
            target_class = class_mapping[source_class]
            box = to_yolo_box(bbox, width, height)
            if box is None:
                clipped_or_invalid_boxes += 1
                continue
            target_class_counts[target_class] += 1
            label_lines.append(f"{target_class} " + " ".join(f"{value:.6f}" for value in box))

        dst_image = images_out / src.name
        place_image(src, dst_image, args.link_mode)
        (labels_out / f"{dst_image.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        converted_records.append({"image": dst_image, "labels": len(label_lines)})

    splits = split_records(converted_records, args.val_fraction, args.test_fraction, args.seed)
    for split_name, split_records_ in splits.items():
        split_text = "\n".join(item["image"].resolve().as_posix() for item in split_records_) + "\n"
        (output / f"{split_name}.txt").write_text(split_text, encoding="utf-8")

    write_yaml(output, mode["names"])
    summary = {
        "class_mode": args.class_mode,
        "description": mode["description"],
        "annotations": str(args.annotations),
        "images_dir": str(args.images_dir),
        "output": str(output),
        "records_total": len(records),
        "records_converted": len(converted_records),
        "missing_images": len(missing_images),
        "ignored_boxes": ignored_boxes,
        "invalid_boxes_after_clipping": clipped_or_invalid_boxes,
        "splits": {name: len(items) for name, items in splits.items()},
        "source_class_counts": {AU_AIR_NAMES[key]: value for key, value in sorted(source_class_counts.items())},
        "target_class_counts": {
            mode["names"][key]: value for key, value in sorted(target_class_counts.items())
        },
    }
    (output / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if missing_images:
        print(f"First missing images: {missing_images[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
