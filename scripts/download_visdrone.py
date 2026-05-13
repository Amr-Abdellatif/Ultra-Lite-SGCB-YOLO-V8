from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics.utils.downloads import download


def visdrone2yolo(dir_path: Path) -> None:
    """Convert VisDrone annotations to YOLO label format."""
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    labels_dir = dir_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm((dir_path / "annotations").glob("*.txt"), desc=f"Converting {dir_path.name}")
    for ann_path in pbar:
        img_path = (dir_path / "images" / ann_path.name).with_suffix(".jpg")
        img_size = Image.open(img_path).size

        lines = []
        with ann_path.open(encoding="utf-8") as f:
            for row in [x.split(",") for x in f.read().strip().splitlines()]:
                if row[4] == "0":
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

        (labels_dir / ann_path.name).write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare the VisDrone2019-DET dataset.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/VisDrone",
        help="Target directory for the dataset (default: data/VisDrone).",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip annotation conversion to YOLO labels.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    urls = [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-challenge.zip",
    ]

    print(f"Downloading VisDrone datasets to: {data_root}")
    download(urls, dir=data_root, curl=True, threads=4)

    if args.skip_convert:
        print("Skipping annotation conversion.")
        return 0

    for split in ["VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"]:
        split_dir = data_root / split
        if not (split_dir / "annotations").exists():
            print(f"Skipping conversion for {split} (annotations not found).")
            continue
        visdrone2yolo(split_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
