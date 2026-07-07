from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path


ANNOTATIONS_ID = "1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA"
FRAMES_ID = "1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ"


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def download_with_gdown(file_id: str, output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"Using existing archive: {output}")
        return
    if shutil.which("gdown") is None:
        raise RuntimeError("gdown is required. Install it with: pip install gdown")
    output.parent.mkdir(parents=True, exist_ok=True)
    run(["gdown", file_id, "-O", str(output)])


def any_matches(directory: Path, patterns: list[str]) -> bool:
    if not directory.exists():
        return False
    for pattern in patterns:
        if next(directory.rglob(pattern), None) is not None:
            return True
    return False


def extract_zip(zip_path: Path, output_dir: Path, force: bool, expected_patterns: list[str]) -> None:
    if any_matches(output_dir, expected_patterns) and not force:
        print(f"Using existing extracted directory: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def find_annotations_json(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.rglob("annotations.json"))
    if not candidates:
        raise FileNotFoundError(f"Could not find annotations.json under {raw_dir}")
    return candidates[0]


def find_image_dir(frames_dir: Path) -> Path:
    image_paths = [
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        for path in frames_dir.rglob(pattern)
    ]
    if not image_paths:
        raise FileNotFoundError(f"Could not find AU-AIR images under {frames_dir}")
    parent_counts = Counter(path.parent for path in image_paths)
    image_dir, count = parent_counts.most_common(1)[0]
    print(f"Detected image directory: {image_dir} ({count} images)")
    return image_dir


def prepare_mode(
    annotations: Path,
    image_dir: Path,
    output: Path,
    class_mode: str,
    overwrite: bool,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "prepare_auair.py"),
        "--annotations",
        str(annotations),
        "--images-dir",
        str(image_dir),
        "--output",
        str(output),
        "--class-mode",
        class_mode,
    ]
    if overwrite:
        command.append("--overwrite")
    run(command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare the AU-AIR dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("data/AU-AIR"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite converted YOLO folders.")
    parser.add_argument(
        "--class-modes",
        nargs="+",
        default=["auair8", "visdrone_vehicle10"],
        choices=["auair8", "visdrone10", "visdrone_vehicle10"],
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    raw_dir = data_root / "raw"
    frames_dir = raw_dir / "frames_extracted"
    annotations_zip = raw_dir / "auair2019annotations.zip"
    frames_zip = raw_dir / "auair2019data.zip"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_with_gdown(ANNOTATIONS_ID, annotations_zip, args.force_download)
        download_with_gdown(FRAMES_ID, frames_zip, args.force_download)

    if not args.skip_extract:
        extract_zip(annotations_zip, raw_dir, args.force_extract, ["annotations.json"])
        extract_zip(frames_zip, frames_dir, args.force_extract, ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"])

    annotations = find_annotations_json(raw_dir)
    image_dir = find_image_dir(frames_dir)

    output_names = {
        "auair8": "yolo-auair8",
        "visdrone10": "yolo-visdrone10",
        "visdrone_vehicle10": "yolo-visdrone-vehicle10",
    }
    for class_mode in args.class_modes:
        prepare_mode(
            annotations=annotations,
            image_dir=image_dir,
            output=data_root / output_names[class_mode],
            class_mode=class_mode,
            overwrite=args.overwrite,
        )

    print("AU-AIR download/prepare complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
