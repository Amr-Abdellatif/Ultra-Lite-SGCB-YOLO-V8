from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


DOWNLOADS = {
    "v1": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip",
        "archive": "DOTAv1.zip",
        "root": "DOTAv1",
        "output": "yolo-dota-v1-hbb",
    },
    "v1.5": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.5.zip",
        "archive": "DOTAv1.5.zip",
        "root": "DOTAv1.5",
        "output": "yolo-dota-v1.5-hbb",
    },
}


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def download(url: str, output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"Using existing archive: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, output.open("wb") as file:
        shutil.copyfileobj(response, file)


def extract_zip(zip_path: Path, output_dir: Path, force: bool) -> None:
    marker = output_dir / ".extract_complete"
    if marker.exists() and not force:
        print(f"Using existing extracted directory: {output_dir}")
        return
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    marker.write_text("ok\n", encoding="utf-8")


def find_source_root(raw_dir: Path, preferred_name: str) -> Path:
    candidates = [
        raw_dir / preferred_name,
        raw_dir / "datasets" / preferred_name,
        raw_dir,
    ]
    for candidate in candidates:
        if (candidate / "images").exists():
            return candidate
    matches = sorted(path for path in raw_dir.rglob("images") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"Could not find DOTA images directory under {raw_dir}")
    return matches[0].parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Ultralytics DOTA and prepare YOLO horizontal-box labels.")
    parser.add_argument("--data-root", type=Path, default=Path("data/DOTA"))
    parser.add_argument("--version", choices=sorted(DOWNLOADS), default="v1.5")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite converted YOLO output.")
    parser.add_argument("--small-only", action="store_true", help="Prepare a small-object-only derivative.")
    parser.add_argument("--max-box-pixels", type=float, default=32.0)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "none"), default="hardlink")
    args = parser.parse_args()

    spec = DOWNLOADS[args.version]
    data_root = args.data_root.resolve()
    raw_dir = data_root / "raw"
    archive_path = raw_dir / spec["archive"]
    extract_dir = raw_dir / spec["root"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download(spec["url"], archive_path, args.force_download)

    if not args.skip_extract:
        extract_zip(archive_path, extract_dir, args.force_extract)

    source_root = find_source_root(extract_dir, spec["root"])
    output_name = spec["output"]
    if args.small_only:
        output_name = f"{output_name}-small{int(args.max_box_pixels)}"

    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "prepare_dota_hbb.py"),
        "--source",
        str(source_root),
        "--output",
        str(data_root / output_name),
        "--version",
        args.version,
        "--splits",
        *args.splits,
        "--link-mode",
        args.link_mode,
        "--max-box-pixels",
        str(args.max_box_pixels),
    ]
    if args.small_only:
        command.append("--small-only")
    if args.overwrite:
        command.append("--overwrite")
    run(command)

    print("DOTA download/prepare complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
