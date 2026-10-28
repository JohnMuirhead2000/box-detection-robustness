"""Copy tagged phone photos from data/incoming/unsorted into data/images.

Reads data/incoming/assignments.csv (written by tag_conditions.py), copies each
listed photo to data/images/img_<phone number>.jpg so every repo image traces
back to the original, converts HEIC with macOS `sips` if needed, and writes the
tag columns plus an empty split column into data/conditions.csv. Pixels are
never resized or edited. Photos not in the manifest are left out on purpose.

Usage:
    python src/import_photos.py [--dry-run]
"""

import argparse
import re
import shutil
import subprocess

import pandas as pd

import config

INCOMING_DIR = config.DATA_DIR / "incoming"
UNSORTED_DIR = INCOMING_DIR / "unsorted"
MANIFEST_PATH = INCOMING_DIR / "assignments.csv"


def repo_name(phone_name: str) -> str:
    """IMG_1708.JPG -> img_1708.jpg; anything else is lowercased with a .jpg suffix."""
    match = re.match(r"IMG_(\d+)", phone_name, re.IGNORECASE)
    stem = f"img_{match.group(1)}" if match else phone_name.rsplit(".", 1)[0].lower()
    return f"{stem}.jpg"


def copy_photo(source, destination, dry_run: bool) -> None:
    print(f"{source.name} -> {destination.name}")
    if dry_run or destination.exists():
        return
    if source.suffix.lower() == ".heic":
        subprocess.run(["sips", "-s", "format", "jpeg", str(source), "--out", str(destination)],
                       check=True, capture_output=True)
    else:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_PATH, dtype=str)
    for axis, allowed in config.TAG_AXES.items():
        bad = manifest[~manifest[axis].isin(allowed)]
        if not bad.empty:
            raise ValueError(f"unknown {axis} values: {bad[axis].unique()}")

    rows = []
    for record in manifest.itertuples(index=False):
        source = UNSORTED_DIR / record.file_name
        destination = config.IMAGES_DIR / repo_name(record.file_name)
        copy_photo(source, destination, args.dry_run)
        rows.append({"image_name": destination.name, **{axis: getattr(record, axis) for axis in config.TAG_AXES}})

    conditions = pd.DataFrame(rows)
    conditions["split"] = ""
    if config.CONDITIONS_CSV.exists():
        # Never move an image that was already split.
        existing = pd.read_csv(config.CONDITIONS_CSV, dtype=str).fillna("")
        conditions["split"] = conditions["image_name"].map(dict(zip(existing["image_name"], existing["split"]))).astype(object).fillna("")
    if not args.dry_run:
        conditions.to_csv(config.CONDITIONS_CSV, index=False)
    print(f"{len(conditions)} images")
    for axis in config.TAG_AXES:
        print(conditions[axis].value_counts().to_string(), end="\n\n")


if __name__ == "__main__":
    main()
