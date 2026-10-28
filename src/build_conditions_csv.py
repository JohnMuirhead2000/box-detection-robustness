"""Assign the held-out train/test split in data/conditions.csv.

Run once, after labeling and before any model is run.
Labeled images (those with a file in data/labels/) are split about 50/50 with a
fixed seed, stratified on the boxes and blur tags so every value of every axis
is represented on both sides. Images already assigned a split are never moved.

Usage:
    python src/build_conditions_csv.py [--test-fraction 0.5]
"""

import argparse
import random

import pandas as pd

import config


def has_label(image_name: str) -> bool:
    return (config.LABELS_DIR / image_name).with_suffix(".txt").exists()


def assign_split(table: pd.DataFrame, test_fraction: float) -> pd.DataFrame:
    rng = random.Random(config.RANDOM_SEED)
    eligible = table[(table["split"] == "") & table["image_name"].map(has_label)]
    for _, group in eligible.groupby(["boxes", "blur"]):
        indices = group.index.tolist()
        rng.shuffle(indices)
        test_count = round(len(indices) * test_fraction)
        table.loc[indices[:test_count], "split"] = "test"
        table.loc[indices[test_count:], "split"] = "train"
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    args = parser.parse_args()

    table = pd.read_csv(config.CONDITIONS_CSV, dtype=str).fillna("")
    table = assign_split(table, args.test_fraction)
    table.to_csv(config.CONDITIONS_CSV, index=False)
    print(table.groupby(["boxes", "blur", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
