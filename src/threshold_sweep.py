"""Sweep the confidence threshold on the TRAIN split only.

Picks the threshold by train-split F1 so the test split is never used for
tuning. Input is a low-threshold prediction file for the train split:

    python run_baseline.py --split train --conf 0.05 \
        --output ../results/predictions/baseline_train_conf005.csv
    python threshold_sweep.py \
        --predictions ../results/predictions/baseline_train_conf005.csv
"""

import argparse
from pathlib import Path

import pandas as pd

import config
import evaluate

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def sweep(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    conditions = pd.read_csv(config.CONDITIONS_CSV, dtype=str).fillna("")
    names = conditions.loc[conditions["split"] == split, "image_name"]
    truths = {name: evaluate.load_ground_truth(name) for name in names}

    rows = []
    for threshold in THRESHOLDS:
        tp = fp = fn = 0
        for name in names:
            subset = predictions[(predictions["image_name"] == name)
                                 & (predictions["confidence"] >= threshold)]
            boxes = [((r.x_min, r.y_min, r.x_max, r.y_max), r.confidence)
                     for r in subset.itertuples()]
            a, b, c, _ = evaluate.match_predictions(
                boxes, truths[name], config.IOU_MATCH_THRESHOLD)
            tp, fp, fn = tp + a, fp + b, fn + c
        precision, recall, f1 = evaluate.precision_recall_f1(tp, fp, fn)
        rows.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(precision, 3),
                     "recall": round(recall, 3), "f1": round(f1, 3)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=config.SPLITS)
    parser.add_argument("--output", type=Path,
                        default=config.RESULTS_DIR / "threshold_sweep_train.csv")
    args = parser.parse_args()

    table = sweep(pd.read_csv(args.predictions), args.split)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    best = table.loc[table["f1"].idxmax()]
    print(f"best F1 on {args.split}: threshold {best.threshold:.2f}")


if __name__ == "__main__":
    main()
