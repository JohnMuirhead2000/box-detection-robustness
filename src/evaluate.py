"""Score a predictions file against the YOLO-format labels.

Written by hand rather than with a library so every number is explainable.

Scoring rules:
  * IoU = intersection area / union area of two boxes.
  * Per image, predictions are visited in descending confidence. One is a
    true positive (TP) if its best IoU with a still-unmatched true box is
    at least config.IOU_MATCH_THRESHOLD; otherwise a false positive (FP).
    Each true box can be claimed once. Unclaimed true boxes are false
    negatives (FN).
  * precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = harmonic
    mean. A 0/0 ratio is reported as 0.0.
  * Reported overall and per value of every tag axis in conditions.csv.

Usage:
    python src/evaluate.py --predictions results/predictions/baseline.csv \
                           --output results/baseline_metrics.csv
    python src/evaluate.py ... --per-image   # also print one row per image
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

import config

# A box is (x_min, y_min, x_max, y_max) in pixels throughout this file.
Box = tuple[float, float, float, float]


def compute_iou(box_a: Box, box_b: Box) -> float:
    """Intersection over union of two pixel-space boxes."""
    inter_width = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    inter_height = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    if inter_width <= 0 or inter_height <= 0:
        return 0.0
    intersection = inter_width * inter_height
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def image_size(image_name: str) -> tuple[int, int]:
    """Width and height of the upright image, matching the label frame."""
    with Image.open(config.IMAGES_DIR / image_name) as image:
        return ImageOps.exif_transpose(image).size


def load_ground_truth(image_name: str) -> list[Box]:
    """YOLO label lines (class cx cy w h, normalised) as pixel boxes."""
    label_path = (config.LABELS_DIR / image_name).with_suffix(".txt")
    width, height = image_size(image_name)
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = (float(v) for v in line.split())
        boxes.append((
            (cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height,
        ))
    return boxes


def match_predictions(
    predictions: list[tuple[Box, float]],
    truths: list[Box],
    iou_threshold: float,
) -> tuple[int, int, int, list[float]]:
    """Greedy confidence-ordered matching -> (tp, fp, fn, matched_ious)."""
    unmatched = list(range(len(truths)))
    true_positives, false_positives, matched_ious = 0, 0, []
    for box, _ in sorted(predictions, key=lambda p: p[1], reverse=True):
        best_iou, best_index = 0.0, None
        for index in unmatched:
            iou = compute_iou(box, truths[index])
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index is not None and best_iou >= iou_threshold:
            true_positives += 1
            matched_ious.append(best_iou)
            unmatched.remove(best_index)
        else:
            false_positives += 1
    return true_positives, false_positives, len(unmatched), matched_ious


def precision_recall_f1(
    tp: int, fp: int, fn: int
) -> tuple[float, float, float]:
    """(precision, recall, f1), with 0/0 defined as 0.0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return precision, recall, f1


def score_images(predictions_csv: Path, split: str) -> pd.DataFrame:
    """One row per image in `split`: its tags plus tp, fp, fn counts."""
    conditions = pd.read_csv(config.CONDITIONS_CSV, dtype=str).fillna("")
    conditions = conditions[conditions["split"] == split]
    predictions = pd.read_csv(predictions_csv)

    rows = []
    for record in conditions.itertuples(index=False):
        image_predictions = predictions[
            predictions["image_name"] == record.image_name
        ]
        prediction_list = [
            ((p.x_min, p.y_min, p.x_max, p.y_max), p.confidence)
            for p in image_predictions.itertuples()
        ]
        truths = load_ground_truth(record.image_name)
        tp, fp, fn, _ = match_predictions(
            prediction_list, truths, config.IOU_MATCH_THRESHOLD
        )
        rows.append({
            "image_name": record.image_name,
            **{axis: getattr(record, axis) for axis in config.TAG_AXES},
            "truth_boxes": len(truths), "predictions": len(prediction_list),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return pd.DataFrame(rows)


def summarise(per_image: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tp/fp/fn overall and per value of each tag axis."""
    groups = [("overall", "all", per_image)]
    for axis in config.TAG_AXES:
        for value, subset in per_image.groupby(axis, sort=False):
            groups.append((axis, value, subset))

    rows = []
    for axis, value, subset in groups:
        tp, fp, fn = int(subset["tp"].sum()), int(subset["fp"].sum()), \
            int(subset["fn"].sum())
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        rows.append({
            "axis": axis, "condition": value, "images": len(subset),
            "truth_boxes": int(subset["truth_boxes"].sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=config.SPLITS)
    parser.add_argument("--per-image", action="store_true")
    args = parser.parse_args()

    per_image = score_images(args.predictions, args.split)
    metrics = summarise(per_image)
    metrics.to_csv(args.output, index=False)
    per_image.to_csv(args.output.with_name(
        args.output.stem + "_per_image.csv"), index=False)
    if args.per_image:
        print(per_image.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
