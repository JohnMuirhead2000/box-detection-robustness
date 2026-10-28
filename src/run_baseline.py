"""Run a detector on a split and save its predictions.

Baseline: Ultralytics YOLO-World prompted with
config.BASELINE_PROMPT. Every detection is written as one row so
evaluate.py can score the run without touching the model again.

Output columns: image_name, x_min, y_min, x_max, y_max (pixels),
confidence, class_name.

Usage:
    python src/run_baseline.py                       # zero-shot baseline
    python src/run_baseline.py --weights best.pt --output finetuned.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

import config

PREDICTION_COLUMNS = [
    "image_name", "x_min", "y_min", "x_max", "y_max", "confidence",
    "class_name",
]


def load_split(split: str) -> list[str]:
    """Image names recorded under `split` in conditions.csv."""
    table = pd.read_csv(config.CONDITIONS_CSV, dtype=str).fillna("")
    return table.loc[table["split"] == split, "image_name"].tolist()


def load_detector(weights: Path | None = None) -> YOLO:
    """YOLO-World with the prompt, or a fine-tuned model from `weights`."""
    if weights is not None:
        return YOLO(weights)
    model = YOLO(config.BASELINE_MODEL)
    model.set_classes([config.BASELINE_PROMPT])
    return model


def detect(
    model: YOLO, image_names: list[str],
    confidence: float = config.CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Run the detector at `confidence`; one row per box."""
    rows = []
    for image_name in image_names:
        result = model.predict(
            config.IMAGES_DIR / image_name,
            conf=confidence,
            verbose=False,
        )[0]
        # Ultralytics applies EXIF orientation, so xyxy matches the
        # upright image that the labels were drawn on.
        for xyxy, box_confidence in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist()
        ):
            rows.append([image_name, *xyxy, box_confidence, "box"])
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=config.SPLITS)
    parser.add_argument(
        "--output", type=Path,
        default=config.PREDICTIONS_DIR / "baseline.csv",
    )
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="fine-tuned .pt to run instead of the zero-shot baseline",
    )
    parser.add_argument(
        "--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
        help="confidence threshold; lower it only for a train-split sweep",
    )
    args = parser.parse_args()

    image_names = load_split(args.split)
    predictions = detect(load_detector(args.weights), image_names, args.conf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(
        f"{len(predictions)} predictions on {len(image_names)} "
        f"{args.split} images -> {args.output}"
    )


if __name__ == "__main__":
    main()
