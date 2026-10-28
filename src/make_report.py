"""Steps 4-6 reporting: README tables, before/after plots, failure images.

Reads results/baseline_metrics.csv and, if present,
results/finetuned_metrics.csv. Prints Markdown tables so README numbers
come straight from the CSVs, saves a per-condition before/after chart,
and renders annotated failure images (green = ground truth, red =
prediction) for the failure analysis.

Usage:
    python src/make_report.py                       # tables (+ plot)
    python src/make_report.py --failures img_1799.jpg img_1757.jpg ...
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

import config
import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

TRUTH_COLOUR = (0, 255, 0)
PREDICTION_COLOUR = (255, 0, 0)
LINE_WIDTH = 14
FONT_SIZE = 70
OUTPUT_LONG_EDGE = 1600


def metrics_to_markdown(metrics: pd.DataFrame) -> str:
    """Markdown table of a metrics frame, one row per condition."""
    columns = list(metrics.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(row[c]) for c in columns) + " |"
        for _, row in metrics.iterrows()
    ]
    return "\n".join([header, rule, *body])


def before_after_table(
    baseline: pd.DataFrame, finetuned: pd.DataFrame
) -> pd.DataFrame:
    """Join the two metric tables per condition with delta columns."""
    keys = ["axis", "condition"]
    merged = baseline.merge(
        finetuned, on=keys, suffixes=("_baseline", "_finetuned")
    )
    for metric in ("precision", "recall", "f1"):
        merged[f"{metric}_delta"] = (
            merged[f"{metric}_finetuned"] - merged[f"{metric}_baseline"]
        ).round(3)
    return merged[keys + ["images_baseline"] + [
        c for m in ("precision", "recall", "f1")
        for c in (f"{m}_baseline", f"{m}_finetuned", f"{m}_delta")
    ]]


def plot_before_after(table: pd.DataFrame, output: Path) -> None:
    """Grouped bars of recall and F1 per condition, before vs after."""
    labels = table["axis"] + ":" + table["condition"]
    positions = range(len(table))
    width = 0.2
    fig, axis = plt.subplots(figsize=(12, 5))
    for offset, column, label in (
        (-1.5, "recall_baseline", "recall (baseline)"),
        (-0.5, "recall_finetuned", "recall (fine-tuned)"),
        (0.5, "f1_baseline", "F1 (baseline)"),
        (1.5, "f1_finetuned", "F1 (fine-tuned)"),
    ):
        axis.bar([p + offset * width for p in positions], table[column],
                 width, label=label)
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def annotate_failure(
    image_name: str, predictions: pd.DataFrame, output_dir: Path
) -> Path:
    """Draw truth (green) and predictions (red, with confidence)."""
    with Image.open(config.IMAGES_DIR / image_name) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=FONT_SIZE)

    for box in evaluate.load_ground_truth(image_name):
        draw.rectangle(box, outline=TRUTH_COLOUR, width=LINE_WIDTH)
    rows = predictions[predictions["image_name"] == image_name]
    for row in rows.itertuples():
        box = (row.x_min, row.y_min, row.x_max, row.y_max)
        draw.rectangle(box, outline=PREDICTION_COLOUR, width=LINE_WIDTH)
        draw.text((row.x_min + 10, row.y_min + 10), f"{row.confidence:.2f}",
                  fill=PREDICTION_COLOUR, font=font)

    image.thumbnail((OUTPUT_LONG_EDGE, OUTPUT_LONG_EDGE))
    output = output_dir / image_name
    image.save(output, quality=85)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", nargs="*", metavar="IMAGE",
                        help="render these test images to results/failures")
    parser.add_argument("--predictions", type=Path,
                        default=config.PREDICTIONS_DIR / "baseline.csv")
    args = parser.parse_args()

    if args.failures:
        predictions = pd.read_csv(args.predictions)
        config.FAILURES_DIR.mkdir(parents=True, exist_ok=True)
        for image_name in args.failures:
            print(annotate_failure(image_name, predictions,
                                   config.FAILURES_DIR))
        return

    baseline = pd.read_csv(config.RESULTS_DIR / "baseline_metrics.csv")
    print(metrics_to_markdown(baseline))
    finetuned_path = config.RESULTS_DIR / "finetuned_metrics.csv"
    if finetuned_path.exists():
        table = before_after_table(baseline, pd.read_csv(finetuned_path))
        print()
        print(metrics_to_markdown(table))
        plot_before_after(table, config.RESULTS_DIR / "before_after.png")


if __name__ == "__main__":
    main()
