"""Run the C++ NMS tool over a predictions CSV.

Thin wrapper so the evaluation pipeline uses cpp/nms_filter rather than a
second Python implementation. Build it first:

    cmake -S cpp -B cpp/build && cmake --build cpp/build

Usage:
    python src/apply_nms.py --predictions results/predictions/baseline.csv \
                            --iou 0.5 --output results/predictions/baseline_nms50.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

import config

NMS_BINARY = config.REPO_ROOT / "cpp" / "build" / "nms_filter"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, required=True,
                        help="suppress a box overlapping a kept box by >= this")
    args = parser.parse_args()

    if not NMS_BINARY.exists():
        sys.exit(f"{NMS_BINARY} not found; build cpp/ first (see docstring)")
    subprocess.run(
        [str(NMS_BINARY), str(args.predictions), str(args.output), str(args.iou)],
        check=True,
    )


if __name__ == "__main__":
    main()
