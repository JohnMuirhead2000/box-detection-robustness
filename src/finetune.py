"""Fine-tune a small pretrained YOLO model on the train split only.

The one targeted change for the worst failure mode (motion blur, baseline
recall 0.14): every train image gets a motion-blurred copy with the same
labels, and brightness jitter is enabled during training. The test split
is never read here. Score the result with run_baseline.py --weights and
evaluate.py so before/after use identical threshold and metric code.

Every setting is in config.py or printed at the end for the README.

Usage:
    python src/finetune.py [--epochs 30] [--imgsz 640] [--no-blur] [--name finetune]
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from ultralytics import YOLO

import config

FINETUNE_DIR = config.DATA_DIR / "finetune"
AUGMENTED_IMAGES = FINETUNE_DIR / "images"
AUGMENTED_LABELS = FINETUNE_DIR / "labels"
TRAIN_LIST = FINETUNE_DIR / "train.txt"
DATASET_YAML = FINETUNE_DIR / "data.yaml"
RUNS_DIR = config.RESULTS_DIR / "runs"
RUN_NAME = "finetune"

# Motion blur kernel length range in pixels at the stored half resolution,
# chosen to look like the hand-shake blur in the blur-tagged photos.
BLUR_LENGTH_RANGE = (30, 70)
BRIGHTNESS_JITTER = 0.4   # ultralytics hsv_v: +/- fraction of value


def train_image_names() -> list[str]:
    table = pd.read_csv(config.CONDITIONS_CSV, dtype=str).fillna("")
    return table.loc[table["split"] == "train", "image_name"].tolist()


def motion_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Convolve with a straight-line kernel of random length and angle."""
    length = rng.randint(*BLUR_LENGTH_RANGE)
    angle = rng.uniform(0, 180)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def make_blurred_copies(image_names: list[str]) -> list[Path]:
    """Blurred copy of each train image; labels copied unchanged."""
    rng = random.Random(config.RANDOM_SEED)
    AUGMENTED_IMAGES.mkdir(parents=True, exist_ok=True)
    AUGMENTED_LABELS.mkdir(parents=True, exist_ok=True)
    outputs = []
    for image_name in image_names:
        stem = Path(image_name).stem
        # Read through PIL to apply EXIF rotation, matching the labels.
        with Image.open(config.IMAGES_DIR / image_name) as raw:
            upright = np.array(ImageOps.exif_transpose(raw).convert("RGB"))
        blurred = motion_blur(upright[:, :, ::-1], rng)   # RGB -> BGR
        output = AUGMENTED_IMAGES / f"{stem}_blur.jpg"
        cv2.imwrite(str(output), blurred, [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copy2(config.LABELS_DIR / f"{stem}.txt",
                     AUGMENTED_LABELS / f"{stem}_blur.txt")
        outputs.append(output)
    return outputs


def write_dataset(image_names: list[str], blurred: list[Path]) -> None:
    """train.txt lists images; ultralytics finds labels via images->labels."""
    paths = [config.IMAGES_DIR / n for n in image_names] + blurred
    TRAIN_LIST.write_text("\n".join(str(p.resolve()) for p in paths) + "\n")
    # val = train: no held-out tuning set with 57 images; the test split
    # is never touched, so best.pt selection cannot leak test information.
    DATASET_YAML.write_text(
        f"train: {TRAIN_LIST.resolve()}\n"
        f"val: {TRAIN_LIST.resolve()}\n"
        f"nc: {len(config.CLASS_NAMES)}\n"
        f"names: {list(config.CLASS_NAMES)}\n"
    )


def train(epochs: int, image_size: int, run_name: str) -> Path:
    model = YOLO(config.FINETUNE_BASE_MODEL)
    model.train(
        data=str(DATASET_YAML), epochs=epochs, imgsz=image_size,
        seed=config.RANDOM_SEED, hsv_v=BRIGHTNESS_JITTER,
        project=str(RUNS_DIR), name=run_name, exist_ok=True,
        verbose=False, plots=False,
    )
    return RUNS_DIR / run_name / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.FINETUNE_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=config.FINETUNE_IMAGE_SIZE)
    parser.add_argument("--no-blur", action="store_true",
                        help="skip the blurred copies (ablation)")
    parser.add_argument("--name", default=RUN_NAME,
                        help="run folder under results/runs")
    args = parser.parse_args()

    names = train_image_names()
    blurred = [] if args.no_blur else make_blurred_copies(names)
    write_dataset(names, blurred)
    weights = train(args.epochs, args.imgsz, args.name)
    print(
        f"settings: base={config.FINETUNE_BASE_MODEL} epochs={args.epochs} "
        f"imgsz={args.imgsz} seed={config.RANDOM_SEED} "
        f"hsv_v={BRIGHTNESS_JITTER} blur_copies={len(blurred)} "
        f"train_images={len(names) + len(blurred)}"
    )
    print(f"best weights: {weights}")


if __name__ == "__main__":
    main()
