"""Project-wide settings for the box-detection robustness test.

Every number that affects a reported result lives here so the README can cite
it and so nothing is tuned by accident.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
CONDITIONS_CSV = DATA_DIR / "conditions.csv"
RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FAILURES_DIR = RESULTS_DIR / "failures"

# Conditions are four independent tag axes rather than exclusive scene
# buckets, because the photos were shot in one basement under one
# light and every photo carries one value per axis. Metrics are reported per
# value of each axis. Tags are assigned in tag_conditions.py and stored in
# data/conditions.csv alongside the split.
TAG_AXES = {
    "boxes": ("box", "no_box", "multiple_box"),
    "blur": ("sharp", "blur"),
    "occlusion": ("unoccluded", "occluded"),
    "color": ("brown", "white", "mixed", "none"),
}
SPLITS = ("train", "test")

# Single class problem.
CLASS_NAMES = ("box",)

# Fixed confidence threshold. Chosen before any test-set result was
# seen and never tuned on the test set.
CONFIDENCE_THRESHOLD = 0.25

# A prediction matches a ground-truth box if IoU >= this value.
IOU_MATCH_THRESHOLD = 0.5

# Baseline: zero-shot YOLO-World with a text prompt.
# TODO: confirm the current model filename in the Ultralytics docs.
BASELINE_MODEL = "yolov8s-worldv2.pt"
BASELINE_PROMPT = "cardboard box"

# Small pretrained model to fine-tune on the train split.
FINETUNE_BASE_MODEL = "yolov8n.pt"
FINETUNE_EPOCHS = 30
FINETUNE_IMAGE_SIZE = 640

RANDOM_SEED = 0
