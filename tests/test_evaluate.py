"""Unit tests for the hand-written scorer in src/evaluate.py.

Same edge cases as cpp/test_iou.cpp so the two implementations can be
checked against each other.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate  # noqa: E402


def test_iou_identical_boxes():
    assert evaluate.compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_normal_overlap():
    assert evaluate.compute_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_no_overlap():
    assert evaluate.compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_touching_edges():
    box = (0, 0, 10, 10)
    assert evaluate.compute_iou(box, (10, 0, 20, 10)) == 0.0
    assert evaluate.compute_iou(box, (0, 10, 10, 20)) == 0.0
    assert evaluate.compute_iou(box, (10, 10, 20, 20)) == 0.0


def test_iou_contained_box():
    assert evaluate.compute_iou((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(0.04)


def test_match_empty_inputs():
    assert evaluate.match_predictions([], [], 0.5) == (0, 0, 0, [])
    assert evaluate.match_predictions([], [(0, 0, 10, 10)], 0.5) == (0, 0, 1, [])
    assert evaluate.match_predictions([((0, 0, 10, 10), 0.9)], [], 0.5) == (0, 1, 0, [])


def test_match_one_true_box_claimed_once():
    truths = [(0, 0, 10, 10)]
    predictions = [((0, 0, 10, 10), 0.6), ((1, 1, 11, 11), 0.9)]
    tp, fp, fn, ious = evaluate.match_predictions(predictions, truths, 0.5)
    assert (tp, fp, fn) == (1, 1, 0)
    # Highest confidence wins the match: the offset box, not the exact one.
    assert ious == [pytest.approx(evaluate.compute_iou((1, 1, 11, 11), (0, 0, 10, 10)))]


def test_match_below_threshold_is_fp_and_fn():
    tp, fp, fn, _ = evaluate.match_predictions(
        [((5, 0, 15, 10), 0.9)], [(0, 0, 10, 10)], 0.5)
    assert (tp, fp, fn) == (0, 1, 1)


def test_match_threshold_is_inclusive():
    # IoU exactly 1/3 counts as a match at threshold 1/3.
    tp, fp, fn, _ = evaluate.match_predictions(
        [((5, 0, 15, 10), 0.9)], [(0, 0, 10, 10)], 1 / 3)
    assert (tp, fp, fn) == (1, 0, 0)


def test_match_prefers_best_overlap():
    truths = [(0, 0, 10, 10), (8, 0, 18, 10)]
    tp, fp, fn, ious = evaluate.match_predictions(
        [((7, 0, 17, 10), 0.9)], truths, 0.5)
    assert (tp, fp, fn) == (1, 0, 1)
    assert ious[0] == pytest.approx(evaluate.compute_iou((7, 0, 17, 10), (8, 0, 18, 10)))


@pytest.mark.parametrize("tp,fp,fn,expected", [
    (0, 0, 0, (0.0, 0.0, 0.0)),
    (5, 0, 0, (1.0, 1.0, 1.0)),
    (0, 5, 0, (0.0, 0.0, 0.0)),
    (3, 1, 1, (0.75, 0.75, 0.75)),
])
def test_precision_recall_f1(tp, fp, fn, expected):
    assert evaluate.precision_recall_f1(tp, fp, fn) == pytest.approx(expected)
