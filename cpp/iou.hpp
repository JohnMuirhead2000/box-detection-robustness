// Axis-aligned bounding-box IoU and non-maximum suppression.
//
// Mirrors the Python scorer in src/evaluate.py so the two can be checked
// against each other. Boxes are (x_min, y_min, x_max, y_max) in pixels.

#pragma once

#include <cstddef>
#include <vector>

namespace boxdet {

struct Box {
    double x_min;
    double y_min;
    double x_max;
    double y_max;
};

struct Detection {
    Box box;
    double confidence;
};

// Intersection area divided by union area. Returns 0 when the boxes do
// not overlap or when both have zero area.
double intersection_over_union(const Box& a, const Box& b);

// Greedy NMS: visit detections in descending confidence, keep one if it
// overlaps every kept detection by less than iou_threshold. Returns the
// indices into `detections` that survive, in the order they were kept.
std::vector<std::size_t> non_max_suppression(
    const std::vector<Detection>& detections, double iou_threshold);

}  // namespace boxdet
