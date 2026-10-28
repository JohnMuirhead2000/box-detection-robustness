#include "iou.hpp"

#include <algorithm>
#include <numeric>

namespace boxdet {

namespace {

double area(const Box& box) {
    const double width = box.x_max - box.x_min;
    const double height = box.y_max - box.y_min;
    if (width <= 0.0 || height <= 0.0) {
        return 0.0;
    }
    return width * height;
}

}  // namespace

double intersection_over_union(const Box& a, const Box& b) {
    const double inter_width =
        std::min(a.x_max, b.x_max) - std::max(a.x_min, b.x_min);
    const double inter_height =
        std::min(a.y_max, b.y_max) - std::max(a.y_min, b.y_min);
    if (inter_width <= 0.0 || inter_height <= 0.0) {
        return 0.0;
    }
    const double intersection = inter_width * inter_height;
    const double union_area = area(a) + area(b) - intersection;
    if (union_area <= 0.0) {
        return 0.0;
    }
    return intersection / union_area;
}

std::vector<std::size_t> non_max_suppression(
    const std::vector<Detection>& detections, double iou_threshold) {
    std::vector<std::size_t> order(detections.size());
    std::iota(order.begin(), order.end(), 0);
    // Stable sort so equal confidences keep input order (deterministic).
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t lhs, std::size_t rhs) {
                         return detections[lhs].confidence >
                                detections[rhs].confidence;
                     });

    std::vector<std::size_t> kept;
    for (const std::size_t candidate : order) {
        const bool suppressed = std::any_of(
            kept.begin(), kept.end(), [&](std::size_t keeper) {
                return intersection_over_union(detections[candidate].box,
                                               detections[keeper].box) >=
                       iou_threshold;
            });
        if (!suppressed) {
            kept.push_back(candidate);
        }
    }
    return kept;
}

}  // namespace boxdet
