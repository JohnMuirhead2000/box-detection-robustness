#include <gtest/gtest.h>

#include "iou.hpp"

using boxdet::Box;
using boxdet::Detection;
using boxdet::intersection_over_union;
using boxdet::non_max_suppression;

TEST(IoU, IdenticalBoxesGiveOne) {
    const Box box{0, 0, 10, 10};
    EXPECT_DOUBLE_EQ(intersection_over_union(box, box), 1.0);
}

TEST(IoU, NormalOverlap) {
    // 10x10 and 10x10 offset by 5 in x: intersection 50, union 150.
    const Box a{0, 0, 10, 10};
    const Box b{5, 0, 15, 10};
    EXPECT_DOUBLE_EQ(intersection_over_union(a, b), 50.0 / 150.0);
    EXPECT_DOUBLE_EQ(intersection_over_union(b, a), 50.0 / 150.0);
}

TEST(IoU, NoOverlapGivesZero) {
    const Box a{0, 0, 10, 10};
    const Box b{20, 20, 30, 30};
    EXPECT_DOUBLE_EQ(intersection_over_union(a, b), 0.0);
}

TEST(IoU, TouchingEdgesGiveZero) {
    const Box a{0, 0, 10, 10};
    const Box right{10, 0, 20, 10};
    const Box below{0, 10, 10, 20};
    const Box corner{10, 10, 20, 20};
    EXPECT_DOUBLE_EQ(intersection_over_union(a, right), 0.0);
    EXPECT_DOUBLE_EQ(intersection_over_union(a, below), 0.0);
    EXPECT_DOUBLE_EQ(intersection_over_union(a, corner), 0.0);
}

TEST(IoU, ContainedBoxIsAreaRatio) {
    const Box outer{0, 0, 10, 10};
    const Box inner{2, 2, 4, 4};  // area 4 inside area 100
    EXPECT_DOUBLE_EQ(intersection_over_union(outer, inner), 0.04);
}

TEST(IoU, DegenerateBoxesGiveZero) {
    const Box empty{5, 5, 5, 5};
    const Box box{0, 0, 10, 10};
    EXPECT_DOUBLE_EQ(intersection_over_union(empty, box), 0.0);
    EXPECT_DOUBLE_EQ(intersection_over_union(empty, empty), 0.0);
}

TEST(NMS, EmptyInputGivesEmptyOutput) {
    EXPECT_TRUE(non_max_suppression({}, 0.5).empty());
}

TEST(NMS, KeepsHighestOfOverlappingPair) {
    const std::vector<Detection> detections{
        {{0, 0, 10, 10}, 0.6},
        {{1, 1, 11, 11}, 0.9},  // overlaps the first well above 0.5
    };
    const auto kept = non_max_suppression(detections, 0.5);
    ASSERT_EQ(kept.size(), 1u);
    EXPECT_EQ(kept[0], 1u);
}

TEST(NMS, KeepsNonOverlappingBoxesInConfidenceOrder) {
    const std::vector<Detection> detections{
        {{0, 0, 10, 10}, 0.3},
        {{50, 50, 60, 60}, 0.8},
        {{100, 100, 110, 110}, 0.5},
    };
    const auto kept = non_max_suppression(detections, 0.5);
    ASSERT_EQ(kept.size(), 3u);
    EXPECT_EQ(kept[0], 1u);
    EXPECT_EQ(kept[1], 2u);
    EXPECT_EQ(kept[2], 0u);
}

TEST(NMS, ThresholdIsInclusive) {
    // IoU exactly 1/3: suppressed at threshold 1/3, kept just above it.
    const std::vector<Detection> detections{
        {{0, 0, 10, 10}, 0.9},
        {{5, 0, 15, 10}, 0.8},
    };
    EXPECT_EQ(non_max_suppression(detections, 1.0 / 3.0).size(), 1u);
    EXPECT_EQ(non_max_suppression(detections, 1.0 / 3.0 + 1e-9).size(), 2u);
}

TEST(NMS, SuppressedBoxDoesNotSuppressOthers) {
    // c overlaps b but not a. b is suppressed by a, so c must survive.
    const std::vector<Detection> detections{
        {{0, 0, 10, 10}, 0.9},    // a
        {{6, 0, 16, 10}, 0.8},    // b: IoU with a = 0.25, suppressed at 0.2
        {{13, 0, 23, 10}, 0.7},   // c: IoU with b = 3/17, with a = 0
    };
    const auto kept = non_max_suppression(detections, 0.2);
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_EQ(kept[0], 0u);
    EXPECT_EQ(kept[1], 2u);
}

TEST(NMS, EqualConfidenceKeepsInputOrder) {
    const std::vector<Detection> detections{
        {{0, 0, 10, 10}, 0.5},
        {{0, 0, 10, 10}, 0.5},
    };
    const auto kept = non_max_suppression(detections, 0.5);
    ASSERT_EQ(kept.size(), 1u);
    EXPECT_EQ(kept[0], 0u);
}
