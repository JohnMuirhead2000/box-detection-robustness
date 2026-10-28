cmake_minimum_required(VERSION 3.14)
project(box_detection_iou CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(iou iou.cpp)
target_include_directories(iou PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(iou PRIVATE -Wall -Wextra -Wpedantic)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(test_iou test_iou.cpp)
target_link_libraries(test_iou PRIVATE iou GTest::gtest_main)
target_compile_options(test_iou PRIVATE -Wall -Wextra -Wpedantic)

include(GoogleTest)
gtest_discover_tests(test_iou)

add_executable(nms_filter nms_filter.cpp)
target_link_libraries(nms_filter PRIVATE iou)
target_compile_options(nms_filter PRIVATE -Wall -Wextra -Wpedantic)
