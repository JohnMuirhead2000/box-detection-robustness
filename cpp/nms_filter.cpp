// Command-line NMS over a predictions CSV, so the C++ implementation is
// part of the evaluation pipeline rather than a standalone demo.
//
// Usage: nms_filter <input.csv> <output.csv> <iou_threshold>
//
// Input and output share the layout written by src/run_baseline.py:
//   image_name,x_min,y_min,x_max,y_max,confidence,class_name
// Rows are grouped by image_name, suppressed per image, and the survivors
// are written in the input order of their image's first appearance.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "iou.hpp"

namespace {

struct Row {
    std::string image_name;
    boxdet::Detection detection;
    std::string class_name;
};

bool parse_row(const std::string& line, Row& row) {
    std::stringstream stream(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    if (fields.size() != 7) {
        return false;
    }
    row.image_name = fields[0];
    row.detection.box = {std::stod(fields[1]), std::stod(fields[2]),
                         std::stod(fields[3]), std::stod(fields[4])};
    row.detection.confidence = std::stod(fields[5]);
    row.class_name = fields[6];
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: nms_filter <input.csv> <output.csv> <iou_threshold>\n";
        return 2;
    }
    const double iou_threshold = std::strtod(argv[3], nullptr);

    std::ifstream input(argv[1]);
    if (!input) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 1;
    }
    std::string header;
    std::getline(input, header);

    // Preserve first-seen image order so output is diff-friendly.
    std::vector<std::string> image_order;
    std::map<std::string, std::vector<Row>> rows_by_image;
    std::string line;
    while (std::getline(input, line)) {
        Row row;
        if (!parse_row(line, row)) {
            std::cerr << "skipping malformed line: " << line << "\n";
            continue;
        }
        if (rows_by_image.find(row.image_name) == rows_by_image.end()) {
            image_order.push_back(row.image_name);
        }
        rows_by_image[row.image_name].push_back(row);
    }

    std::ofstream output(argv[2]);
    output << header << "\n";
    std::size_t kept_total = 0, input_total = 0;
    for (const std::string& image_name : image_order) {
        const std::vector<Row>& rows = rows_by_image[image_name];
        std::vector<boxdet::Detection> detections;
        for (const Row& row : rows) {
            detections.push_back(row.detection);
        }
        input_total += rows.size();
        for (const std::size_t index :
             boxdet::non_max_suppression(detections, iou_threshold)) {
            const Row& row = rows[index];
            output << row.image_name << "," << row.detection.box.x_min << ","
                   << row.detection.box.y_min << "," << row.detection.box.x_max
                   << "," << row.detection.box.y_max << ","
                   << row.detection.confidence << "," << row.class_name << "\n";
            ++kept_total;
        }
    }
    std::cerr << "kept " << kept_total << " of " << input_total
              << " detections at IoU " << iou_threshold << "\n";
    return 0;
}
