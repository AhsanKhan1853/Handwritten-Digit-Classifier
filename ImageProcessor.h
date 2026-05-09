#pragma once
#include <vector>
#include <string>
using namespace std;

class ImageProcessor {
private:
    vector<double> pixels; // 784 normalized values

public:
    // Loads image, converts to grayscale, normalizes
    bool load(const string& path);

    // Getter — returns the 784 pixel values
    const vector<double>& getPixels() const { return pixels; }
};