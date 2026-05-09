#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "ImageProcessor.h"
#include <iostream>

bool ImageProcessor::load(const string& path) {
    int w, h, c;
    
    // Load raw image bytes
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 0);

    if (!data) {
        cout << "Failed to load image\n";
        return false;
    }

    if (w != 28 || h != 28) {
        cout << "Image must be 28x28 pixels\n";
        stbi_image_free(data);
        return false;
    }

    pixels.resize(784);

    // Convert each pixel to grayscale and normalize
    for (int i = 0; i < 784; i++) {
        double gray;

        if (c == 1)
            // Already grayscale
            gray = data[i];
        else
            // RGB to grayscale using luminance formula
            gray = 0.299 * data[i*c] + 0.587 * data[i*c+1] + 0.114 * data[i*c+2];

        // Normalize 0-255 to 0.0-1.0
        pixels[i] = gray / 255.0;
    }

    stbi_image_free(data);
    return true;
}