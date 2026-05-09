#include <iostream>
#include <cstdlib>
#include <ctime>
#include "NeuralNetwork.h"
#include "ImageProcessor.h"
using namespace std;

int main() {
    
    // Seed random once — affects all weight initialization
    srand(time(0));

    // Load and process image
    ImageProcessor img;
    if (!img.load("digit.png")) return 1;

    cout << "Image loaded successfully!" << endl;

    // Build network
    NeuralNetwork nn;

    // Predict
    int prediction = nn.predict(img.getPixels());

    cout << "Predicted digit: " << prediction << endl;

    return 0;
}