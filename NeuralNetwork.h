#pragma once
#include "HiddenLayer.h"
#include "OutputLayer.h"

class NeuralNetwork {
private:
    HiddenLayer hidden;   // 784 inputs → 128 neurons
    OutputLayer output;   // 128 inputs → 10 neurons

public:
    // Constructor — builds both layers with correct sizes
    NeuralNetwork();

    // Takes 784 pixel values, returns predicted digit 0-9
    int predict(const vector<double>& input);
};