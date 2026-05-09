#pragma once
#include "Layer.h"

class OutputLayer : public Layer {
public:
    // Constructor passes values up to Layer's constructor
    OutputLayer(int inputSize, int outputSize) : Layer(inputSize, outputSize) {}

    // Override activate() with Softmax logic
    void activate(const vector<double>& input) override;
};