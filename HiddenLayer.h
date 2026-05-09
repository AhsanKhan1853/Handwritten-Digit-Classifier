#pragma once
#include "Layer.h"

class HiddenLayer : public Layer {
public:
    // Constructor just passes values up to Layer's constructor
    HiddenLayer(int inputSize, int outputSize) : Layer(inputSize, outputSize) {}

    // Override activate() with ReLU logic
    void activate(const vector<double>& input) override;
};