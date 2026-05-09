#include "Layer.h"
#include <cstdlib>

Layer::Layer(int inputSize, int outputSize) {
    
    // Size the output vector — one value per neuron in this layer
    output.resize(outputSize, 0.0);

    // Size the bias vector — one bias per neuron, initialized to 0
    bias.resize(outputSize, 0.0);

    // Size the weight matrix — outputSize rows, inputSize columns
    weights.resize(outputSize, vector<double>(inputSize));

    // Fill every weight with a random value between -0.5 and 0.5
    for (int i = 0; i < outputSize; i++)
        for (int j = 0; j < inputSize; j++)
            weights[i][j] = ((double)rand() / RAND_MAX) - 0.5;
}