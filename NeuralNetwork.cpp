#include "NeuralNetwork.h"
#include <algorithm>

// Constructor initializer list — builds both layers with correct sizes
NeuralNetwork::NeuralNetwork() : hidden(784, 128), output(128, 10) {}

int NeuralNetwork::predict(const vector<double>& input) {

    // Step 1 — pass input through hidden layer
    hidden.activate(input);

    // Step 2 — pass hidden layer output to output layer
    output.activate(hidden.getOutput());

    // Step 3 — find which neuron has highest probability
    const vector<double>& probs = output.getOutput();
    return max_element(probs.begin(), probs.end()) - probs.begin();
}