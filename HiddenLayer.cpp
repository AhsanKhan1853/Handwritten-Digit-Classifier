#include "HiddenLayer.h"
#include <algorithm>

void HiddenLayer::activate(const vector<double>& input) {
    
    // Loop through every neuron in this layer
    for (int i = 0; i < output.size(); i++) {
        
        double temp = 0.0; // accumulator for weighted sum
        
        // Multiply every input by its corresponding weight and sum
        for (int j = 0; j < input.size(); j++)
            temp += input[j] * weights[i][j];
        
        // Add bias
        temp += bias[i];
        
        // Apply ReLU — if negative set to 0, else keep as is
        output[i] = max(0.0, temp);
    }
}