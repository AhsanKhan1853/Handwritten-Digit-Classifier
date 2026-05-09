#include "OutputLayer.h"
#include <cmath>

void OutputLayer::activate(const vector<double>& input) {

    double sum = 0.0; // will hold sum of all e^x values

    // First loop — calculate weighted sum for each output neuron
    for (int i = 0; i < output.size(); i++) {
        
        double temp = 0.0;
        
        for (int j = 0; j < input.size(); j++)
            temp += input[j] * weights[i][j];
        
        temp += bias[i];
        output[i] = temp;        // store raw value first
        sum += exp(temp);        // accumulate e^x for softmax denominator
    }

    // Second loop — divide each by sum to get probabilities
    for (int i = 0; i < output.size(); i++)
        output[i] = exp(output[i]) / sum;
}