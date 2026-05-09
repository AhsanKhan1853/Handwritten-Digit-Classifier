#pragma once
#include <vector>
using namespace std;

class Layer {
protected:
    // protected means child classes can access these directly
    vector<vector<double>> weights;
    vector<double> bias;
    vector<double> output;

public:
    // Constructor — takes input and output size
    Layer(int inputSize, int outputSize);

    // Pure virtual — so for activation logic we can write different code in HiddenLayer vs OutputLayer 
    virtual void activate(const vector<double>& input) = 0;

    // Getter so next layer can read this layer's output
    const vector<double>& getOutput() const { return output; }
};