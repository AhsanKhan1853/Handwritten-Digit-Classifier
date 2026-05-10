#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std;

// ═══════════════════════════════════════════
// ABSTRACT BASE CLASS
// ═══════════════════════════════════════════
class Layer {
protected:
    vector<vector<double>> weights;
    vector<double> bias;
    vector<double> output;

public:
    Layer(int inputSize, int outputSize);
    virtual void activate(const vector<double>& input) = 0;
    const vector<double>& getOutput() const { return output; }
};

// ═══════════════════════════════════════════
// HIDDEN LAYER — ReLU
// ═══════════════════════════════════════════
class HiddenLayer : public Layer {
public:
    HiddenLayer(int inputSize, int outputSize) : Layer(inputSize, outputSize) {}
    void activate(const vector<double>& input) override;
};

// ═══════════════════════════════════════════
// OUTPUT LAYER — Softmax
// ═══════════════════════════════════════════
class OutputLayer : public Layer {
public:
    OutputLayer(int inputSize, int outputSize) : Layer(inputSize, outputSize) {}
    void activate(const vector<double>& input) override;
};

// ═══════════════════════════════════════════
// NEURAL NETWORK
// ═══════════════════════════════════════════
class NeuralNetwork {
private:
    HiddenLayer hidden;
    OutputLayer output;

public:
    NeuralNetwork();
    int predict(const vector<double>& input);
};

// ═══════════════════════════════════════════
// DATA LOADER
// ═══════════════════════════════════════════
class DataLoader {
private:
    vector<vector<double>> images;
    vector<int> labels;
    int count;

public:
    DataLoader();
    bool loadImages(const string& path);
    bool loadLabels(const string& path);
    const vector<vector<double>>& getImages() const { return images; }
    const vector<int>& getLabels() const { return labels; }
    int getCount() const { return count; }
};