#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
using namespace std;

class Layer {
public:
    vector<vector<double>> weights;
    vector<double> bias;
    vector<double> output;
    vector<double> lastInput;
    vector<double> delta;

    Layer(int inputSize, int outputSize);
    virtual void activate(const vector<double>& input) = 0;
    const vector<double>& getOutput() const { return output; }
};

class HiddenLayer : public Layer {
public:
    HiddenLayer(int inputSize, int outputSize) : Layer(inputSize, outputSize) {}
    void activate(const vector<double>& input) override;
};

class OutputLayer : public Layer {
public:
    OutputLayer(int inputSize, int outputSize) : Layer(inputSize, outputSize) {}
    void activate(const vector<double>& input) override;
};

class NeuralNetwork {
public:
    HiddenLayer hidden1;
    OutputLayer output;

    NeuralNetwork();
    int predict(const vector<double>& input);
    void train(const vector<double>& input, int correctLabel, double lr);
    void saveModel(const string& path);
    bool loadModel(const string& path);
};

class DataLoader {
private:
    vector<vector<double>> images;
    vector<int> labels;
    int count;
public:
    DataLoader();
    bool loadImages(const string& path, int numImages);
    bool loadLabels(const string& path, int numLabels);
    const vector<vector<double>>& getImages() const { return images; }
    const vector<int>& getLabels() const { return labels; }
    int getCount() const { return count; }
};