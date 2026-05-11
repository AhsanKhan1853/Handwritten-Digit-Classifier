#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include "NeuralNetwork.h"
using namespace std;

// ═══════════════════════════════════════════
// LAYER
// ═══════════════════════════════════════════
Layer::Layer(int inputSize, int outputSize) {
    output.resize(outputSize, 0.0);
    bias.resize(outputSize, 0.0);
    weights.resize(outputSize, vector<double>(inputSize));
    for (int i = 0; i < outputSize; i++)
        for (int j = 0; j < inputSize; j++)
            weights[i][j] = ((double)rand() / RAND_MAX) - 0.5;
}

// ═══════════════════════════════════════════
// HIDDEN LAYER — ReLU
// ═══════════════════════════════════════════
void HiddenLayer::activate(const vector<double>& input) {
     lastInput = input;
    for (int i = 0; i < (int)output.size(); i++) {
        double temp = 0.0;
        for (int j = 0; j < (int)input.size(); j++)
            temp += input[j] * weights[i][j];
        temp += bias[i];
        output[i] = max(0.0, temp);
    }
}

// ═══════════════════════════════════════════
// OUTPUT LAYER — Softmax
// ═══════════════════════════════════════════
void OutputLayer::activate(const vector<double>& input) {
    lastInput = input;
    double sum = 0.0;
    for (int i = 0; i < (int)output.size(); i++) {
        double temp = 0.0;
        for (int j = 0; j < (int)input.size(); j++)
            temp += input[j] * weights[i][j];
        temp += bias[i];
        output[i] = temp;
        sum += exp(temp);
    }
    for (int i = 0; i < (int)output.size(); i++)
        output[i] = exp(output[i]) / sum;
}

// ═══════════════════════════════════════════
// NEURAL NETWORK
// ═══════════════════════════════════════════
NeuralNetwork::NeuralNetwork() : hidden(784, 128), output(128, 10) {}

int NeuralNetwork::predict(const vector<double>& input) {
    hidden.activate(input);
    output.activate(hidden.getOutput());
    const vector<double>& probs = output.getOutput();
    int best = 0;
    for (int i = 1; i < 10; i++)
        if (probs[i] > probs[best])
            best = i;
    return best;
}
void NeuralNetwork::train(const vector<double>& input, int correctLabel, double learningRate) {

    // ── Step 1: Forward pass ──
    hidden.activate(input);
    output.activate(hidden.getOutput());

    // ── Step 2: Build correct one-hot vector ──
    // All zeros except index of correct digit = 1
    vector<double> correct(10, 0.0);
    correct[correctLabel] = 1.0;

    // ── Step 3: Output layer error ──
    // error = predicted - correct
    vector<double> outputError(10);
    for (int i = 0; i < 10; i++)
        outputError[i] = output.getOutput()[i] - correct[i];

    // ── Step 4: Update output layer weights and biases ──
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 128; j++)
            output.weights[i][j] -= learningRate * outputError[i] * output.lastInput[j];
        output.bias[i] -= learningRate * outputError[i];
    }

    // ── Step 5: Pass error back to hidden layer ──
    // Each hidden neuron gets a share of the output error
    vector<double> hiddenError(128, 0.0);
    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < 10; i++)
            hiddenError[j] += outputError[i] * output.weights[i][j];

        // ReLU backprop — block error if neuron was not active
        if (hidden.getOutput()[j] <= 0)
            hiddenError[j] = 0.0;
    }

    // ── Step 6: Update hidden layer weights and biases ──
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 784; j++)
            hidden.weights[i][j] -= learningRate * hiddenError[i] * hidden.lastInput[j];
        hidden.bias[i] -= learningRate * hiddenError[i];
    }
}
// ═══════════════════════════════════════════
// DATA LOADER
// ═══════════════════════════════════════════
DataLoader::DataLoader() : count(0) {}

bool DataLoader::loadImages(const string& path) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        cout << "Cannot open: " << path << endl;
        return false;
    }

    // Read first 10 bytes
    unsigned char buffer[10];
    for (int i = 0; i < 10; i++)
        buffer[i] = (unsigned char)file.get();

    // Get header length from bytes 8 and 9
    int headerLen = buffer[8] + (buffer[9] * 256);

    // Skip remaining header bytes
    for (int i = 0; i < headerLen; i++)
        file.get();

    // Each pixel is 1 byte (uint8) — just read directly
    count = 60000;
    images.resize(count, vector<double>(784));

    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 784; j++) {
            unsigned char pixel = (unsigned char)file.get();
            images[i][j] = pixel / 255.0; // normalize to 0.0-1.0
        }
    }

    file.close();
    cout << "Images loaded successfully!" << endl;
    return true;
}

bool DataLoader::loadLabels(const string& path) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        cout << "Cannot open: " << path << endl;
        return false;
    }

    // Skip header same way
    unsigned char buffer[10];
    for (int i = 0; i < 10; i++)
        buffer[i] = (unsigned char)file.get();

    int headerLen = buffer[8] + (buffer[9] * 256);
    for (int i = 0; i < headerLen; i++)
        file.get();

    // Each label is 1 byte — digit 0 to 9
    labels.resize(60000);
    for (int i = 0; i < 60000; i++)
        labels[i] = (int)(unsigned char)file.get();

    file.close();
    cout << "Labels loaded successfully!" << endl;
    return true;
}

// ═══════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════
int main() {
    srand(time(0));

    // ── Load MNIST ──
    DataLoader loader;
    if (!loader.loadImages("train_images.npy")) return 1;
    if (!loader.loadLabels("train_labels.npy")) return 1;
    cout << "Total images loaded: " << loader.getCount() << endl;

    // ── Build Network ──
    NeuralNetwork nn;

    // ── Training Loop ──
    double learningRate = 0.01;
    int epochs = 3; // 3 passes through all 60000 images

    for (int e = 0; e < epochs; e++) {
        int correct = 0;

        for (int i = 0; i < loader.getCount(); i++) {
            // Train on this image
            nn.train(loader.getImages()[i], loader.getLabels()[i], learningRate);

            // Check prediction
            int prediction = nn.predict(loader.getImages()[i]);
            if (prediction == loader.getLabels()[i])
                correct++;

            // Print progress every 10000 images
            if ((i + 1) % 10000 == 0)
                cout << "Epoch " << e+1 << " | Image " << i+1
                     << " | Accuracy so far: "
                     << (correct * 100.0 / (i+1)) << "%" << endl;
        }

        cout << "=== Epoch " << e+1 << " complete | Accuracy: "
             << (correct * 100.0 / loader.getCount()) << "% ===" << endl;
    }

    return 0;
}