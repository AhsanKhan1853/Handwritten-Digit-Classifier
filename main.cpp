#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include "NeuralNetwork.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
using namespace std;

// ═══════════════════════════════════════
// LAYER
// ═══════════════════════════════════════
Layer::Layer(int inputSize, int outputSize) {
    output.resize(outputSize, 0.0);
    bias.resize(outputSize, 0.0);
    delta.resize(outputSize, 0.0);
    weights.resize(outputSize, vector<double>(inputSize));
    double scale = sqrt(2.0 / inputSize);
    for (int i = 0; i < outputSize; i++)
        for (int j = 0; j < inputSize; j++)
            weights[i][j] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * scale;
}

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// ═══════════════════════════════════════
// HIDDEN LAYER — Sigmoid
// ═══════════════════════════════════════
void HiddenLayer::activate(const vector<double>& input) {
    lastInput = input;
    for (int i = 0; i < (int)output.size(); i++) {
        double temp = bias[i];
        for (int j = 0; j < (int)input.size(); j++)
            temp += input[j] * weights[i][j];
        output[i] = sigmoid(temp);
    }
}

// ═══════════════════════════════════════
// OUTPUT LAYER — Softmax
// ═══════════════════════════════════════
void OutputLayer::activate(const vector<double>& input) {
    lastInput = input;
    double maxVal = 0.0;
    for (int i = 0; i < (int)output.size(); i++) {
        double temp = bias[i];
        for (int j = 0; j < (int)input.size(); j++)
            temp += input[j] * weights[i][j];
        output[i] = temp;
        if (temp > maxVal) maxVal = temp;
    }
    double sum = 0.0;
    for (int i = 0; i < (int)output.size(); i++) {
        output[i] = exp(output[i] - maxVal);
        sum += output[i];
    }
    for (int i = 0; i < (int)output.size(); i++)
        output[i] /= sum;
}

// ═══════════════════════════════════════
// NEURAL NETWORK
// ═══════════════════════════════════════
NeuralNetwork::NeuralNetwork() : hidden1(784, 128), output(128, 10) {}

int NeuralNetwork::predict(const vector<double>& input) {
    hidden1.activate(input);
    output.activate(hidden1.getOutput());
    const vector<double>& probs = output.getOutput();
    int best = 0;
    for (int i = 1; i < 10; i++)
        if (probs[i] > probs[best]) best = i;
    return best;
}

void NeuralNetwork::train(const vector<double>& input, int correctLabel, double lr) {
    hidden1.activate(input);
    output.activate(hidden1.getOutput());

    vector<double> target(10, 0.0);
    target[correctLabel] = 1.0;

    for (int i = 0; i < 10; i++)
        output.delta[i] = output.output[i] - target[i];

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 128; j++)
            output.weights[i][j] -= lr * output.delta[i] * output.lastInput[j];
        output.bias[i] -= lr * output.delta[i];
    }

    for (int j = 0; j < 128; j++) {
        double error = 0.0;
        for (int i = 0; i < 10; i++)
            error += output.delta[i] * output.weights[i][j];
        hidden1.delta[j] = error * hidden1.output[j] * (1.0 - hidden1.output[j]);
    }

    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 784; j++)
            hidden1.weights[i][j] -= lr * hidden1.delta[i] * hidden1.lastInput[j];
        hidden1.bias[i] -= lr * hidden1.delta[i];
    }
}

void NeuralNetwork::saveModel(const string& path) {
    ofstream file(path, ios::binary);
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 784; j++)
            file.write((char*)&hidden1.weights[i][j], sizeof(double));
        file.write((char*)&hidden1.bias[i], sizeof(double));
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 128; j++)
            file.write((char*)&output.weights[i][j], sizeof(double));
        file.write((char*)&output.bias[i], sizeof(double));
    }
    file.close();
    cout << "Model saved!" << endl;
}

bool NeuralNetwork::loadModel(const string& path) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) return false;
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 784; j++)
            file.read((char*)&hidden1.weights[i][j], sizeof(double));
        file.read((char*)&hidden1.bias[i], sizeof(double));
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 128; j++)
            file.read((char*)&output.weights[i][j], sizeof(double));
        file.read((char*)&output.bias[i], sizeof(double));
    }
    file.close();
    cout << "Model loaded!" << endl;
    return true;
}

// ═══════════════════════════════════════
// DATA LOADER
// ═══════════════════════════════════════
DataLoader::DataLoader() : count(0) {}

bool DataLoader::loadImages(const string& path, int numImages) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) { cout << "Cannot open: " << path << endl; return false; }
    unsigned char buffer[10];
    for (int i = 0; i < 10; i++) buffer[i] = (unsigned char)file.get();
    int headerLen = buffer[8] + (buffer[9] * 256);
    for (int i = 0; i < headerLen; i++) file.get();
    count = numImages;
    images.resize(count, vector<double>(784));
    for (int i = 0; i < count; i++)
        for (int j = 0; j < 784; j++)
            images[i][j] = (unsigned char)file.get() / 255.0;
    file.close();
    cout << "Loaded " << count << " images!" << endl;
    return true;
}

bool DataLoader::loadLabels(const string& path, int numLabels) {
    ifstream file(path, ios::binary);
    if (!file.is_open()) { cout << "Cannot open: " << path << endl; return false; }
    unsigned char buffer[10];
    for (int i = 0; i < 10; i++) buffer[i] = (unsigned char)file.get();
    int headerLen = buffer[8] + (buffer[9] * 256);
    for (int i = 0; i < headerLen; i++) file.get();
    labels.resize(numLabels);
    for (int i = 0; i < numLabels; i++)
        labels[i] = (int)(unsigned char)file.get();
    file.close();
    cout << "Loaded " << numLabels << " labels!" << endl;
    return true;
}

// ═══════════════════════════════════════
// IMAGE PREPROCESSING HELPERS
// ═══════════════════════════════════════

// Thickens thin strokes to match MNIST style
vector<double> dilate(const vector<double>& pixels) {
    vector<double> result(784, 0.0);
    for (int row = 0; row < 28; row++)
        for (int col = 0; col < 28; col++) {
            double maxVal = 0.0;
            for (int dr = -1; dr <= 1; dr++)
                for (int dc = -1; dc <= 1; dc++) {
                    int r = row + dr;
                    int c = col + dc;
                    if (r >= 0 && r < 28 && c >= 0 && c < 28)
                        if (pixels[r*28+c] > maxVal)
                            maxVal = pixels[r*28+c];
                }
            result[row*28+col] = maxVal;
        }
    return result;
}

// Centers digit in the 28x28 frame
vector<double> centerDigit(const vector<double>& pixels) {
    int minRow=27, maxRow=0, minCol=27, maxCol=0;
    for (int r = 0; r < 28; r++)
        for (int c = 0; c < 28; c++)
            if (pixels[r*28+c] > 0.1) {
                if (r < minRow) minRow = r;
                if (r > maxRow) maxRow = r;
                if (c < minCol) minCol = c;
                if (c > maxCol) maxCol = c;
            }

    // nothing found — return as is
    if (minRow > maxRow) return pixels;

    int shiftRow = 14 - (minRow + maxRow) / 2;
    int shiftCol = 14 - (minCol + maxCol) / 2;

    vector<double> result(784, 0.0);
    for (int r = 0; r < 28; r++)
        for (int c = 0; c < 28; c++) {
            int nr = r + shiftRow;
            int nc = c + shiftCol;
            if (nr >= 0 && nr < 28 && nc >= 0 && nc < 28)
                result[nr*28+nc] = pixels[r*28+c];
        }
    return result;
}

// ═══════════════════════════════════════
// MAIN
// ═══════════════════════════════════════
int main() {
    srand(42);

    NeuralNetwork nn;

    if (nn.loadModel("model.bin")) {
        cout << "Loaded existing model!" << endl;
    } else {
        cout << "Training from scratch..." << endl;

        DataLoader loader;
        if (!loader.loadImages("train_images.npy", 60000)) return 1;
        if (!loader.loadLabels("train_labels.npy", 60000)) return 1;

        double lr = 0.01;
        int epochs = 3;

        for (int e = 0; e < epochs; e++) {
            // shuffle each epoch
            vector<int> order(loader.getCount());
            for (int i = 0; i < loader.getCount(); i++) order[i] = i;
            for (int i = loader.getCount()-1; i > 0; i--) {
                int j = rand() % (i+1);
                swap(order[i], order[j]);
            }

            for (int idx = 0; idx < loader.getCount(); idx++)
                nn.train(loader.getImages()[order[idx]],
                         loader.getLabels()[order[idx]], lr);

            // accuracy check after epoch
            int correct = 0;
            for (int i = 0; i < loader.getCount(); i++)
                if (nn.predict(loader.getImages()[i]) == loader.getLabels()[i])
                    correct++;

            cout << "=== Epoch " << e+1 << " | Accuracy: "
                 << (correct * 100.0 / loader.getCount()) << "% ===" << endl;

            lr *= 0.95;
        }
        nn.saveModel("model.bin");
    }

    // ── Test on unseen data ──
    DataLoader testLoader;
    if (!testLoader.loadImages("test_images.npy", 10000)) return 1;
    if (!testLoader.loadLabels("test_labels.npy", 10000)) return 1;

    int confusion[10][10] = {};
    int correct = 0;
    for (int i = 0; i < 10000; i++) {
        int actual    = testLoader.getLabels()[i];
        int predicted = nn.predict(testLoader.getImages()[i]);
        confusion[actual][predicted]++;
        if (predicted == actual) correct++;
    }

    cout << "\n========================================\n";
    cout << "           RESULTS SUMMARY\n";
    cout << "========================================\n";
    cout << "Correct       : " << correct << " / 10000\n";
    cout << "Test Accuracy : " << (correct * 100.0 / 10000) << "%\n";

    cout << "\n--- Accuracy Per Digit ---\n";
    for (int i = 0; i < 10; i++) {
        int total = 0;
        for (int j = 0; j < 10; j++) total += confusion[i][j];
        cout << "Digit " << i << ": " << confusion[i][i]
             << "/" << total << " ("
             << (confusion[i][i] * 100.0 / total) << "%)\n";
    }

    cout << "\n--- Common Mistakes ---\n";
    for (int a = 0; a < 10; a++)
        for (int p = 0; p < 10; p++)
            if (a != p && confusion[a][p] >= 10)
                cout << "Digit " << a << " mistaken as "
                     << p << " : " << confusion[a][p] << " times\n";

    // ── Single image prediction ──
    string imagePath;
    cout << "\nEnter image path (or 'skip'): ";
    cin >> imagePath;

    if (imagePath != "skip") {
        int w, h, c;
        unsigned char* img = stbi_load(imagePath.c_str(), &w, &h, &c, 1);
        if (!img) { cout << "Could not load image!\n"; return 1; }

        // calculate average brightness to decide invert or not
        double avgBrightness = 0.0;
        for (int i = 0; i < w * h; i++)
            avgBrightness += img[i];
        avgBrightness /= (w * h);
        bool shouldInvert = (avgBrightness > 127.0);

        // resize to 28x28
        vector<double> pixels(784);
        for (int row = 0; row < 28; row++)
            for (int col = 0; col < 28; col++) {
                int srcRow = row * h / 28;
                int srcCol = col * w / 28;
                double pixel = img[srcRow * w + srcCol] / 255.0;
                pixels[row*28+col] = shouldInvert ? (1.0 - pixel) : pixel;
            }
        stbi_image_free(img);

        // preprocess to match MNIST style
        pixels = dilate(pixels);
        pixels = centerDigit(pixels);

        int result = nn.predict(pixels);
        cout << "================================\n";
        cout << " Predicted digit: " << result << "\n";
        cout << "================================\n";
    }

    return 0;
}