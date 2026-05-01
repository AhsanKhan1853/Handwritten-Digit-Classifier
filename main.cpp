#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>

// Also write understandable comments in the code to explain what each part does.

using namespace std;

int main()
{

    vector<double> input(784);                                // Create a vector to hold the input data (28x28 pixels)
    vector<vector<double>> weights(128, vector<double>(784)); // Create a 2D vector to hold the weights for 128 neurons, each with 784 inputs
    vector<double>bias(128);

    // Initialize the weights with random values between -0.5 and 0.5
    srand(time(0)); // Seed the random number generator with the current time

    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            double randomWeight = ((double)rand() / RAND_MAX) - 0.5;
            weights[i][j] = randomWeight;
        }
    }
    
    vector<double> hidden_layer(128);

    for(int i = 0;i<128;i++){
            double output;
            double temp = 0.0; // Initialize the variable to hold the weighted sum of inputs for each neuron
        for(int j=0;j<784;j++){
 
                temp += input[j] * weights[i][j]; // Calculate the weighted sum of inputs for each neuron
                
            }
            temp += bias[i]; // Add the bias to the weighted sum
            output = 1/(1+exp(-temp));
            hidden_layer[i] = output;
        }  
    
    return 0;
}