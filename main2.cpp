#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>


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
            temp += bias[i];            // Add the bias to the weighted sum
            output = max(0.0,temp);     // Apply the ReLU activation function to the weighted sum to get the output of the neuron
          hidden_layer[i] = output;   // Store the output of the neuron in the hidden layer vector    
        }

     vector<vector<double>> outer_weights(10, vector<double>(128));
        for (int i = 0; i < 10; i++){
            for (int j = 0; j < 128; j++)
            {
                double randomWeight2 = ((double)rand() / RAND_MAX) - 0.5;
                outer_weights[i][j] = randomWeight2;
            }
        }
   
    vector<double>output_layer(10);
    for(int i=0;i<10;i++){
        double output;
        double temp = 0.0;
        for(int j=0;j<128;j++){
            temp += hidden_layer[j] *outer_weights[i][j];

        }
        temp += bias[i];
        double sum =0;
        
        output_layer[i]= temp;
        for(int k=0;k<10;k++){
            sum += exp(output_layer[k]);
         }
        output_layer[i] = exp(output_layer[i])/sum; // Apply the softmax activation function to the output layer to get the final output probabilities for each class
        
    }

    
    
    return 0;
}