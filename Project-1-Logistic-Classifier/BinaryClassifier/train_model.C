/*
 * 2nd step in the process: train a logistic regression model
*/

#include "logistic_model.h"
#include "TRandom3.h"
#include <vector>
#include <iostream>

// Function to train the logistic regression model
void train_model(int nSamples = 500, int nFeatures = 4, int maxIter = 1000, double lr = 0.01) {
    // Create random number generator
    TRandom3 rand(42);

    // Vectors to hold feature data and labels
    std::vector<std::vector<double>> X;
    std::vector<double> Y;

    // Generate synthetic data
    for (int i = 0; i < nSamples; ++i) {
        std::vector<double> features(nFeatures);
        
        // Create data: half the data is from a Gaussian with mean 2 (label 1), and the other half with mean -2 (label 0)
        if (i < nSamples / 2) {
            for (int j = 0; j < nFeatures; ++j)
                features[j] = rand.Gaus(2, 1);  // Gaussian with mean 2, stddev 1
            Y.push_back(1);
        } else {
            for (int j = 0; j < nFeatures; ++j)
                features[j] = rand.Gaus(-2, 1); // Gaussian with mean -2, stddev 1
            Y.push_back(0);
        }
        
        // Push the generated features into the dataset
        X.push_back(features);
    }

    // Initialize and train the logistic regression model
    LogisticModel model(nFeatures);
    model.train(X, Y, maxIter, lr);  // Train with the provided iterations and learning rate

    // Save the trained model
    model.save("logistic_model.root");

    // Export model weights to a text file
    model.exportToText("logistic_model.txt");

    // Output a message confirming model training and saving
    std::cout << "Trained and saved model with " << nFeatures << " features.\n";
}

