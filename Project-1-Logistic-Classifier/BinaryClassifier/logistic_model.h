#ifndef LOGISTIC_MODEL_H
#define LOGISTIC_MODEL_H

#include <vector>
#include <fstream>
#include <iostream>
#include "TVectorD.h"
#include "TMath.h"
#include "TFile.h"

// Support function: sigmoid function
inline double sigmoid(double z) {
    return 1.0 / (1.0 + TMath::Exp(-z));
}

// Support function: compute the gradient of the log-likelihood function
inline void computeGradientStep(std::vector<double>& grad, const int& sampleSize, TVectorD& weights, const int& nFeatures, const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {

    // loop through sample instances
    for (size_t i = 0; i < sampleSize; ++i) {
        // Compute linear combination (dot product + bias)
        double z = weights[0];
        for (size_t j = 0; j < nFeatures; ++j) {
            z += weights[j + 1] * X[i][j];
        }
        // Apply sigmoid to get predicted probability (p = P(y=1 | x))
        double p = sigmoid(z);

        // Compute error: true label - predicted probability
        // This is the gradient component from the derivative of the log-likelihood
        double error = p - Y[i];

        // Accumulate gradient for bias term
        grad[0] += error;

        // Accumulate gradient for each weight
        for (size_t j = 0; j < nFeatures; ++j) {
            grad[j + 1] += error * X[i][j];
        }

        // Normalize gradient by sample size
        for (size_t j = 0; j <= nFeatures; ++j) {
            grad[j] /= sampleSize;
        }
    }

}

// LogisticModel class for binary classification using logistic regression
class LogisticModel {
public:
    LogisticModel(int nFeatures) : weights(nFeatures + 1) {}  // +1 for bias

    double predict(const std::vector<double>& x) const {
        double z = weights[0]; // bias
        for (size_t i = 0; i < x.size(); ++i)
            z += weights[i + 1] * x[i];
        return sigmoid(z);
    }

    // This function maximizes the log-likelihood of the logistic regression model via gradient descent
    // The gradient is derived from the cross-entropy loss, which is the negative log-likelihood.
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& Y,
               int maxIter = 1000, double lr = 0.01) {
        size_t sampleSize = X.size();
        size_t nFeatures = X[0].size();

        // Ensure weights vector is correctly sized: one for each feature + one for bias
        weights.ResizeTo(nFeatures + 1); // +1 for bias term (weights[0])
        
        // Perform gradient descent to maximize the log-likelihood function
        for (int iter = 0; iter < maxIter; ++iter) {
            std::vector<double> grad(nFeatures + 1, 0.0); // Initialize gradient accumulator
            
            // Compute gradient of the negative log-likelihood (cross-entropy loss) of current iteration
            computeGradientStep(grad, sampleSize, weights, nFeatures, X, Y);

            // Update weights using gradient descent
            // (we're maximizing the log-likelihood function)
            for (size_t j = 0; j <= nFeatures; ++j) {
                weights[j] = weights[j] - lr * grad[j];
            }
        }
    }

    void save(const char* filename) const {
        TFile f(filename, "RECREATE");
        weights.Write("weights");
        f.Close();
    }

    void load(const char* filename) {
        TFile f(filename, "READ");
        TVectorD* w = (TVectorD*)f.Get("weights");
        weights.ResizeTo(w->GetNrows());
        weights = *w;
        f.Close();
    }

    void exportToText(const char* filename) const {
        std::ofstream out(filename);
        if (!out.is_open()) {
            std::cerr << "Error opening file: " << filename << "\n";
            return;
        }

        // Write number of weights
        out << weights.GetNrows() << "\n";

        // Write each weight
        for (int i = 0; i < weights.GetNrows(); ++i)
            out << weights[i] << "\n";

        out.close();
        std::cout << "Model exported to " << filename << "\n";
    }

    TVectorD getWeights() const { return weights; }

private:
    TVectorD weights;
};


#endif
