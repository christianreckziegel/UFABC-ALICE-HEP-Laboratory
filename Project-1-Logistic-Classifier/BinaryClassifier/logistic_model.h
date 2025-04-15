#ifndef LOGISTIC_MODEL_H
#define LOGISTIC_MODEL_H

#include <vector>
#include <fstream>
#include <iostream>
#include "TVectorD.h"
#include "TMath.h"
#include "TFile.h"

class LogisticModel {
public:
    LogisticModel(int nFeatures) : weights(nFeatures + 1) {}  // +1 for bias

    double sigmoid(double z) const {
        return 1.0 / (1.0 + TMath::Exp(-z));
    }

    double predict(const std::vector<double>& x) const {
        double z = weights[0]; // bias
        for (size_t i = 0; i < x.size(); ++i)
            z += weights[i + 1] * x[i];
        return sigmoid(z);
    }

    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& Y,
               int maxIter = 1000, double lr = 0.01) {
        size_t n = X.size();
        size_t d = X[0].size();

        // Ensure weights are sized correctly for the number of features
        weights.ResizeTo(d + 1); // +1 for bias

        for (int iter = 0; iter < maxIter; ++iter) {
            std::vector<double> grad(d + 1, 0.0);
            for (size_t i = 0; i < n; ++i) {
                double z = weights[0];
                for (size_t j = 0; j < d; ++j)
                    z += weights[j + 1] * X[i][j];
                double p = sigmoid(z);
                double error = Y[i] - p;

                grad[0] += error;
                for (size_t j = 0; j < d; ++j)
                    grad[j + 1] += error * X[i][j];
            }
            for (size_t j = 0; j <= d; ++j)
                weights[j] += lr * grad[j];
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
