/*
 * 1st step in the process: generate a synthetic dataset for W jet identification
*/

#include <iostream>
#include <fstream>
#include <random>
#include "TMath.h"
#include "TFile.h"
#include "TTree.h"

// Function to generate the synthetic dataset
void generate_dataset(int nEvents = 1000, const char* outputFilename = "generated_dataset.root") {
    // Create a ROOT file to store the dataset
    TFile *outputFile = new TFile(outputFilename, "RECREATE");

    // Create a TTree to store the data
    TTree *tree = new TTree("tree", "Synthetic dataset for W jet identification");

    // Variables for the features and labels
    float feature1, feature2, feature3, feature4;
    int label;

    // Create branches for each feature and label
    tree->Branch("feature1", &feature1, "feature1/F");
    tree->Branch("feature2", &feature2, "feature2/F");
    tree->Branch("feature3", &feature3, "feature3/F");
    tree->Branch("feature4", &feature4, "feature4/F");
    tree->Branch("label", &label, "label/I");

    // Random number generators
    std::random_device rd;  // Seed for randomness
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<float> distribution(-5.0, 5.0); // For features
    std::bernoulli_distribution labelDist(0.5); // 50% probability for binary labels (0 or 1)

    // Fill the tree with synthetic data
    for (int i = 0; i < nEvents; ++i) {
        // Generate random feature values
        feature1 = distribution(generator);
        feature2 = distribution(generator);
        feature3 = distribution(generator);
        feature4 = distribution(generator);

        // Assign label (randomly 0 or 1 for this example)
        label = labelDist(generator);

        // Optionally, you could make labels depend on some of the features for realism
        // For example, create a correlation: if feature1 > 1, label = 1
        if (feature1 > 1) {
            label = 1;
        } else {
            label = 0;
        }

        // Fill the tree
        tree->Fill();
    }

    // Write the tree to the file
    tree->Write();

    // Close the ROOT file
    outputFile->Close();

    std::cout << "Dataset generated with " << nEvents << " events." << std::endl;
}
