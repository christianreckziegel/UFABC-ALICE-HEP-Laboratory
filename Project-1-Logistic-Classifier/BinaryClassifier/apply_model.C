/*
 * 3rd step in the process: apply the trained model to new data
*/

#include "logistic_model.h"
#include "TFile.h"
#include "TTree.h"
#include <iostream>

void apply_model() {
    const int nFeatures = 4;

    // Initialize and load the model
    LogisticModel model(nFeatures);
    model.load("logistic_model.root");

    // Open the ROOT file containing new data
    TFile* f = TFile::Open("generated_dataset.root", "READ");
    if (!f || f->IsZombie()) {
        std::cerr << "Error: Could not open file 'new_data.root'.\n";
        return;
    }

    // Retrieve the tree from the file
    TTree* tree = (TTree*)f->Get("tree");
    if (!tree) {
        std::cerr << "Error: Could not find 'tree' in 'new_data.root'.\n";
        f->Close();
        return;
    }

    // Declare variables to hold feature data
    float feature1, feature2, feature3, feature4;

    // Set branch addresses to link tree columns to feature variables
    tree->SetBranchAddress("feature1", &feature1);
    tree->SetBranchAddress("feature2", &feature2);
    tree->SetBranchAddress("feature3", &feature3);
    tree->SetBranchAddress("feature4", &feature4);

    // Loop over the tree entries
    for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
        tree->GetEntry(i);
        
        // Prepare the feature vector for prediction
        std::vector<double> x = {feature1, feature2, feature3, feature4};
        
        // Predict the probability of the signal
        double prob = model.predict(x);
        
        // Output the prediction for the current entry
        std::cout << "Entry " << i << ": P(signal) = " << prob << "\n";
    }

    // Close the ROOT file
    f->Close();
}
