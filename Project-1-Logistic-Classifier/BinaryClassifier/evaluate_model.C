/*
 * 4th step in the process: evaluate the model performance
*/

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TPaveText.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TMath.h"
#include <vector>
#include <iostream>
#include "logistic_model.h"  // Ensure you include the model header

void plotROC(std::vector<int>& trueLabels, std::vector<float>& predictedScores) {
    int n = trueLabels.size();
    std::vector<float> fpr, tpr;  // false positive rate, true positive rate
    int pos = 0, neg = 0;

    for (int i = 0; i < n; ++i) {
        if (trueLabels[i] == 1) pos++;
        else neg++;
    }

    // Threshold values
    std::vector<float> thresholds = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    for (auto& threshold : thresholds) {
        float tp = 0, fp = 0, fn = 0, tn = 0;
        for (int i = 0; i < n; ++i) {
            if (predictedScores[i] > threshold) {
                if (trueLabels[i] == 1) tp++;
                else fp++;
            } else {
                if (trueLabels[i] == 1) fn++;
                else tn++;
            }
        }

        // Calculate true positive rate (sensitivity) and false positive rate (1-specificity)
        fpr.push_back(fp / float(fp + tn));
        tpr.push_back(tp / float(tp + fn));
    }

    // Create and draw ROC curve
    TGraph* rocGraph = new TGraph(fpr.size(), &fpr[0], &tpr[0]);
    TCanvas* c = new TCanvas("rocCanvas", "ROC Curve", 800, 600);
    rocGraph->SetTitle("ROC Curve;False Positive Rate;True Positive Rate");
    rocGraph->SetLineColor(kBlue);
    rocGraph->SetLineWidth(2);
    rocGraph->Draw("AL");

    // Draw diagonal (random classifier line)
    TGraph* diagGraph = new TGraph(2);
    diagGraph->SetPoint(0, 0, 0);
    diagGraph->SetPoint(1, 1, 1);
    diagGraph->SetLineColor(kRed);
    diagGraph->Draw("L");
}

void evaluateModel(const char* filename) {
    // Open the ROOT file with the trained model and test dataset
    TFile* file = TFile::Open(filename);
    TTree* tree;
    file->GetObject("tree", tree);

    // Variables to store data
    float feature1, feature2, feature3, feature4;
    int trueLabel;
    tree->SetBranchAddress("feature1", &feature1);
    tree->SetBranchAddress("feature2", &feature2);
    tree->SetBranchAddress("feature3", &feature3);
    tree->SetBranchAddress("feature4", &feature4);
    tree->SetBranchAddress("label", &trueLabel);

    // Create vectors to hold true labels and predicted scores
    std::vector<int> trueLabels;
    std::vector<float> predictedScores;

    // Load the trained model
    LogisticModel model(4);  // Adjust based on the number of features
    model.load("logistic_model.root");

    // Loop through the dataset and apply the model to get predictions
    int nEntries = tree->GetEntries();
    for (int i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);

        // Construct the feature vector
        std::vector<double> features = {feature1, feature2, feature3, feature4};

        // Calculate the prediction using the logistic regression model
        float prediction = model.predict(features);

        // Store the true label and predicted score
        trueLabels.push_back(trueLabel);
        predictedScores.push_back(prediction);
    }

    // Call ROC plotting function
    plotROC(trueLabels, predictedScores);

    // Optionally, print other metrics (accuracy, precision, recall)
    // Implement accuracy, precision, recall, etc., as needed
    float accuracy = 0;
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        if (trueLabels[i] == (predictedScores[i] > 0.5 ? 1 : 0)) {  // Assuming 0.5 as threshold
            accuracy++;
        }
    }
    accuracy /= trueLabels.size();
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    //
    // Histogram of Predicted Scores
    //
    TCanvas* cHist = new TCanvas("cHist", "Predicted Scores", 800, 600);
    TH1F* hPredictedScores = new TH1F("hPredictedScores", "Predicted Scores", 100, 0, 1);
    for (size_t i = 0; i < predictedScores.size(); ++i) {
        hPredictedScores->Fill(predictedScores[i]);
    }
    hPredictedScores->SetXTitle("Predicted Score (P(signal))");
    hPredictedScores->SetYTitle("Frequency");
    hPredictedScores->Draw();

    //
    // True vs Predicted Labels (Scatter Plot)
    //
    TCanvas* cScatter = new TCanvas("cScatter", "True vs Predicted", 800, 600);
    TGraph* trueVsPredicted = new TGraph(trueLabels.size());
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        trueVsPredicted->SetPoint(i, predictedScores[i], trueLabels[i]);
    }
    trueVsPredicted->SetTitle("True vs Predicted Labels;Predicted Score;True Label");
    trueVsPredicted->SetMarkerStyle(20);
    trueVsPredicted->Draw("AP");

    //
    // Accuracy Plot vs Threshold
    //
    std::vector<float> thresholds = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<float> accuracies;
    for (float threshold : thresholds) {
        int correct = 0;
        for (size_t i = 0; i < trueLabels.size(); ++i) {
            int predictedLabel = predictedScores[i] > threshold ? 1 : 0;
            if (predictedLabel == trueLabels[i]) {
                correct++;
            }
        }
        float accuracy = static_cast<float>(correct) / trueLabels.size();
        accuracies.push_back(accuracy);
    }

    // Plot accuracy vs threshold
    TCanvas* cAccuracy = new TCanvas("cAccuracy", "Accuracy vs Threshold", 800, 600);
    TGraph* accuracyGraph = new TGraph(thresholds.size(), &thresholds[0], &accuracies[0]);
    accuracyGraph->SetTitle("Accuracy vs Threshold;Threshold;Accuracy");
    accuracyGraph->SetLineColor(kBlue);
    accuracyGraph->Draw("AL");

    //
    // Confusion Matrix
    //
    int tp = 0, fp = 0, tn = 0, fn = 0;
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        int predictedLabel = predictedScores[i] > 0.5 ? 1 : 0;  // Assume threshold of 0.5
        if (predictedLabel == 1 && trueLabels[i] == 1) tp++;
        else if (predictedLabel == 1 && trueLabels[i] == 0) fp++;
        else if (predictedLabel == 0 && trueLabels[i] == 0) tn++;
        else if (predictedLabel == 0 && trueLabels[i] == 1) fn++;
    }

    std::cout << "Confusion Matrix: \n";
    std::cout << "TP: " << tp << " FP: " << fp << " TN: " << tn << " FN: " << fn << "\n";

    // Optionally, you can plot this as a 2D histogram or just print it

    //
    // Precision-Recall Curve
    //
    std::vector<float> precisions, recalls;
    for (auto threshold : thresholds) {
        int tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < trueLabels.size(); ++i) {
            int predictedLabel = predictedScores[i] > threshold ? 1 : 0;
            if (predictedLabel == 1) {
                if (trueLabels[i] == 1) tp++;
                else fp++;
            } else {
                if (trueLabels[i] == 1) fn++;
            }
        }

        float precision = (tp + fp) > 0 ? tp / float(tp + fp) : 0;
        float recall = (tp + fn) > 0 ? tp / float(tp + fn) : 0;

        precisions.push_back(precision);
        recalls.push_back(recall);
    }

    // Plot Precision-Recall curve
    TCanvas* cPR = new TCanvas("cPR", "Precision-Recall", 800, 600);
    TGraph* prGraph = new TGraph(thresholds.size(), &recalls[0], &precisions[0]);
    prGraph->SetTitle("Precision-Recall Curve;Recall;Precision");
    prGraph->SetLineColor(kGreen);
    prGraph->Draw("AL");

    //
    // Loss Curve (Optional if you have loss values saved)
    //
    // Assuming you have an array or vector of loss values
    std::vector<float> lossValues = {0.68, 0.62, 0.58, 0.53, 0.50};  // Fill this with your loss values, Dummy data for now
    TCanvas* cLoss = new TCanvas("cLoss", "Loss Curve", 800, 600);
    TGraph* lossGraph = new TGraph(lossValues.size());
    for (size_t i = 0; i < lossValues.size(); ++i) {
        lossGraph->SetPoint(i, i, lossValues[i]);
    }
    lossGraph->SetTitle("Loss Curve;Epoch;Loss");
    lossGraph->SetLineColor(kRed);
    lossGraph->Draw("AL");


}

void evaluate_model() {
    // Evaluate the model on a test dataset
    evaluateModel("generated_dataset.root");
}
