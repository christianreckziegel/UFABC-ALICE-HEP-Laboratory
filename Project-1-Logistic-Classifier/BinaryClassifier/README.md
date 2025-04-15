# Binary Classifier with Logistic Regression (ROOT-based)

This folder contains a minimal logistic regression implementation in C++ using ROOT libraries, designed to perform binary classification tasks.

## 📦 Files Overview

| File                  | Description |
|-----------------------|-------------|
| `generate_dataset.C`  | Generates synthetic data and stores it in `generated_dataset.root`. |
| `train_model.C`       | Loads data, trains a `LogisticModel`, and saves the trained model. |
| `evaluate_model.C`    | Loads the trained model and evaluates its performance on the dataset. |
| `apply_model.C`       | Applies a trained model to new data (optional script). |
| `logistic_model.h`    | Header file defining the `LogisticModel` class. |
| `logistic_model.root` | ROOT file containing the saved trained model. |
| `logistic_model.txt`  | Optional: stores human-readable weights of the model. |

---

## 🚀 How to Use the LogisticModel

The `LogisticModel` class is a simple logistic regression model that can be trained and applied to binary classification tasks.

### ✨ Class Interface

```cpp
class LogisticModel {
public:
    LogisticModel(int nFeatures);

    double sigmoid(double z) const;

    double predict(const std::vector<double>& x) const;

    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& Y,
               int maxIter = 1000,
               double lr = 0.01);

    void save(const char* filename) const;
    void load(const char* filename);

    void exportToText(const char* filename) const;

    TVectorD getWeights() const;

private:
    TVectorD weights;
};
```

## 🏋️‍♂️ Training a Model

To train a logistic regression model using your dataset, follow the steps below. This assumes that the dataset is already loaded into two vectors:

    X — a `std::vector<std::vector<double>>` representing the input features. For example
    X = {
        {0.1, 0.5},   // jet 1 (2 features: pT and eta)
        {0.3, 0.7},   // jet 2 (2 features: pT and eta)
        {0.8, 0.2},   // jet 3 (2 features: pT and eta)
        {0.9, 0.4}    // jet 4 (2 features: pT and eta)
    };

    Y — a `std::vector<double>` with labels 0.0 (class 0 = not signal) or 1.0 (class 1 = signal).

Fill the Input Feature Matrix X and the label vector Y 
(In the future, implement a way of passing TMatrixD and TVectorD ROOT objects for more memory efficient when dealing with large datasets.)
```cpp
for (const auto& jet : jets) {
    X.push_back({jetpt(), jet.eta()})
    if (jet.IsSignal()) {
        Y.push_back(1);
    } else {
        Y.push_back(0);
    }
    
}
```
Pass the number of features (i.e., the dimensionality of the input vector):
```cpp
int nFeatures = X[0].size(); // assuming X is non-empty
LogisticModel model(nFeatures);
```

Train the model

Use the train() method. You can set the number of iterations (maxIter) and learning rate (lr):
```cpp
int maxIter = 1000;
double learningRate = 0.01;

model.train(X, Y, maxIter, learningRate);
```

Under the hood:

    The model uses gradient ascent to optimize weights.

    A bias term is included as weights[0].

Save the trained model

To persist the model in a ROOT file:
```cpp
model.save("logistic_model.root");
```
You can also export weights to a human-readable .txt:
```cpp
model.exportToText("logistic_model.txt");
```

## 📊 Evaluating the Model

To evaluate the trained model on the same dataset:
```cpp
root -l -b -q evaluate_model.C
```
This script computes and prints the accuracy and confusion matrix.


## 📂 Model I/O

    Save model:
```cpp
model.save("logistic_model.root");
```
Load model:

    LogisticModel model(n_features);
    model.load("logistic_model.root");

## 🧪 Making Predictions

After loading a model, you can run:
```cpp
std::vector<double> x = {0.5, -1.2, 0.9}; // example feature vector
int label = model.predict(x);            // returns 0 or 1
double p = model.predict_proba(x);       // returns probability
```

## 📝 Dataset Format

The dataset is expected to be stored as a TTree in a ROOT file with the following branches:

    features (vector<double>) — Feature vector.

    label (int) — Target class (0 or 1).

You can generate the dataset using:
```cpp
root -l -b -q generate_dataset.C
```

## 💡 Notes

    Make sure all scripts include the logistic_model.h header.

    This binary setup can be expanded for multi-class classification using the One-vs-Rest (OvR) strategy.

    Compatible with ROOT 6+.

## ✍️ Author

Christian (UFABC-ALICE Team)