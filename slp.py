"""
Single Layer Perceptron (SLP) / Softmax Regression from Scratch on Iris Dataset
Author: Ved Patel
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def softmax(z):
    """Numerically stable softmax activation function."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy loss with epsilon clipping to prevent log(0)."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def main():
    # 1. Load Iris dataset
    iris = load_iris()
    X = iris.data      # Shape: (150, 4) - sepal length, sepal width, petal length, petal width
    y = iris.target    # Shape: (150,) - 0 = setosa, 1 = versicolor, 2 = virginica

    # 2. One-hot encode class labels
    num_classes = 3
    y_one_hot = np.eye(num_classes)[y]   # Shape: (150, 3)

    # 3. Stratified split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_features = X_train.shape[1]
    n_samples = X_train.shape[0]

    # 5. Initialize weights and biases
    np.random.seed(0)
    W = np.random.randn(num_features, num_classes) * 0.01  # Shape: (4, 3)
    b = np.zeros((1, num_classes))                         # Shape: (1, 3)

    # 6. Training Hyperparameters
    lr = 0.1
    epochs = 3000

    print("=" * 60)
    print("Training Single Layer Perceptron (SLP) on Iris Dataset")
    print("=" * 60)

    # 7. Training Loop (Batch Gradient Descent)
    for epoch in range(epochs):
        # Forward pass: linear combination followed by softmax
        z = np.dot(X_train, W) + b
        A = softmax(z)

        # Loss calculation
        loss = categorical_cross_entropy(y_train, A)

        # Analytical gradients via backpropagation
        dz = (A - y_train) / n_samples
        dW = np.dot(X_train.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)

        # Gradient descent update
        W -= lr * dW
        b -= lr * db

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:4d}/{epochs} - Loss: {loss:.4f}")

    print("\nFinal Gradients:")
    print("dW:\n", dW)
    print("db:\n", db)

    # 8. Evaluation on Test Set
    z_test = np.dot(X_test, W) + b
    A_test = softmax(z_test)

    pred = np.argmax(A_test, axis=1)
    true = np.argmax(y_test, axis=1)
    test_accuracy = np.mean(pred == true) * 100

    train_pred = np.argmax(softmax(np.dot(X_train, W) + b), axis=1)
    train_true = np.argmax(y_train, axis=1)
    train_accuracy = np.mean(train_pred == train_true) * 100

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Test accuracy:  {test_accuracy:.2f}%")
    print(f"Train accuracy: {train_accuracy:.2f}%")

    # Sample single flower prediction inspection
    i = 0
    print("\nSample Test Prediction (Index 0):")
    print("True species:     ", iris.target_names[true[i]])
    print("Predicted probs:  ", np.round(A_test[i], 4))
    print("Predicted species:", iris.target_names[pred[i]])


if __name__ == "__main__":
    main()
