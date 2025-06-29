"""
Train script for Logistic Regression
"""

import argparse
import numpy as np
from logistic_regression import LogisticRegressionScratch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    model = LogisticRegressionScratch()
    model.fit(X, y)
    preds = model.predict(X)
    print("Training complete. Predictions:", preds)


if __name__ == "__main__":
    main()
