"""
KMeansScratch - from scratch implementation of K Means.
"""

import numpy as np


class KMeansScratch:
    def __init__(self):
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        self.is_fitted = True
        # TODO: Implement training logic

    def predict(self, X: np.ndarray):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        # TODO: Implement prediction logic
        pass
