from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from recpack.algorithms.base import Algorithm

class DecayPopularity(Algorithm):
    """A popularity-based algorithm with exponential decay over data from earlier time windows.

    :param K: Number of items to recommend, defaults to 200
    :type K: int, optional
    """

    def __init__(self, K: int = 200):
        super().__init__()
        self.K = K
        self.historical_data: list[csr_matrix] = []  # Store all historical training data

    def _fit(self, X: csr_matrix) -> "DecayPopularity":
        """
        Fit the model by applying decay to historical data and adding new data.

        :param X: Interaction matrix (users x items) for the current window
        :type X: csr_matrix
        """
        # Append new data to historical data
        self.historical_data.append(X)

        # Initialize decayed scores
        num_items = X.shape[1]
        if num_items < self.K:
            warn("K is larger than the number of items.", UserWarning)

        decayed_scores = np.zeros(num_items)

        # Apply decay to each historical matrix
        for i, matrix in enumerate(self.historical_data):
            # length 2, i = 0 -> 2-1-0 = 1, i = 1 -> 2-1-1 = 0
            # length 3, i = 0 -> 3-1-0 = 2, i = 1 -> 3-1-1 = 1, i = 2 -> 3-1-2 = 0
            decay_factor = np.exp(-(len(self.historical_data) - 1 - i))
            decayed_scores += matrix.sum(axis=0).A[0] * decay_factor

        normalized_scores = decayed_scores / decayed_scores.max()
        
        K = min(self.K, num_items)
        ind = np.argpartition(normalized_scores, -K)[-K:]
        a = np.zeros(num_items)
        a[ind] = normalized_scores[ind]
        self.decayed_scores_ = a
        return self

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """
        Predict the K most popular item for each user scaled by the decay factor.
        """
        users = list(set(X.nonzero()[0]))
        X_pred = lil_matrix(X.shape)
        X_pred[users] = self.decayed_scores_
        return X_pred.tocsr()
    