
from typing import Optional

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from streamsight.algorithms.base import Algorithm
from streamsight.algorithms.utils import get_top_K_values
from streamsight.matrix import ItemUserBasedEnum


def compute_cosine_similarity(X: csr_matrix) -> csr_matrix:
    """Compute the cosine similarity between the items in the matrix.

    Self similarity is removed.

    :param X: user x item matrix with scores per user, item pair.
    :type X: csr_matrix
    :return: similarity matrix
    :rtype: csr_matrix
    """
    # X.T otherwise we are doing a user KNN
    item_cosine_similarities = cosine_similarity(X.T, dense_output=False)
    item_cosine_similarities.setdiag(0)
    # Set diagonal to 0, because we don't want to support self similarity

    return item_cosine_similarities


class ItemKNN(Algorithm):
    """Item K Nearest Neighbours model.

    First described in 'Item-based top-n recommendation algorithms.' :cite:`10.1145/963770.963776`
    
    This code is adapted from RecPack :cite:`recpack`

    For each item the K most similar items are computed during fit.
    Similarity parameter decides how to compute the similarity between two items.

    Cosine similarity between item i and j is computed as

    .. math::
        sim(i,j) = \\frac{X_i X_j}{||X_i||_2 ||X_j||_2}

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    """
    ITEM_USER_BASED = ItemUserBasedEnum.ITEM
    
    def __init__(
        self,
        K=10
    ):
        super().__init__()
        self.K = K

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item
        We assume that X is a binary matrix of shape (n_users, n_items)
        """
        item_similarities = compute_cosine_similarity(X)
        item_similarities = get_top_K_values(item_similarities, K=self.K)

        self.similarity_matrix_ = item_similarities

    def _predict(self, X: csr_matrix, predict_frame:Optional[pd.DataFrame]=None) -> csr_matrix:
        """Predict scores for nonzero users in X

        Scores are computed by matrix multiplication of X
        with the stored similarity matrix.

        :param X: csr_matrix with interactions
        :type X: csr_matrix
        :param predict_frame: DataFrame containing the user IDs to predict for
        :type predict_frame: pd.DataFrame, optional
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        scores = X @ self.similarity_matrix_
        return scores