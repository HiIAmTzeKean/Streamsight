# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import warnings
from typing import Optional

import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from streamsight.algorithms.base import Algorithm
from streamsight.utils.util import get_top_K_values

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

    First described in 'Item-based top-n recommendation algorithms.'
    Deshpande, Mukund, and George Karypis,
    ACM Transactions on Information Systems (TOIS) 22.1 (2004): 143-177

    For each item the K most similar items are computed during fit.
    Similarity parameter decides how to compute the similarity between two items.
    Supported options are: ``"cosine"`` and ``"conditional_probability"``

    Cosine similarity between item i and j is computed as

    .. math::
        sim(i,j) = \\frac{X_i X_j}{||X_i||_2 ||X_j||_2}

    The conditional probablity based similarity of item i with j is computed as

    .. math ::
        sim(i,j) = \\frac{\\sum\\limits_{u \\in U} \\mathbb{I}_{u,i} X_{u,j}}{Freq(i) \\times Freq(j)^{\\alpha}}

    Where I_ui is 1 if the user u has visited item i, and 0 otherwise.
    And alpha is the pop_discount parameter.
    Note that this is a non-symmetric similarity measure.
    Given that X is a binary matrix, and alpha is set to 0, this simplifies to pure conditional probability.

    .. math::
        sim(i,j) = \\frac{Freq(i \\land j)}{Freq(i)}

    If sim_normalize is True, the scores are normalized per predictive item,
    making sure the sum of each row in the similarity matrix is 1.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, optional
    :param similarity: Which similarity measure to use,
        can be one of ["cosine", "conditional_probability"], defaults to "cosine"
    :type similarity: str, optional
    :param pop_discount: Power applied to the comparing item in the denominator,
        to discount contributions of very popular items.
        Should be between 0 and 1. If None, apply no discounting.
        Defaults to None.
    :type pop_discount: float, optional
    :param normalize_X: Normalize rows in the interaction matrix so that
        the contribution of users who have viewed more items is smaller,
        defaults to False
    :type normalize_X: bool, optional
    :param normalize_sim: Normalize scores per row in the similarity matrix to
        counteract artificially large similarity scores when the predictive item is
        rare, defaults to False.
    :type normalize_sim: bool, optional
    :raises ValueError: If an unsupported similarity measure is passed.
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability"]
    """The supported similarity options"""

    def __init__(
        self,
        K=200,
        normalize_X: bool = False,
        normalize_sim: bool = False
    ):
        super().__init__()
        self.K = K
        
        self.normalize_X = normalize_X
        # Sim_normalize takes precedence.
        self.normalize_sim = normalize_sim


    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item"""

        transformer = Normalizer(norm="l1", copy=False)

        if self.normalize_X:
            X = transformer.transform(X)

        item_similarities = compute_cosine_similarity(X)

        item_similarities = get_top_K_values(item_similarities, K=self.K)

        # j, M (*, j) = 1
        if self.normalize_sim:
            # Normalize such that sum per row = 1
            item_similarities = transformer.transform(item_similarities)

        self.similarity_matrix_ = item_similarities

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for nonzero users in X

        Scores are computed by matrix multiplication of X
        with the stored similarity matrix.

        :param X: csr_matrix with interactions
        :type X: csr_matrix
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        scores = X @ self.similarity_matrix_

        # If self.similarity_matrix_ is not a csr matrix,
        # scores will also not be a csr matrix
        if not isinstance(scores, csr_matrix):
            scores = csr_matrix(scores)

        return scores