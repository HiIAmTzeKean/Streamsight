# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert
from typing import Optional
import warnings

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from streamsight.algorithms.base import Algorithm
from streamsight.utils.util import get_top_K_values

class Random(Algorithm):
    """Uniform random algorithm, each item has an equal chance of getting recommended.

    Simple baseline, recommendations are sampled uniformly without replacement
    from the items that were interacted with in the matrix provided to fit.
    Scores are given based on sampling rank, such that the items first
    in the sample has the highest score

    :param K: How many items to sample for recommendation, defaults to 200
    :type K: int, optional
    :param seed: Seed for the random number generator used, defaults to None
    :type seed: int, optional
    :param use_only_interacted_items: Should only items visited in the training
        matrix be used to recommend from. If False all items will be recommended
        uniformly at random.
        Defaults to True.
    :type use_only_interacted_items: boolean, optional
    """

    def __init__(self, K: Optional[int] = 200, seed: Optional[int] = None, use_only_interacted_items: bool = True):
        super().__init__()
        self.items = None
        self.K = K  # TODO Allow K to be set to zero?
        self.use_only_interacted_items = use_only_interacted_items

        if seed is None:
            seed = np.random.get_state()[1][0]
        self.seed = seed
        self.rand_gen = np.random.default_rng(seed=self.seed)

    def _fit(self, X: csr_matrix) -> "Random":
        if self.use_only_interacted_items:
            self.items_ = list(set(X.nonzero()[1]))
        else:
            self.items_ = list(np.arange(X.shape[1]))

        if self.K is not None and len(self.items_) < self.K:
            warnings.warn("K is larger than the number of items.", UserWarning)

        return self

    def _predict(self, X: csr_matrix) -> csr_matrix:
        # For each user choose random K items, and generate a score for these items
        # Then create a matrix with the scores on the right indices
        users = list(set(X.nonzero()[0]))

        num_items = X.shape[1]
        K = min(len(self.items_), self.K) if self.K is not None else self.K
        # Generate random scores for all items
        random_scores = self.rand_gen.random((len(users), num_items))

        # Filter out only allowed items
        allowed_items = np.zeros(num_items)
        allowed_items[self.items_] = 1
        # Get top K of allowed items per user
        top_scores = get_top_K_values(csr_matrix(random_scores * allowed_items), K=K)

        X_pred = csr_matrix(X.shape)
        X_pred[users] = top_scores

        return X_pred