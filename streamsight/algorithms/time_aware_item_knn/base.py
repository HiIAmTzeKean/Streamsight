# Adopted from RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
import time
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from streamsight.algorithms.item_similarity_base import TopKItemSimilarityMatrixAlgorithm
from recpack.algorithms.nearest_neighbour import (
    compute_conditional_probability,
    compute_cosine_similarity,
    compute_pearson_similarity,
)
from recpack.algorithms.time_aware_item_knn.decay_functions import (
    ExponentialDecay,
    LogDecay,
    LinearDecay,
    ConcaveDecay,
    ConvexDecay,
    InverseDecay,
    NoDecay,
)
from streamsight.matrix import InteractionMatrix, Matrix
from recpack.util import get_top_K_values

EPSILON = 1e-13

logger = logging.getLogger(__name__)

class TARSItemKNN(TopKItemSimilarityMatrixAlgorithm):
    """Framework for time aware variants of the ItemKNN algorithm.

    This class was inspired by works from Liu, Nathan N., et al. (2010), Ding et al. (2005) and Lee et al. (2007).

    The framework for these approaches can be summarised as:

    - When training the user interaction matrix is weighted to take into account temporal information.
    - Similarities are computed on this weighted matrix, using various similarity measures.
    - When predicting the interactions are similarly weighted, giving more weight to more recent interactions.
    - Recommendation scores are obtained by multiplying the weighted interaction matrix with
      the previously computed similarity matrix.

    The similarity between items is based on their decayed interaction vectors:

    .. math::

        \\text{sim}(i,j) = s(\\Gamma(A_i), \\Gamma(A_j))

    Where :math:`s` is a similarity function (like ``cosine``),
    :math:`\\Gamma` a decay function (like ``exponential_decay``) and
    :math:`A_i` contains the distances to now from when the users interacted with item `i`,
    if they interacted with the item at all (else the value is 0).

    During computation, 'now' is considered as the maximal timestamp in the matrix + 1.
    As such the age is always a positive non-zero value.

    :param K: How many neigbours to use per item,
        make sure to pick a value below the number of columns of the matrix to fit on.
        Defaults to 200
    :type K: int, Optional
    :param fit_decay: Defines the decay scaling used for decay during model fitting.
        Defaults to `` 1 / (24 * 3600)`` (one day).
    :type fit_decay: float, optional
    :param predict_decay: Defines the decay scaling used for decay during prediction.
        Defaults to ``1 / (24 * 3600)`` (one day).
    :type predict_decay: float, optional
    :param decay_interval: Size of a single time unit in seconds.
        Allows more finegrained parameters for large scale datasets where events are collected over months of data.
        Defaults to 1 (second).
    :type decay_interval: int, optional
    :param similarity: Which similarity measure to use. Defaults to ``"cosine"``.
        ``["cosine", "conditional_probability", "pearson"]`` are supported.
    :type similarity: str, Optional
    :param decay_function: The decay function to use, defaults to ``"exponential"``.
        Supported values are ``["exponential", "log", "linear", "concave", "convex", "inverse"]``
    """

    SUPPORTED_SIMILARITIES = ["cosine", "conditional_probability", "pearson"]
    DECAY_FUNCTIONS = {
        "exponential": ExponentialDecay,
        "log": LogDecay,
        "linear": LinearDecay,
        "concave": ConcaveDecay,
        "convex": ConvexDecay,
        "inverse": InverseDecay,
    }

    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / (24 * 3600),
        predict_decay: float = 1 / (24 * 3600),
        decay_interval: int = 1,
        similarity: str = "cosine",
        decay_function: str = "exponential",
    ):
        # Uses other default parameters for ItemKNN
        super().__init__(K=K)

        if decay_interval <= 0 or type(decay_interval) == float:
            raise ValueError("Parameter decay_interval needs to be a positive integer.")

        self.decay_interval = decay_interval

        if similarity not in self.SUPPORTED_SIMILARITIES:
            raise ValueError(f"Similarity {similarity} is not supported.")
        self.similarity = similarity

        if decay_function not in self.DECAY_FUNCTIONS:
            raise ValueError(f"Decay function {decay_function} is not supported.")

        self.decay_function = decay_function

        # Verify decay parameters
        if self.decay_function in ["exponential", "log", "linear", "concave", "convex"]:
            if fit_decay != 0:
                self.DECAY_FUNCTIONS[decay_function].validate_decay(fit_decay)

            if predict_decay != 0:
                self.DECAY_FUNCTIONS[decay_function].validate_decay(predict_decay)

        self.fit_decay = fit_decay
        self.predict_decay = predict_decay
        self.decay_function = decay_function

    def _get_decay_func(self, decay, max_value):
        if decay == 0:
            return NoDecay()

        elif self.decay_function == "inverse":
            return self.DECAY_FUNCTIONS[self.decay_function]()
        elif self.decay_function in ["exponential", "convex"]:
            return self.DECAY_FUNCTIONS[self.decay_function](decay)
        elif self.decay_function in ["log", "linear", "concave"]:
            return self.DECAY_FUNCTIONS[self.decay_function](decay, max_value)

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Predict scores for nonzero users in X.

        Scores are computed by matrix multiplication of weighted X
        with the stored similarity matrix.

        :param X: csr_matrix with interactions
        :type X: csr_matrix
        :return: csr_matrix with scores
        :rtype: csr_matrix
        """
        X = self._add_decay_to_predict_matrix(X)
        return super()._predict(X)

    def _transform_fit_input(self, X: Matrix) -> InteractionMatrix:
        """Weigh each of the interactions by the decay factor of its timestamp."""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _transform_predict_input(self, X: Matrix) -> InteractionMatrix:
        """Weigh each of the interactions by the decay factor of its timestamp."""
        self._assert_is_interaction_matrix(X)
        self._assert_has_timestamps(X)
        return X

    def _fit(self, X: csr_matrix) -> None:
        """Fit a cosine similarity matrix from item to item."""
        X = self._add_decay_to_fit_matrix(X)

        if self.similarity == "cosine":
            item_similarities = compute_cosine_similarity(X)
        elif self.similarity == "conditional_probability":
            item_similarities = compute_conditional_probability(X)
        elif self.similarity == "pearson":
            item_similarities = compute_pearson_similarity(X)

        item_similarities = get_top_K_values(item_similarities, K=self.K)

        self.similarity_matrix_ = item_similarities

    def _add_decay_to_interaction_matrix(self, X: InteractionMatrix, decay: float) -> csr_matrix:
        """Weigh the interaction matrix based on age of the events.

        If decay is 0, it is assumed to be disabled, and so we just return binary matrix.
        :param X: Interaction matrix.
        :type X: InteractionMatrix
        :return: Weighted csr matrix.
        :rtype: csr_matrix
        """
        timestamp_mat = X.latest_interaction_timestamps_matrix
        
        # To get 'now', we add 1 to the maximal timestamp. This makes sure there are no vanishing zeroes.
        now = timestamp_mat.data.max() + 1
        ages = (now - timestamp_mat.data) / self.decay_interval
        timestamp_mat.data = self._get_decay_func(decay, ages.max())(ages)

        return csr_matrix(timestamp_mat)

    def _add_decay_to_fit_matrix(self, X: InteractionMatrix) -> csr_matrix:
        return self._add_decay_to_interaction_matrix(X, self.fit_decay)

    def _add_decay_to_predict_matrix(self, X: InteractionMatrix) -> csr_matrix:
        return self._add_decay_to_interaction_matrix(X, self.predict_decay)

class TARSItemKNNIncremental(TARSItemKNN):
    def __init__(
        self,
        K: int = 200,
        fit_decay: float = 1 / (24 * 3600),
        predict_decay: float = 1 / (24 * 3600),
        decay_interval: int = 1,
        similarity: str = "cosine",
        decay_function: str = "exponential",
    ):
        super().__init__(
            K=K, fit_decay=fit_decay, predict_decay=predict_decay, similarity=similarity, decay_function=decay_function, decay_interval=decay_interval
        )
        self.historical_data: InteractionMatrix = None
    
    def fit(self, X: InteractionMatrix) -> "TARSItemKNNIncremental":
        start = time.time()
        if self.historical_data is None:
            self.historical_data = X.copy()
        else:
            logger.debug(f"Updating historical data for {self.name}")
            self.historical_data = self.historical_data + X
        end = time.time()
        logger.debug(
            f"Updated historical data for {self.name} - Took {end - start :.3}s"
        )
        super().fit(self.historical_data)
        return self
