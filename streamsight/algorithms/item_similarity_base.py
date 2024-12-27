import logging
import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from streamsight.matrix import InteractionMatrix, to_csr_matrix, Matrix

from warnings import warn

logger = logging.getLogger(__name__)

class ItemSimilarityBaseAlgorithm(BaseEstimator, ABC):
    """Base class for all item similarity algorithms.

    Normally new algorithms only have to implement the _predict methods.
    """

    def __init__(self):
        super().__init__()
        if not hasattr(self, "seed"):
            self.seed = 42
        self.rand_gen = np.random.default_rng(seed=self.seed)

    @property
    def name(self):
        """Name of the object's class.
        
        :return: Name of the object's class
        :rtype: str
        """
        return self.__class__.__name__

    @property
    def identifier(self):
        """Identifier of the object.

        Identifier is made by combining the class name with the parameters
        passed at construction time.

        Constructed by recreating the initialisation call.
        Example: `Algorithm(param_1=value)`
        
        :return: Identifier of the object
        :rtype: str
        """
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"

    def __str__(self):
        return self.name

    def set_params(self, **params):
        """Set the parameters of the estimator.

        :param params: Estimator parameters
        :type params: dict
        """
        super().set_params(**params)

    @abstractmethod
    def _fit(self, X: csr_matrix):
        """Stub implementation for fitting an algorithm.

        Will be called by the :meth:`fit` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix to fit the model to
        :type X: csr_matrix
        :raises NotImplementedError: Implement this method in the child class
        """
        raise NotImplementedError("Please implement _fit")

    @abstractmethod
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

    def _check_fit_complete(self):
        """Helper function to check if model was correctly fitted

        Uses the sklearn check_is_fitted function,
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        """
        check_is_fitted(self)

        # Additional checks on the fitted matrix.
        # Check if actually exists!
        assert hasattr(self, "similarity_matrix_")

        # Check row wise, since that will determine the recommendation options.
        items_with_score = set(self.similarity_matrix_.nonzero()[0])
        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            warn(f"{self.name} missing similar items for {missing} items.")

    def _check_prediction(self, X_pred: csr_matrix, X: csr_matrix) -> None:
        """Checks that the predictions matches expectations

        Checks implemented:

        - Check that all users with history got at least 1 recommendation

        For failing checks a warning is printed.

        :param X_pred: Predictions made for all nonzero users in X
        :type X_pred: csr_matrix
        :param X: User-item interaction matrix used as input to predict
        :type X: csr_matrix
        """

        users = set(X.nonzero()[0])
        predicted_users = set(X_pred.nonzero()[0])
        missing = users.difference(predicted_users)
        if len(missing) > 0:
            warn(f"{self.name} failed to recommend any items " f"for {len(missing)} users")

    def _transform_fit_input(self, X: Matrix) -> csr_matrix:
        """Transform the training data to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix to fit the model to
        :type X: Matrix
        :return: Transformed user-item interaction matrix to fit the model
        :rtype: csr_matrix
        """
        return to_csr_matrix(X, binary=True)

    def _transform_predict_input(self, X: Matrix) -> csr_matrix:
        """Transform the input of predict to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix used as input to predict
        :type X: Matrix
        :return: Transformed user-item interaction matrix used as input to predict
        :rtype: csr_matrix
        """
        return to_csr_matrix(X, binary=True)

    def _assert_is_interaction_matrix(self, *matrices: Matrix) -> None:
        """Make sure that the passed matrices are all an InteractionMatrix."""
        for X in matrices:
            if type(X) != InteractionMatrix:
                raise TypeError(f"{self.name} requires Interaction Matrix as input. Got {type(X)}.")

    def _assert_has_timestamps(self, *matrices: InteractionMatrix):
        """Make sure that the matrices all have timestamp information."""
        for X in matrices:
            if not X.has_timestamps:
                raise ValueError(f"{self.name} requires timestamp information in the InteractionMatrix.")

    def fit(self, X: Matrix) -> "ItemSimilarityBaseAlgorithm":
        """Fit the model to the input interaction matrix.

        The input data is transformed to the expected type using
        :meth:`_transform_fit_input`. The fitting is done using the
        :meth:`_fit` method. Finally the method checks that the fitting
        was successful using :meth:`_check_fit_complete`.

        :param X: The interactions to fit the model on.
        :type X: InteractionMatrix
        :return: Fitted item similarity base algorithm
        :rtype: ItemSimilarityBaseAlgorithm
        """
        start = time.time()
        X = self._transform_fit_input(X)
        self._fit(X)

        self._check_fit_complete()
        end = time.time()
        logger.info(f"Fitting {self.name} complete - Took {end - start :.3}s")
        return self

    def predict(self, X: Matrix) -> csr_matrix:
        """Predicts scores, given the interactions in X

        The input data is transformed to the expected type using
        :meth:`_transform_predict_input`. The predictions are made
        using the :meth:`_predict` method. Finally the predictions
        are then padded with random items for users that are not in the
        training data.

        :param X: interactions to predict from.
        :type X: InteractionMatrix
        :return: The recommendation scores in a sparse matrix format.
        :rtype: csr_matrix
        """
        self._check_fit_complete()

        X = self._transform_predict_input(X)

        X_pred = self._predict(X)

        self._check_prediction(X_pred, X)

        return X_pred
    

class TopKItemSimilarityMatrixAlgorithm(ItemSimilarityBaseAlgorithm):
    """Base algorithm for algorithms that fit an item to item similarity model with K similar items for every item

    Model that encodes the similarity between items is expected
    under the ``similarity_matrix_`` attribute.

    This matrix should have shape ``(|items| x |items|)``.
    This can be dense or sparse matrix depending on the algorithm used.

    Predictions are made by computing the dot product of the history vector of a user
    and the similarity matrix.

    Normally new algorithms only have to implement the _predict methods.
    to construct the `self.similarity_matrix_` attribute.

    :param K: How many similar items will be kept per item.
    :type K: int
    """

    def __init__(self, K):
        super().__init__()
        self.K = K