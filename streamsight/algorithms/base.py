import logging
import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from streamsight.matrix import (InteractionMatrix, ItemUserBasedEnum,
                                to_csr_matrix)
from streamsight.utils.util import add_rows_to_csr_matrix

logger = logging.getLogger(__name__)


class Algorithm(BaseEstimator,ABC):
    """Base class for all streamsight algorithm implementations."""

    ITEM_USER_BASED: ItemUserBasedEnum

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        """Name of the object's class."""
        return self.__class__.__name__

    @property
    def identifier(self):
        """Identifier of the object.

        Identifier is made by combining the class name with the parameters
        passed at construction time.

        Constructed by recreating the initialisation call.
        Example: `Algorithm(param_1=value)`
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

        Will be called by the `fit` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix to fit the model to
        :type X: csr_matrix
        :raises NotImplementedError: Implement this method in the child class
        """
        raise NotImplementedError("Please implement _fit")

    @abstractmethod
    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Stub for predicting scores to users

        Will be called by the `predict` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix used as input to predict
        :type X: csr_matrix
        :raises NotImplementedError: Implement this method in the child class
        :return: Predictions made for all nonzero users in X
        :rtype: csr_matrix
        """
        raise NotImplementedError("Please implement _predict")

    def _check_fit_complete(self):
        """Helper function to check if model was correctly fitted

        Uses the sklear check_is_fitted function,
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        """
        check_is_fitted(self)

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
            logger.warning(f"{self.name} failed to recommend any items for {len(missing)} users. There are a total of {len(users)} users with history.")

    def _transform_fit_input(self, X: InteractionMatrix) -> csr_matrix:
        """Transform the training data to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix to fit the model to
        :type X: Matrix
        :return: Transformed user-item interaction matrix to fit the model
        :rtype: csr_matrix
        """
        return to_csr_matrix(X, binary=True)

    def _transform_predict_input(self, X: InteractionMatrix) -> csr_matrix:
        """Transform the input of predict to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix used as input to predict
        :type X: Matrix
        :return: Transformed user-item interaction matrix used as input to predict
        :rtype: csr_matrix
        """
        return to_csr_matrix(X, binary=True)

    def fit(self, X: InteractionMatrix) -> "Algorithm":
        """Fit the model to the input interaction matrix.

        The input data is transformed to the expected type using
        :meth:`_transform_fit_input`. The fitting is done using the
        :meth:`_fit` method. Finally the method checks that the fitting
        was successful using :meth:`_check_fit_complete`.

        :param X: The interactions to fit the model on.
        :type X: Matrix
        :return: **self**, fitted algorithm
        :rtype: Algorithm
        """
        start = time.time()
        X = self._transform_fit_input(X)
        self._fit(X)

        self._check_fit_complete()
        end = time.time()
        logger.info(f"Fitting {self.name} complete - Took {end - start :.3}s")
        return self

    def predict(self, X: InteractionMatrix) -> csr_matrix:
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

        # X will contain past sequential interaction and IDs to predict
        to_predict_frame = X.get_prediction_data()
        prev_interaction = X.get_interaction_data()
        prev_interaction = self._transform_predict_input(prev_interaction)

        X_pred = self._predict(prev_interaction)
        known_shape = X_pred.shape
        logger.debug("Predictions by algorithm completed")

        # ID indexing starts at 0, so max_id + 1 is the number of unique IDs
        max_user_id  = to_predict_frame.max_user_id + 1 
        max_item_id  = to_predict_frame.max_item_id + 1 
        intended_shape = (max(max_user_id, X.shape[0]), max(max_item_id, X.shape[1]))
        # ? did not add col which represents the unknown items
        X_pred = add_rows_to_csr_matrix(X_pred, intended_shape[0]-known_shape[0])
        # pad users with random items
        logger.debug(f"Padding user ID in range({known_shape[0]}, {intended_shape[0]}) with random items")
        row = []
        col = []
        for user_id in to_predict_frame.user_ids:
            if user_id >= known_shape[0]:
                row.append(user_id)
                col.append(np.random.randint(0, X_pred.shape[1]))
        pad = csr_matrix((np.ones(len(row)), (row, col)), shape=X_pred.shape)
        X_pred += pad
        logger.debug(f"Padding completed")
        return X_pred
