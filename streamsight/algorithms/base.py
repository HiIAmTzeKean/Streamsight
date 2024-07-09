from abc import ABC, abstractmethod
import logging
import time
from typing import Union
import warnings

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from streamsight.matrix.interation_matrix import InteractionMatrix
from streamsight.matrix.util import to_csr_matrix

logger = logging.getLogger(__name__)
Matrix = Union[InteractionMatrix, csr_matrix]


class Algorithm(BaseEstimator,ABC):
    """Base class for all streamsight algorithm implementations."""

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        """Name of the object's class."""
        return self.__class__.__name__

    @property
    def identifier(self):
        """Name of the object.

        Name is made by combining the class name with the parameters
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

    def fit(self, X: Matrix) -> "Algorithm":
        """Fit the model to the input interaction matrix.

        After fitting the model will be ready to use for prediction.

        This function will handle some generic bookkeeping
        for each of the child classes,

        - The fit function gets timed, and this will get printed
        - Input data is converted to expected type using call to
          :meth:`_transform_predict_input`
        - The model is trained using the :meth:`_fit` method
        - :meth:`_check_fit_complete` is called to check fitting was succesful

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

    def predict(self, X: Matrix) -> csr_matrix:
        """Predicts scores, given the interactions in X

        Recommends items for each nonzero user in the X matrix.

        This function is a wrapper around the :meth:`_predict` method,
        and performs checks on in- and output data to guarantee proper computation.

        - Checks that model is fitted correctly
        - checks the output using :meth:`_check_prediction` function

        :param X: interactions to predict from.
        :type X: Matrix
        :return: The recommendation scores in a sparse matrix format.
        :rtype: csr_matrix
        """
        self._check_fit_complete()

        X = self._transform_predict_input(X)

        X_pred = self._predict(X)

        self._check_prediction(X_pred, X)

        return X_pred