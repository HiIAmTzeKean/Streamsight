import logging
from typing import Dict, Optional, Tuple
from warnings import warn

import numpy as np
from scipy.sparse import csr_matrix, vstack

from streamsight.algorithms.utils import get_top_K_ranks
from streamsight.utils.util import add_columns_to_csr_matrix


logger = logging.getLogger(__name__)


class Metric:
    """Base class for all metrics.

    A Metric object is stateful, i.e. after ``calculate``
    the results can be retrieved in one of two ways:
      - Detailed results are stored in :attr:`results`,
      - Aggregated result value can be retrieved using :attr:`value`
    """

    def __init__(self,
                 timestamp_limit: Optional[int] = None,
                 cache: bool = False):
        self._num_users = 0
        self._num_items = 0
        self._timestamp_limit = timestamp_limit
        self.cache = cache
        
        self._scores: csr_matrix
        self._value: float
        self._y_true: csr_matrix
        self._y_pred: csr_matrix
        
    @property
    def name(self):
        """Name of the metric."""
        return self.__class__.__name__
    
    @property
    def params(self):
        """Parameters of the metric."""
        return {"timestamp_limit": self._timestamp_limit}
    
    def get_params(self):
        """Get the parameters of the metric."""
        return self.params
    
    @property
    def identifier(self):
        """Name of the metric."""
        # return f"{super().identifier[:-1]},K={self.K})"
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items() if v is not None))
        return self.__class__.__name__ + "(" + paramstring + ")"

    def _calculate(self, y_true, y_pred) -> None:
        raise NotImplementedError()

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Calculates this metric for all nonzero users in ``y_true``,
        given true labels and predicted scores.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        """
        y_true, y_pred = self._eliminate_empty_users(y_true, y_pred)
        self._verify_shape(y_true, y_pred)
        self._set_shape(y_true)
        self._calculate(y_true, y_pred)

    def cache_values(self, y_true:csr_matrix, y_pred:csr_matrix):
        if not self.cache:
            raise ValueError("Caching is disabled for this metric.")
        
        if not hasattr(self, "_y_true") or not hasattr(self, "_y_pred"):
            self._y_true = y_true
            self._y_pred = y_pred
            return
        
        # reshape old y_true and y_pred to add the new columns
        if y_true.shape[1] > self._y_true.shape[1]:
            self._y_true = add_columns_to_csr_matrix(self._y_true, y_true.shape[1]-self._y_true.shape[1])
            self._y_pred = add_columns_to_csr_matrix(self._y_pred, y_pred.shape[1]-self._y_pred.shape[1])
            
        #? np.vstack([self._y_true.toarray(), y_true.toarray()]) faster ?
        self._y_true = vstack([self._y_true, y_true])
        self._y_pred = vstack([self._y_pred, y_pred])
    
    def calculate_cached(self):
        if not self.cache:
            raise ValueError("Caching is disabled for this metric.")
        if not hasattr(self, "_y_true") or not hasattr(self, "_y_pred"):
            self._scores = None
            # raise AttributeError("No cached values found. Call cache_values() first.")
            return
        
        self.calculate(self._y_true, self._y_pred)

    @property
    def micro_result(self) -> Dict[str, np.ndarray]:
        """Micro results for the metric.
        
        :return: Detailed results for the metric.
        :rtype: Dict[str, np.ndarray]
        """
        return {"score": np.array(self.macro_result)}

    @property
    def macro_result(self) -> float:
        """The global metric value."""
        if not hasattr(self, "_value"):
            raise ValueError("Metric has not been calculated yet.")
        return self._value

    @property
    def timestamp_limit(self):
        """The timestamp limit for the metric."""
        return self._timestamp_limit

    @property
    def num_items(self) -> int:
        """Dimension of the item-space in both ``y_true`` and ``y_pred``"""
        return self._num_items

    @property
    def num_users(self) -> int:
        """Dimension of the user-space in both ``y_true`` and ``y_pred``
        after elimination of users without interactions in ``y_true``.
        """
        return self._num_users

    @property
    def _indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Indices in the prediction matrix for which scores were computed."""
        row, col = np.indices((self._num_users, self._num_items))

        return row.flatten(), col.flatten()

    def _verify_shape(self, y_true: csr_matrix, y_pred: csr_matrix) -> bool:
        """Make sure the dimensions of y_true and y_pred match.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :raises AssertionError: Shape mismatch between y_true and y_pred.
        :return: True if dimensions match.
        :rtype: bool
        """
        check = y_true.shape == y_pred.shape
        if not check:
            raise AssertionError(f"Shape mismatch between y_true: {y_true.shape} and y_pred: {y_pred.shape}")
        return check

    def _set_shape(self, y_true: csr_matrix) -> None:
        self._num_users, self._num_items = y_true.shape

    def _eliminate_empty_users(self, y_true: csr_matrix, y_pred: csr_matrix) -> Tuple[csr_matrix, csr_matrix]:
        """Eliminate users that have no interactions in ``y_true``.

        We cannot make accurate predictions of interactions for
        these users as there are none.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        :return: (y_true, y_pred), with zero users eliminated.
        :rtype: Tuple[csr_matrix, csr_matrix]
        """
        # Get the rows (users) that are not empty
        nonzero_users = list(set(y_true.nonzero()[0]))

        self.user_id_map_ = np.array(nonzero_users)

        return y_true[nonzero_users, :], y_pred[nonzero_users, :]

    def _map_users(self, users):
        """Map internal identifiers of users to actual user identifiers."""
        if hasattr(self, "user_id_map_"):
            return self.user_id_map_[users]
        else:
            return users


class MetricTopK(Metric):
    """Base class for all metrics computed based on the Top-K recommendations for every user.

    A MetricTopK object is stateful, i.e. after ``calculate``
    the results can be retrieved in one of two ways:
      - Detailed results are stored in :attr:`results`,
      - Aggregated result value can be retrieved using :attr:`value`

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K:int = 10,
                 timestamp_limit: Optional[int] = None,
                 cache: bool = False):
        super().__init__(timestamp_limit, cache)
        self.K = K

    @property
    def name(self):
        """Name of the metric."""
        return f"{super().name}_{self.K}"
    
    @property
    def params(self):
        """Parameters of the metric."""
        return super().params | {"K": self.K}

    @property
    def _indices(self):
        """Indices in the prediction matrix for which scores were computed."""
        row, col = self.y_pred_top_K_.nonzero()
        return row, col

    def _calculate(self, y_true:csr_matrix, y_pred_top_K:csr_matrix):
        """Computes metric given true labels ``y_true`` and predicted scores ``y_pred``. Only Top-K recommendations are considered.

        To be implemented in the child class.

        :param y_true: Expected interactions per user.
        :type y_true: csr_matrix
        :param y_pred_top_K: Ranks for topK recommendations per user
        :type y_pred_top_K: csr_matrix
        """
        raise NotImplementedError()

    def calculate(self, y_true: csr_matrix, y_pred: csr_matrix) -> None:
        """Computes metric given true labels ``y_true`` and predicted scores ``y_pred``. Only Top-K recommendations are considered.

        Detailed metric results can be retrieved with :attr:`results`.
        Global aggregate metric value is retrieved as :attr:`value`.

        :param y_true: True user-item interactions.
        :type y_true: csr_matrix
        :param y_pred: Predicted affinity of users for items.
        :type y_pred: csr_matrix
        """
        # Perform checks and cleaning
        y_true, y_pred = self._eliminate_empty_users(y_true, y_pred)
        self._verify_shape(y_true, y_pred)
        self._set_shape(y_true)

        # Compute the topK for the predicted affinities
        y_pred_top_K = get_top_K_ranks(y_pred, self.K)
        self.y_pred_top_K_ = y_pred_top_K

        # Compute the metric.
        self._calculate(y_true, y_pred_top_K)


class ElementwiseMetricK(MetricTopK):
    """Base class for all metrics that can be calculated for
    each user-item pair in the Top-K recommendations.

    :attr:`results` contains an entry for each user-item pair.

    Examples are: HitK, IPSHitRateK

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    @property
    def col_names(self):
        """The names of the columns in the results DataFrame."""
        return ["user_id", "item_id", "score"]

    @property
    def micro_result(self) -> dict[str, np.ndarray]:
        """Get the detailed results for this metric.

        Contains an entry for every user-item pair in the Top-K recommendations list of every user.

        If there is a user with no recommendations,
        the results DataFrame will contain K rows for
        that user with item_id = NaN and score = 0.

        :return: The results DataFrame with columns: user_id, item_id, score
        :rtype: pd.DataFrame
        """
        if not hasattr(self, "_scores"):
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            warn(UserWarning("No scores were computed. Returning empty dict."))
            return dict(zip(self.col_names, (np.array([]), np.array([]), np.array([]))))
        
        scores = self._scores.toarray()

        all_users = set(range(self._scores.shape[0]))
        int_users, items = self._indices
        values = scores[int_users, items]

        # For all users in all_users but not in int_users,
        # add K items np.nan with value = 0
        missing_users = all_users.difference(set(int_users))

        # This should barely occur, so it's not too bad to append np arrays.
        for u in list(missing_users):
            for i in range(self.K):
                int_users = np.append(int_users, u)
                values = np.append(values, 0)
                items = np.append(items, np.nan)

        users = self._map_users(int_users)

        return dict(zip(self.col_names, (users, items, values)))

    @property
    def macro_result(self):
        """Global metric value obtained by summing up scores for every user then taking the average over all users."""
        return self._scores.sum(axis=1).mean()


class ListwiseMetricK(MetricTopK):
    """Base class for all metrics that can be calculated for every Top-K recommendation list,
    i.e. one value for each user.
    Examples are: DiversityK, NDCGK, ReciprocalRankK, RecallK

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    @property
    def col_names(self):
        """The names of the columns in the results DataFrame."""
        return ["user_id", "score"]

    @property
    def _indices(self):
        """Indices in the prediction matrix for which scores were computed."""
        row = np.arange(self.y_pred_top_K_.shape[0])
        col = np.zeros(self.y_pred_top_K_.shape[0], dtype=np.int32)
        return row, col

    @property
    def micro_result(self) -> dict[str, np.ndarray]:
        """Get the detailed results for this metric.

        Contains an entry for every user.

        :return: The results DataFrame with columns: user_id, score
        :rtype: pd.DataFrame
        """
        if not hasattr(self, "_scores"):
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            warn(UserWarning("No scores were computed. Returning empty dict."))
            return dict(zip(self.col_names, (np.array([]), np.array([]))))
    
        scores = self._scores.toarray()

        int_users, items = self._indices
        values = scores[int_users, items]

        users = self._map_users(int_users)

        return dict(zip(self.col_names, (users, values)))
    

    @property
    def macro_result(self):
        """Global metric value obtained by taking the average over all users."""
        if not hasattr(self, "_scores"):
            raise ValueError("Metric has not been calculated yet.")
        elif self._scores is None:
            warn(UserWarning("No scores were computed. Returning None."))
            return None
        
        return self._scores.mean()
