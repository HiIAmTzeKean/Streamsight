import logging
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from streamsight.evaluators.accumulator import MetricAccumulator
from streamsight.evaluators.util import MetricLevelEnum, UserItemBaseStatus
from streamsight.registries import MetricEntry
from streamsight.settings import Setting

logger = logging.getLogger(__name__)

class EvaluatorBase(object):
    """Base class for evaluator.
    
    :param metric_entries: List of metric entries to compute
    :type metric_entries: List[MetricEntry]
    :param setting: Setting object
    :type setting: Setting
    :param ignore_unknown_user: Ignore unknown users, defaults to True
    :type ignore_unknown_user: bool, optional
    :param ignore_unknown_item: Ignore unknown items, defaults to True
    :type ignore_unknown_item: bool, optional
    """
    def __init__(
        self,
        metric_entries: List[MetricEntry],
        setting: Setting,
        metric_k: int,
        ignore_unknown_user: bool = True,
        ignore_unknown_item: bool = True,
        seed: Optional[int] = None
    ):
        self.metric_entries = metric_entries
        self.setting = setting
        """Setting to evaluate the algorithms on."""
        self.metric_k = metric_k
        """Value of K for the metrics."""
        self.ignore_unknown_user = ignore_unknown_user
        """To ignore unknown users during evaluation."""
        self.ignore_unknown_item = ignore_unknown_item
        """To ignore unknown items during evaluation."""

        self._acc: MetricAccumulator
        self.user_item_base = UserItemBaseStatus()

        self.ignore_unknown_user = ignore_unknown_user
        self.ignore_unknown_item = ignore_unknown_item

        if not seed:
            seed = 42
        self.seed = seed

        self._run_step = 0
        self._current_timestamp: int

    def _prediction_shape_handler(
        self, X_true_shape: Tuple[int, int], X_pred: csr_matrix
    ) -> csr_matrix:
        """Handle shape difference of the prediction matrix.

        If there is a difference in the shape of the prediction matrix and the
        ground truth matrix, this function will handle the difference based on
        :attr:`ignore_unknown_user` and :attr:`ignore_unknown_item`.

        :param X_true_shape: Shape of the ground truth matrix
        :type X_true_shape: Tuple[int,int]
        :param X_pred: Prediction matrix
        :type X_pred: csr_matrix
        :raises ValueError: If the user dimension of the prediction matrix is less than the ground truth matrix
        :return: Prediction matrix with the same shape as the ground truth matrix
        :rtype: csr_matrix
        """
        if X_pred.shape != X_true_shape:
            # We cannot expect the algorithm to predict an unknown item, so we
            # only check user dimension
            if (
                X_pred.shape[0] < X_true_shape[0]
                and not self.ignore_unknown_user
            ):
                raise ValueError(
                    "Prediction matrix shape, user dimension, is less than the ground truth matrix shape."
                )

            if not self.ignore_unknown_item:
                # prediction matrix would not contain unknown item ID
                # update the shape of the prediction matrix to include the ID
                X_pred = csr_matrix(
                    (X_pred.data, X_pred.indices, X_pred.indptr),
                    shape=(X_pred.shape[0], X_true_shape[1]),
                )

            # shapes might not be the same in the case of dropping unknowns
            # from the ground truth data. We ensure that the same unknowns
            # are dropped from the predictions
            if self.ignore_unknown_user:
                X_pred = X_pred[: X_true_shape[0], :]
            if self.ignore_unknown_item:
                X_pred = X_pred[:, : X_true_shape[1]]

        return X_pred

    def metric_results(
        self,
        level: Union[
            MetricLevelEnum, Literal["macro", "micro", "window", "user"]
        ] = MetricLevelEnum.MACRO,
        only_current_timestamp: Optional[bool] = False,
        filter_timestamp: Optional[int] = None,
        filter_algo: Optional[str] = None,
    ) -> pd.DataFrame:
        """Results of the metrics computed.

        Computes the metrics of all algorithms based on the level specified and
        return the results in a pandas DataFrame. The results can be filtered
        based on the algorithm name and the current timestamp.

        Specifics
        ---------
        - User level: User level metrics computed across all timestamps.
        - Window level: Window level metrics computed across all timestamps. This can
          be viewed as a macro level metric in the context of a single window, where
          the scores of each user is averaged within the window.
        - Macro level: Macro level metrics computed for entire timeline. This
          score is computed by averaging the scores of all windows, treating each
          window equally.
        - Micro level: Micro level metrics computed for entire timeline. This
          score is computed by averaging the scores of all users, treating each
          user and the timestamp the user is in as unique contribution to the
          overall score.

        :param level: Level of the metric to compute, defaults to "macro"
        :type level: Union[MetricLevelEnum, Literal["macro", "micro", "window", "user"]]
        :param only_current_timestamp: Filter only the current timestamp, defaults to False
        :type only_current_timestamp: bool, optional
        :param filter_timestamp: Timestamp value to filter on, defaults to None.
            If both `only_current_timestamp` and `filter_timestamp` are provided,
            `filter_timestamp` will be used.
        :type filter_timestamp: Optional[int], optional
        :param filter_algo: Algorithm name to filter on, defaults to None
        :type filter_algo: Optional[str], optional
        :return: Dataframe representation of the metric
        :rtype: pd.DataFrame
        """
        if not MetricLevelEnum.has_value(level):
            raise ValueError("Invalid level specified")
        level = MetricLevelEnum(level)

        timestamp = None
        if only_current_timestamp:
            timestamp = self._current_timestamp

        if filter_timestamp:
            timestamp = filter_timestamp

        return self._acc.df_metric(filter_algo=filter_algo,
                             filter_timestamp=timestamp,
                             level=level)
