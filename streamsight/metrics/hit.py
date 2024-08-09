import logging
from typing import Optional

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from streamsight.metrics.base import ElementwiseMetricK


logger = logging.getLogger(__name__)


class HitK(ElementwiseMetricK):
    """Computes the number of hits in a list of Top-K recommendations.

    A hit is counted when a recommended item in the top K for this user was interacted with.

    Detailed :attr:`results` show which of the items in the list of Top-K recommended items
    were hits and which were not.

    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def __init__(self, K=10, timestamp_limit: Optional[int] = None):
        super().__init__(K,timestamp_limit)

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:

        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1

        scores = scores.tocsr()

        self._scores = scores