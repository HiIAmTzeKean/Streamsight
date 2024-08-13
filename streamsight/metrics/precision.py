import logging
from typing import Optional

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from streamsight.metrics.base import ListwiseMetricK

logger = logging.getLogger(__name__)


class PrecisionK(ListwiseMetricK):
    """Computes the fraction of top-K recommendations that correspond
    to true interactions.

    Different from the definition for information retrieval
    a recommendation algorithm is expected to always return K items
    when the Top-K recommendations are requested.
    When fewer than K items received scores, these are considered a miss.
    As such recommending fewer items is not beneficial for a
    recommendation algorithm.

    Precision is computed per user as:

    .. math::

        \\text{Precision}(u) = \\frac{\\sum\\limits_{i \\in \\text{Top-K}(u)} y^{true}_{u,i}}{K}\\
    
    ref: RecPack
    
    :param K: Size of the recommendation list consisting of the Top-K item predictions.
    :type K: int
    """

    def _calculate(self, y_true: csr_matrix, y_pred_top_K: csr_matrix) -> None:
        scores = scipy.sparse.lil_matrix(y_pred_top_K.shape)

        # Elementwise multiplication of top K predicts and true interactions
        scores[y_pred_top_K.multiply(y_true).astype(bool)] = 1
        # ? csr to lil is costly, other alternatives?
        scores = scores.tocsr()

        self._scores = csr_matrix(scores.sum(axis=1)) / self.K
        
        logger.debug(f"Precision compute complete - {self.name}")
